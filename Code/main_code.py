
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from tqdm import tqdm
import os
from torchvision.transforms import v2
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights, DenseNet161_Weights, VGG16_BN_Weights, VGG19_BN_Weights, ResNet101_Weights, ResNet18_Weights
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
import glob
import subprocess
import sys
try:
    from g_mlp_pytorch import gMLPVision
except:
    subprocess.run(['pip', 'install', 'g-mlp-pytorch'])
import warnings
warnings.filterwarnings('ignore')

SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

n_epoch = 20
BATCH_SIZE = 64
LR =  0.001

## Image processing
CHANNELS = 1
IMAGE_SIZE = 224

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SAVE_MODEL = True

def download_dataset():
    # Ensure the .kaggle directory exists
    os.makedirs(os.path.join(os.path.expanduser('~'), '.kaggle'), exist_ok=True)

    # Download the dataset from Kaggle
    subprocess.run(['kaggle', 'datasets', 'download', '-d', 'ninadaithal/imagesoasis', '--unzip', '-p', 'data/'])
    
    # Update subfolder names
    name_dict = {'Non Demented':'Non_Demented',
                'Very mild Dementia':'Very_mild_Dementia',
                'Mild Dementia':'Mild_Dementia',
                'Moderate Dementia':'Moderate_Dementia'}
    
    for key,value in name_dict.items():
        subprocess.run(['mv', f'data/Data/{key}', f'data/Data/{value}'])

    # Confirmation message
    print("Download completed and files are extracted to the 'data/' directory.")


def download_metadata():
    # Create metadata folder
    directory = 'metadata' 
    OR_PATH = os.getcwd() 
    path = os.path.join(OR_PATH, directory) 
    os.mkdir(path)

   # Download file to metadata folder 
    os.chdir(path)
    subprocess.run(['wget', 'https://oasis-brains.org/files/oasis_cross-sectional.csv'])

    # Return to original folder
    os.chdir(OR_PATH)

    print("Download completed and metadata files extracted to the metadata/ directory")


def make_data_file():

    image_path = "./data/Data/"
    metadata_path = "./metadata/"

    label_encode = {'Non_Demented':0, 'Very_mild_Dementia':0.5, 'Moderate_Dementia':2, 'Mild_Dementia':1}

    label_list = glob.glob(image_path + "*")
    label_list = [label_name.split("/")[-1] for label_name in label_list]

    # Export path and label in new file
    image_dict = {}
    for label_name in label_list:
        file_list = glob.glob(image_path + label_name + "/*")
        image_dict[label_name] = [file for file in file_list]

    image_df = pd.DataFrame(columns=["id","target"])
    for key,value in image_dict.items():
        new_df = pd.DataFrame(value,columns=["id"])
        new_df["target"] = label_encode[key]
        new_df["patient"] = [i.split("_mpr")[0].split("/")[-1] for i in new_df["id"]]
        image_df = pd.concat([image_df,new_df])

    metadata_df = pd.read_csv(metadata_path + "oasis_cross-sectional.csv")

    combined = image_df.merge(metadata_df,left_on='patient', right_on='ID')
    combined.fillna("0",inplace=True)

    combined.to_csv('data.csv',index=False)

class FocalLoss(nn.Module):
    def __init__(self, weight=None, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure that targets and inputs are on the same device
        if inputs.device != targets.device:
            targets = targets.to(inputs.device)
            
        # Convert targets to one-hot encoding
        #targets_one_hot = torch.eye(inputs.size(-1), device=inputs.device)[targets]

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print("Input size:", x.size())
        avg_pooled = self.avg_pool(x)
        max_pooled = self.max_pool(x)
        #print("Avg pooled size:", avg_pooled.size())
        #print("Max pooled size:", max_pooled.size())
        
        avg_out = self.fc2(self.relu1(self.fc1(avg_pooled)))
        max_out = self.fc2(self.relu1(self.fc1(max_pooled)))
        out = avg_out + max_out
        
        scale = self.sigmoid(out)  # Sigmoid activation
        return x * scale.expand_as(x)  # Scale the input

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print("Input size:", x.size())
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        #print("Avg out size:", avg_out.size())
        #print("Max out size:", max_out.size())
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_cat)
        scale = self.sigmoid(x_out)  # Sigmoid activation
        return x * scale  # Scale the input


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelAttention(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialAttention()

    def forward(self, x):
        # Apply Channel Attention
        channel_attention_map = self.ChannelGate(x)
        x = x * channel_attention_map.expand_as(x)
        
        # Apply Spatial Attention
        spatial_attention_map = self.SpatialGate(x)
        # The spatial attention map is 1xHxW, and needs to be broadcasted across the channel dimension
        # You should not multiply x by x_out again, as it has already been modified by the channel attention
        x = x * spatial_attention_map.expand_as(x)
    
        return x

    
class CBAMBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(CBAMBottleneck, self).__init__()
        # Assuming 'out_planes' is 4 times 'in_planes' for a bottleneck
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        out = self.cbam(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class AttentionCNN(nn.Module):
    def __init__(self):
        super(AttentionCNN, self).__init__()
        # Assume the input image size is 128x128
        self.conv1 = nn.Conv2d(CHANNELS, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2) # Output size: 64x64
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # Output size: 64x64
        self.bn2 = nn.BatchNorm2d(32)
        # Maybe only one pooling layer is needed, so we'll comment out the next pooling line.
        # self.pool = nn.MaxPool2d(2, 2) # Commented out to prevent over-reduction
        self.cbam = CBAM(32) # Attention mechanism

        # Calculate the correct total number of features after the conv and pool layers
        # For example, if after the pooling layer you have a 32x32 feature map with 32 channels:
        # self.fc1 = nn.Linear(32 * 32 * 32, 120)
        # You will need to calculate the correct size here based on your actual output.
        
        # Finally, define the fully connected layers
        self.fc1 = nn.Linear(100352, 120) # Adjust this size accordingly
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, OUTPUTS_a)

    def forward(self, x, tab, register_hook=False):
        x = F.relu(self.bn1(self.conv1(x))) # First convolutional layer
        x = self.pool(x) # First pooling layer
    
        # Save the feature maps and register a hook for the gradients
        self.features = x
        if register_hook:
            self.features.requires_grad_()
            self.features.register_hook(self.save_gradients)
    
        x = F.relu(self.bn2(self.conv2(self.features))) # Apply conv2 here
    
        x = self.pool(x) # Second pooling layer
        x = self.cbam(x) # CBAM layer
    
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x)) # Fully connected layers
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save_gradients(self, grad):
        self.gradients = grad

    def get_activation_gradients(self):
        return self.gradients

    def get_activations(self):
        return self.features
    

class ResNet50_w_metadata(nn.Module):
    def __init__(self):
        super(ResNet50_w_metadata, self).__init__()

        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.input = nn.Conv2d(CHANNELS, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.features = nn.Sequential(*list(model.children())[1:-1])
        self.classifier = nn.Linear(model.fc.in_features, OUTPUTS_a)

        self.act = torch.relu
        self.dropout = nn.Dropout(0.2)
        
        self.linear2 = nn.Linear(2051,1000) # Input shape = 2048 + metadata len
        self.bn2 = nn.BatchNorm1d(num_features=1000)
        self.linear3 = nn.Linear(1000,256)
        self.bn3 = nn.BatchNorm1d(num_features=256)
        self.linear4 = nn.Linear(256,128)
        self.bn4 = nn.BatchNorm1d(num_features=128)
        self.linear5 = nn.Linear(128,64)
        self.bn5 = nn.BatchNorm1d(num_features=64)
        self.linear6 = nn.Linear(64,OUTPUTS_a)

    def forward(self, x, tab):
        x = self.input(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        #x = self.classifier(x)

        tab = torch.cat((x,tab), dim=1)
        
        tab = self.act(tab)
        tab = self.linear2(tab)
        tab = self.dropout(tab)
        tab = self.bn2(tab)
        tab = self.act(tab)
        tab = self.linear3(tab)
        tab = self.dropout(tab)
        tab = self.bn3(tab)
        tab = self.act(tab)
        tab = self.linear4(tab)
        tab = self.dropout(tab)
        tab = self.bn4(tab)
        tab = self.act(tab)
        tab = self.linear5(tab)
        tab = self.dropout(tab)
        tab = self.bn5(tab)
        tab = self.act(tab)
        tab = self.linear6(tab)

        return tab

class GMLP(nn.Module):
    def __init__(self):
        super(GMLP, self).__init__()

        PATCH_SIZE = 16

        self.model = gMLPVision(
            image_size = IMAGE_SIZE,
            patch_size = PATCH_SIZE,
            num_classes = OUTPUTS_a,
            dim = 512,
            depth = 6)

        self.model.to_patch_embed[1] = nn.Linear(PATCH_SIZE*PATCH_SIZE*CHANNELS, 512, bias=True)
        
    def forward(self, x, tab, register_hook=False):
        x = self.model(x)
        
        return x
    
    def save_gradients(self, grad):
        self.gradients = grad

    def get_activation_gradients(self):
        return self.gradients

    def get_activations(self):
        return self.features

class Dataset(data.Dataset):
    '''
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    '''
    def __init__(self, list_IDs, type_data):
        #Initialization'
        self.type_data = type_data
        self.list_IDs = list_IDs

    def __len__(self):
        #Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        #Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Get label
        if self.type_data == 'train':
            y = xdf_dset.target_class.get(ID)

        elif self.type_data == 'validation':
            y = xdf_dset_val.target_class.get(ID)

        else:
            y = xdf_dset_test.target_class.get(ID)

        labels_ohe = np.zeros(OUTPUTS_a)

        for idx, label in enumerate(range(OUTPUTS_a)):
            if label == y:
                labels_ohe[idx] = 1

        y = torch.FloatTensor(labels_ohe)

        # Load images
        if self.type_data == 'train':
            file = xdf_dset.id.get(ID)

        elif self.type_data == 'validation':
            file = xdf_dset_val.id.get(ID)

        else:
            file = xdf_dset_test.id.get(ID)

        # Add normalization step
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        #img = cv2.normalize(img, None, 0, 1.0,cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = torch.FloatTensor(img)
        image = torch.reshape(image, (CHANNELS, IMAGE_SIZE, IMAGE_SIZE))

        # Load tabular data  #https://rosenfelder.ai/multi-input-neural-network-pytorch/
        #metadata_features = ['M','F','Educ1','Educ2','Educ3','Educ4','Educ5','SES0','SES1', 'SES2', 'SES3', 'SES4', 'SES5', 'Age','eTIV','nWBV', 'ASF']
        metadata_features = ['eTIV','nWBV', 'ASF']
        
        # Demographic features only
        #metadata_features = ['M','F','Educ1','Educ2','Educ3','Educ4','Educ5','SES0','SES1', 'SES2', 'SES3', 'SES4', 'SES5', 'Age']
        

        if self.type_data == 'train':
            tabular = xdf_dset[metadata_features].iloc[ID].to_numpy().astype(float)
            tabular = torch.FloatTensor(tabular)

        elif self.type_data == 'validation':
            tabular = xdf_dset_val[metadata_features].iloc[ID].to_numpy().astype(float)
            tabular = torch.FloatTensor(tabular)

        else:
            tabular = xdf_dset_test[metadata_features].iloc[ID].to_numpy().astype(float)
            tabular = torch.FloatTensor(tabular)
         
        return image, tabular, y 


def read_data(train_test_split):
    ## read the data data from the file

    # ---------------------- Parameters for the data loader --------------------------------

    list_of_ids = list(xdf_dset.index)
    list_of_ids_val = list(xdf_dset_val.index)
    list_of_ids_test = list(xdf_dset_test.index)


    # Datasets
    partition = {
        'train': list_of_ids,
        'validation': list_of_ids_val,
        'test' : list_of_ids_test
    }

    # Data Loaders

    if train_test_split == 'train':

        params = {'batch_size': BATCH_SIZE,
              'shuffle': True}

        training_set = Dataset(partition['train'], 'train')
        training_generator = data.DataLoader(training_set, **params)

        params = {'batch_size': BATCH_SIZE,
              'shuffle': False}

        val_set = Dataset(partition['validation'], 'validation')
        val_generator = data.DataLoader(val_set, **params)

        return training_generator, val_generator

    elif train_test_split == 'test':

        params = {'batch_size': BATCH_SIZE,
              'shuffle': False}

        test_set = Dataset(partition['test'], 'test')
        test_generator = data.DataLoader(test_set, **params) 
    
        return test_generator

def model_definition():
    # Define a Keras sequential model
    # Compile the model

    if MODEL_NAME == 'attentioncnn_model':
        model = AttentionCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    if MODEL_NAME == 'resnet50_model':
        model = ResNet50_w_metadata()
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    if MODEL_NAME == 'gmlp':
        model = GMLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model = model.to(device)

    #criterion = nn.CrossEntropyLoss()

    criterion = FocalLoss(alpha=1, gamma=2)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=1, verbose=True)

    return model, optimizer, criterion, scheduler

def train_and_val(train_ds, val_ds, list_of_metrics, list_of_agg, save_on):
    # Use a breakpoint in the code line below to debug your script.

    model, optimizer, criterion, scheduler = model_definition()

    cont = 0
    train_loss_item = list([])
    val_loss_item = list([])
    
    train_loss_list = []
    val_loss_list = []
    train_accuracies = []
    val_accuracies = []


    pred_labels_per_hist = list([])

    model.phase = 0

    met_val_best = 0 # Change to 0 if f1_score or acc
    for epoch in range(n_epoch):
        train_loss, steps_train = 0, 0

        model.train()

        pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        train_hist = list([])
        val_hist = list([])

        with tqdm(total=len(train_ds), desc="Epoch {}".format(epoch)) as pbar:

            for xdata,xtabular,xtarget in train_ds:

                xdata, xtabular,xtarget = xdata.to(device), xtabular.to(device), xtarget.to(device)

                #xdata.requires_grad = True
                #xtabular.requires_grad = True

                optimizer.zero_grad()

                output = model(xdata,xtabular)

                loss = criterion(output, xtarget)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                cont += 1

                steps_train += 1

                train_loss_item.append([epoch, loss.item()])

                pred_labels_per = output.detach().to(torch.device('cpu')).numpy()

                if len(pred_labels_per_hist) == 0:
                    pred_labels_per_hist = pred_labels_per
                else:
                    pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])

                if len(train_hist) == 0:
                    train_hist = xtarget.cpu().numpy()
                else:
                    train_hist = np.vstack([train_hist, xtarget.cpu().numpy()])

                pbar.update(1)
                pbar.set_postfix_str("Train Loss: {:.5f}".format(train_loss / steps_train))

                pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
                real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))
    
        train_loss_list.append(train_loss)

        pred_labels = pred_logits[1:]
        pred_labels = [np.argmax(a) for a in pred_labels]
        real_labels = real_labels[1:]
        real_labels = [np.argmax(a) for a in real_labels]
        
        train_acc = accuracy_score(real_labels, pred_labels)
        train_accuracies.append(train_acc)

        # Metric Evaluation
        train_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels, pred_labels)

        #avg_train_loss = train_loss / steps_train

        ## Finish with Training

        ## Testing the model

        model.eval()

        pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        val_loss, steps_val = 0, 0
        met_val = 0

        with torch.no_grad():

            with tqdm(total=len(val_ds), desc="Epoch {}".format(epoch)) as pbar:

                for xdata,xtabular,xtarget in val_ds:

                    xdata, xtabular, xtarget = xdata.to(device), xtabular.to(device), xtarget.to(device)

                    optimizer.zero_grad()

                    output = model(xdata, xtabular)

                    loss = criterion(output, xtarget)

                    val_loss += loss.item()
                    cont += 1

                    steps_val += 1

                    val_loss_item.append([epoch, loss.item()])

                    pred_labels_per = output.detach().to(torch.device('cpu')).numpy()

                    if len(pred_labels_per_hist) == 0:
                        pred_labels_per_hist = pred_labels_per
                    else:
                        pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])

                    if len(val_hist) == 0:
                        val_hist = xtarget.cpu().numpy()
                    else:
                        val_hist = np.vstack([val_hist, xtarget.cpu().numpy()])

                    pbar.update(1)
                    pbar.set_postfix_str("Validation Loss: {:.5f}".format(val_loss / steps_val))

                    pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
                    real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))

        val_loss_list.append(val_loss)

        # Update learning rate
        scheduler.step(val_loss)

        pred_labels = pred_logits[1:]
        pred_labels = [np.argmax(a) for a in pred_labels]
        real_labels = real_labels[1:]
        real_labels = [np.argmax(a) for a in real_labels]

        val_acc = accuracy_score(real_labels, pred_labels)
        val_accuracies.append(val_acc)

        val_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels, pred_labels)

        xstrres = "Epoch {}: ".format(epoch)
        for met, dat in train_metrics.items():
            xstrres = xstrres +' Train '+met+ ' {:.5f}'.format(dat)


        xstrres = xstrres + " - "
        for met, dat in val_metrics.items():
            xstrres = xstrres + ' Validation '+met+ ' {:.5f}'.format(dat)
            if met == save_on:
                met_val = dat

        print(xstrres)

        #plt_confusion_matrix(real_labels, pred_labels)

        if met_val > met_val_best and SAVE_MODEL:

            torch.save(model.state_dict(), f"{MODEL_NAME}.pt")

            print("The model has been saved!")
            met_val_best = met_val
    
     # Plotting
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='Train Loss', linewidth=3.0)
    plt.plot(val_loss_list, label='Validation Loss', linewidth=3.0)
    plt.title('Loss over epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy', linewidth=3.0)
    plt.plot(val_accuracies, label='Validation Accuracy', linewidth=3.0)
    plt.title('Accuracy over epochs')
    plt.legend()
    plt.savefig(f'{MODEL_NAME}_accuracy_loss.png')
    
    plt.show()

    
def test_model(test_ds, list_of_metrics, list_of_agg):
        
    model, optimizer, criterion, scheduler = model_definition()

    model.load_state_dict(torch.load(f'{MODEL_NAME}.pt', map_location=device))

    cont = 0
    test_loss_item = list([])
    pred_labels_per_hist = list([])
    test_hist = list([])

    ## Testing the model

    model.eval()

    pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

    test_loss, steps_test = 0, 0

    with torch.no_grad():

        with tqdm(total=len(test_ds)) as pbar:

            for xdata,xtabular,xtarget in test_ds:

                xdata, xtabular, xtarget = xdata.to(device), xtabular.to(device), xtarget.to(device)

                optimizer.zero_grad()

                output = model(xdata, xtabular)

                loss = criterion(output, xtarget)

                test_loss += loss.item()
                cont += 1

                steps_test += 1

                test_loss_item.append(loss.item())

                pred_labels_per = output.detach().to(torch.device('cpu')).numpy()

                if len(pred_labels_per_hist) == 0:
                    pred_labels_per_hist = pred_labels_per
                else:
                    pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])

                if len(test_hist) == 0:
                    test_hist = xtarget.cpu().numpy()
                else:
                    test_hist = np.vstack([test_hist, xtarget.cpu().numpy()])

                pbar.update(1)
                pbar.set_postfix_str("Test Loss: {:.5f}".format(test_loss / steps_test))

                pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
                real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))
        

        pred_labels = pred_logits[1:]
        pred_labels = [np.argmax(a) for a in pred_labels]
        real_labels = real_labels[1:]
        real_labels = [np.argmax(a) for a in real_labels]

        test_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels, pred_labels)

        for met, dat in test_metrics.items():
            print ('Test:' +met+ ' {:.5f}'.format(dat))


    plt_confusion_matrix(real_labels, pred_labels)

def metrics_func(metrics, aggregates, y_true, y_pred):
    '''
    multiple functiosn of metrics to call each function
    f1, cohen, accuracy, mattews correlation
    list of metrics: f1_micro, f1_macro, f1_avg, coh, acc, mat
    list of aggregates : avg, sum
    :return:
    '''

    def f1_score_metric(y_true, y_pred, type):
        '''
            type = micro,macro,weighted,samples
        :param y_true:
        :param y_pred:
        :param average:
        :return: res
        '''
        res = f1_score(y_true, y_pred, average=type)
        return res

    def cohen_kappa_metric(y_true, y_pred):
        res = cohen_kappa_score(y_true, y_pred)
        return res

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        return res

    def matthews_metric(y_true, y_pred):
        res = matthews_corrcoef(y_true, y_pred)
        return res

    def hamming_metric(y_true, y_pred):
        res = hamming_loss(y_true, y_pred)
        return res

    xcont = 1
    xsum = 0
    xavg = 0
    res_dict = {}
    for xm in metrics:
        if xm == 'f1_micro':
            # f1 score average = micro
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            # f1 score average = macro
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            # f1 score average = macro
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'f1_min':
            # f1 score average =
            xmet = f1_score_metric(y_true, y_pred, None)
            #xmet = min(xmet)
        elif xm == 'coh':
             # Cohen kappa
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            # Accuracy
            xmet =accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            # Matthews
            xmet =matthews_metric(y_true, y_pred)
        elif xm == 'hlm':
            xmet =hamming_metric(y_true, y_pred)
        else:
            xmet = 0

        res_dict[xm] = xmet

        xsum = xsum + xmet
        xcont = xcont +1

    if 'sum' in aggregates:
        res_dict['sum'] = xsum
    if 'avg' in aggregates and xcont > 0:
        res_dict['avg'] = xsum/xcont
    # Ask for arguments for each metric

    return res_dict

def process_target():

    dict_target = {}
    xerror = 0

    xtarget = list(np.array(xdf_data['target'].unique()))
    le = LabelEncoder()
    le.fit(xtarget)
    final_target = le.transform(np.array(xdf_data['target']))
    class_names=(xtarget)
    xdf_data['target_class'] = final_target

    ## We add the column to the main dataset

    return class_names

def preprocess_data(data):

    df = data[['id','target_class','M/F','Age','Educ','SES','eTIV','nWBV','ASF']]

    convert_dict = {'M/F': 'category',
                    'Educ': 'category',
                    'SES': 'category'}
 
    df = df.astype(convert_dict) 

    enc = OneHotEncoder() 
    df1 = df.join(pd.DataFrame(enc.fit_transform(df[['M/F']]).toarray(),columns=('M','F'))).drop(columns=('M/F'))
    df2 = df1.join(pd.DataFrame(enc.fit_transform(df[['Educ']]).toarray(),columns=('Educ1','Educ2','Educ3','Educ4','Educ5'))).drop(columns=('Educ'))
    df3 = df2.join(pd.DataFrame(enc.fit_transform(df[['SES']]).toarray(),columns=('SES0','SES1','SES2','SES3','SES4','SES5'))).drop(columns=('SES'))
    
    return df3

def transform_data(train_ds, val_ds, test_ds):

    train_ds, val_ds, test_ds = train_ds.reset_index(drop=True), val_ds.reset_index(drop=True), test_ds.reset_index(drop=True)

    float_cols = ['Age','eTIV','nWBV','ASF']

    # The Scaler
    ss = MinMaxScaler()

    # Standardize the training data
    train_ds_ss = ss.fit_transform(train_ds[float_cols])
    train_ds_ss = train_ds.drop(columns=float_cols).join(pd.DataFrame(train_ds_ss, columns=float_cols))

    # Standardize the validation data
    val_ds_ss = ss.transform(val_ds[float_cols])
    val_ds_ss = val_ds.drop(columns=float_cols).join(pd.DataFrame(val_ds_ss, columns=float_cols))

    # Standardize the test data
    test_ds_ss = ss.transform(test_ds[float_cols])
    test_ds_ss = test_ds.drop(columns=float_cols).join(pd.DataFrame(test_ds_ss, columns=float_cols))

    return train_ds_ss, val_ds_ss, test_ds_ss

def plt_confusion_matrix(targets, preds):

    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{MODEL_NAME}_cm.png')

    return plt.show()

def visualize_grad_cam(activations, gradients, original_image, alpha=0.4):
    # Select the activations and gradients for the first image
    activations = activations[0]  # Shape becomes [16, 64, 64]
    gradients = gradients[0]      # Shape becomes [16, 64, 64]

    # Compute pooled gradients
    pooled_gradients = torch.mean(gradients, dim=[1, 2])
    for i in range(activations.shape[0]):
        activations[i, :, :] *= pooled_gradients[i]

    # Generate heatmap
    heatmap = torch.mean(activations, dim=0).squeeze()  # Remove the channel dimension
    heatmap = np.maximum(heatmap.detach().cpu(), 0)
    heatmap /= torch.max(heatmap)

    # Reshape the heatmap
    heatmap = heatmap.unsqueeze(0).unsqueeze(0)  # Add N and C dimensions

    # Resize heatmap to match the original image size
    heatmap = F.interpolate(heatmap, size=(original_image.shape[1], original_image.shape[2]), mode='bilinear', align_corners=False)
    heatmap = heatmap.squeeze().numpy()

    # Overlay the heatmap
    # Overlay the heatmap
    plt.imshow(original_image.permute(1, 2, 0).cpu().detach().numpy())  # Detach the tensor before conversion
    plt.imshow(heatmap, cmap='jet', alpha=alpha)
    plt.axis('off')

    plt.savefig(f'{MODEL_NAME}_xai.png')

    plt.show()

def xai():

    model = AttentionCNN()
    
    # Load the saved weights into the model
    model.load_state_dict(torch.load(f'{MODEL_NAME}.pt'))
    model.to(device)  # Make sure to send the model to the appropriate device

    # Set the model to evaluation mode
    model.eval()  # This ensures the model is in evaluation mode

    # Load and preprocess the test image
    test_image, xtabular, xtarget = next(iter(train_ds))
    test_image.requires_grad_()  # Ensure gradients will be tracked for this image
    test_image = test_image.to(device)

    # Forward pass with hook registration on the test image
    output = model(test_image, xtabular, register_hook=True)  # Gradients need to be tracked here, so don't use torch.no_grad()

    # Compute gradients and activations
    target_class = output.argmax(dim=1)
    model.zero_grad()
    output[:, target_class].sum().backward()  # Backward pass should be outside of torch.no_grad()
    gradients = model.get_activation_gradients()
    activations = model.get_activations()

    # Clear any previous plots to ensure a clean state
    plt.clf()  # Clear the current figure
    plt.close()  # Close the figure window

    # Visualize Grad-CAM
    plt.figure()  # Start a new figure
    visualize_grad_cam(activations, gradients, test_image[0])

if __name__ == '__main__':

    # Set up environment
    # download_dataset()
    # download_metadata()
    # make_data_file()

    FILE_NAME = 'data.csv'

    MODEL_NAME = 'gmlp'
    lr = LR
    BATCH_SIZE = BATCH_SIZE

    try:
        model_name = sys.argv[1]
        if model_name in ['gmlp', 'attentioncnn_model', 'resnet50_model']:
            MODEL_NAME = model_name
        lr = float(sys.argv[3])
        BATCH_SIZE = int(sys.argv[2])

    except:
        pass

    print(f"\nMODEL: {MODEL_NAME}\tBATCH_SIZE: {BATCH_SIZE}\tLEARNING RATE: {lr}")
    
    # Reading and filtering Excel file
    xdf_data_og = pd.read_csv(FILE_NAME)

    xdf_data = xdf_data_og[xdf_data_og['Age']>=60].reset_index(drop=True) # Restrict age to 60 and older
    
    xdf_data = xdf_data[xdf_data['target']!=2].reset_index(drop=True)

    class_names = process_target()

    patient_df = xdf_data[['patient','target']].drop_duplicates()

    train, val = train_test_split(patient_df, test_size=0.30, random_state=490)

    val, test = train_test_split(val, test_size=0.50, random_state=490)

    xdf_dset = xdf_data[xdf_data['patient'].isin(train['patient'])]
    xdf_dset_val = xdf_data[xdf_data['patient'].isin(val['patient'])]
    xdf_dset_test = xdf_data[xdf_data['patient'].isin(test['patient'])]

    xdf_dset = preprocess_data(xdf_dset)
    xdf_dset_val = preprocess_data(xdf_dset_val)
    xdf_dset_test = preprocess_data(xdf_dset_test)

    xdf_dset, xdf_dset_val, xdf_dset_test = transform_data(xdf_dset, xdf_dset_val, xdf_dset_test)
    
    ## read_data creates the dataloaders

    train_ds,val_ds = read_data('train')

    OUTPUTS_a = len(class_names)

    list_of_metrics = ['f1_macro','acc','coh']
    list_of_agg = []

    # Train model and test validation split
    train_and_val(train_ds, val_ds, list_of_metrics, list_of_agg, save_on='f1_macro')

    test_ds = read_data('test')

    # Test on test split
    test_model(test_ds,list_of_metrics, list_of_agg)

    # Explanable AI
    if MODEL_NAME == 'attentioncnn_model':
        xai() 
