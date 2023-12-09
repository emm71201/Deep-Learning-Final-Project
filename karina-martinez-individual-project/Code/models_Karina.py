# This is a sample Python script.
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

'''
LAST UPDATED 11/10/2021, lsdr
'''

## Process images in parallel

## folder "Data" images
## folder "excel" excel file , whatever is there is the file
## get the classes from the excel file
## folder "Documents" readme file

SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

n_epoch = 10
BATCH_SIZE = 32
LR =  0.001

## Image processing
CHANNELS = 1
IMAGE_SIZE = 224

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SAVE_MODEL = False
        
#---- Define the model ---- #
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.act = torch.relu

        self.linear2 = nn.Linear(3,256)
        self.linear3 = nn.Linear(256,256)
        self.linear4 = nn.Linear(256,128)
        self.linear5 = nn.Linear(128,64)
        self.linear6 = nn.Linear(64,OUTPUTS_a)

    def forward(self, x, tab):

        tab = self.linear2(tab)
        tab = self.act(tab)
        tab = self.linear3(tab)
        tab = self.act(tab)
        tab = self.linear4(tab)
        tab = self.act(tab)
        tab = self.linear5(tab)
        tab = self.act(tab)

        return self.linear6(tab)


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
    def __init__(self, num_classes):
        super(AttentionCNN, self).__init__()
        # Assume the input image size is 128x128
        self.conv1 = nn.Conv2d(CHANNELS, 16, kernel_size=3, padding=1) # Output size: 128x128
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
        self.fc1 = nn.Linear(100352, 5000) # Adjust this size accordingly
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, num_classes)

        self.act = torch.relu

        self.linear2 = nn.Linear(5017,1000)
        self.linear3 = nn.Linear(1000,256)
        self.linear4 = nn.Linear(256,128)
        self.linear5 = nn.Linear(128,64)
        self.linear6 = nn.Linear(64,OUTPUTS_a)

    def forward(self, x, tab):
        x = F.relu(self.bn1(self.conv1(x)))
        #print("Size after conv1 and relu:", x.size())
        x = self.pool(x)
       # print("Size after pool1:", x.size())
        x = F.relu(self.bn2(self.conv2(x)))
        #print("Size after conv2 and relu:", x.size())
        x = self.pool(x)
        #print("Size after pool2:", x.size())
        x = self.cbam(x)
        #print("Size after CBAM:", x.size())
        x = x.view(x.size(0), -1)
        #print("Size before fc1:", x.size()) #[32, 100352]
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        
        tab = torch.cat((x,tab), dim=1)

        tab = self.act(tab)
        tab = self.linear2(tab)
        tab = self.act(tab)
        tab = self.linear3(tab)
        tab = self.act(tab)
        tab = self.linear4(tab)
        tab = self.act(tab)
        tab = self.linear5(tab)
        tab = self.act(tab)

        return self.linear6(tab)
    


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(CHANNELS, 16, (3, 3))
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pad1 = nn.ZeroPad2d(2)

        self.conv2 = nn.Conv2d(16, 128, (3, 3))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(128, 4)
        self.act = torch.relu

        self.linear2 = nn.Linear(17,32)
        self.linear3 = nn.Linear(32,16)
        self.linear4 = nn.Linear(16,4)

        self.linear5 = nn.Linear(8,OUTPUTS_a)

    def forward(self, x, tab): 
        x = self.pad1(self.convnorm1(self.act(self.conv1(x))))
        x = self.act(self.conv2(self.act(x)))
        x = self.linear(self.global_avg_pool(x).view(-1, 128))

        tab = self.linear2(tab)
        tab = self.act(tab)
        tab = self.linear3(tab)
        tab = self.act(tab)
        tab = self.linear4(tab)
        tab = self.act(tab)

        x = torch.cat((x,tab), dim=1)
        x = self.act(x)

        return self.linear5(x)
    

class ResNet50_w_metadata(nn.Module):
    def __init__(self):
        super(ResNet50_w_metadata, self).__init__()
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.input = nn.Conv2d(CHANNELS, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.features = nn.Sequential(*list(model.children())[1:-1])
        self.classifier = nn.Linear(model.fc.in_features, OUTPUTS_a)

        self.act = torch.relu

        self.linear2 = nn.Linear(2065,1000)
        self.linear3 = nn.Linear(1000,256)
        self.linear4 = nn.Linear(256,128)
        self.linear5 = nn.Linear(128,64)
        self.linear6 = nn.Linear(64,OUTPUTS_a)

    def forward(self, x, tab):
        x = self.input(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        #x = self.classifier(x)

        tab = torch.cat((x,tab), dim=1)

        tab = self.act(tab)
        tab = self.linear2(tab)
        tab = self.act(tab)
        tab = self.linear3(tab)
        tab = self.act(tab)
        tab = self.linear4(tab)
        tab = self.act(tab)
        tab = self.linear5(tab)
        tab = self.act(tab)

        return self.linear6(tab)


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
        else:
            file = xdf_dset_test.id.get(ID)

        # Add normalization step
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        image = torch.FloatTensor(img)
        image = torch.reshape(image, (CHANNELS, IMAGE_SIZE, IMAGE_SIZE))

        # Load tabular data  #https://rosenfelder.ai/multi-input-neural-network-pytorch/
        metadata_features = ['M','F','Educ1','Educ2','Educ3','Educ4','Educ5','SES0','SES1', 'SES2', 'SES3', 'SES4', 'SES5', 'Age','eTIV','nWBV', 'ASF']
        #metadata_features = ['eTIV','nWBV', 'ASF']

        if self.type_data == 'train':
            tabular = xdf_dset[metadata_features].iloc[ID].to_numpy().astype(float)
            tabular = torch.FloatTensor(tabular)

        else:
            tabular = xdf_dset_test[metadata_features].iloc[ID].to_numpy().astype(float)
            tabular = torch.FloatTensor(tabular)
         
        return image, tabular, y 


def read_data():
    ## Only the training set
    ## xdf_dset ( data set )
    ## read the data data from the file


    #ds_inputs = np.array(xdf_dset['id'])

    #ds_targets = xdf_dset['target_class']

    # ---------------------- Parameters for the data loader --------------------------------

    list_of_ids = list(xdf_dset.index)
    list_of_ids_test = list(xdf_dset_test.index)


    # Datasets
    partition = {
        'train': list_of_ids,
        'test' : list_of_ids_test
    }

    # Data Loaders

    params = {'batch_size': BATCH_SIZE,
              'shuffle': True}

    training_set = Dataset(partition['train'], 'train')
    training_generator = data.DataLoader(training_set, **params)

    params = {'batch_size': BATCH_SIZE,
              'shuffle': False}

    test_set = Dataset(partition['test'], 'test')
    test_generator = data.DataLoader(test_set, **params)

    ## Make the channel as a list to make it variable

    return training_generator, test_generator

#def save_model(model):
    # Open the file

    #print(model, file=open('summary_{}.txt'.format(NICKNAME), "w"))

def model_definition():
    # Define a Keras sequential model
    # Compile the model

    #model = CNN()
    #model = ResNet50_w_metadata()
    #model = AttentionCNN(num_classes=4)
    model = MLP()

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=1, verbose=True)

    #save_model(model) # Generate summary file

    return model, optimizer, criterion, scheduler

def train_and_test(train_ds, test_ds, list_of_metrics, list_of_agg, save_on):
    # Use a breakpoint in the code line below to debug your script.

    model, optimizer, criterion, scheduler = model_definition()

    cont = 0
    train_loss_item = list([])
    test_loss_item = list([])

    pred_labels_per_hist = list([])

    model.phase = 0

    met_test_best = 0 # Change to 0 if f1_score or acc
    for epoch in range(n_epoch):
        train_loss, steps_train = 0, 0

        model.train()

        pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        train_hist = list([])
        test_hist = list([])

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
                

        pred_labels = pred_logits[1:]
        pred_labels = [np.argmax(a) for a in pred_labels]
        real_labels = real_labels[1:]
        real_labels = [np.argmax(a) for a in real_labels]
        

        # Metric Evaluation
        train_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels, pred_labels)

        #avg_train_loss = train_loss / steps_train

        ## Finish with Training

        ## Testing the model

        model.eval()

        pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        test_loss, steps_test = 0, 0
        met_test = 0

        with torch.no_grad():

            with tqdm(total=len(test_ds), desc="Epoch {}".format(epoch)) as pbar:

                for xdata,xtabular,xtarget in test_ds:

                    xdata, xtabular, xtarget = xdata.to(device), xtabular.to(device), xtarget.to(device)

                    optimizer.zero_grad()

                    output = model(xdata, xtabular)

                    loss = criterion(output, xtarget)

                    test_loss += loss.item()
                    cont += 1

                    steps_test += 1

                    test_loss_item.append([epoch, loss.item()])

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

        # Update learning rate
        scheduler.step(test_loss)

        pred_labels = pred_logits[1:]
        pred_labels = [np.argmax(a) for a in pred_labels]
        real_labels = real_labels[1:]
        real_labels = [np.argmax(a) for a in real_labels]

        test_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels, pred_labels)

        #acc_test = accuracy_score(real_labels[1:], pred_labels)
        #hml_test = hamming_loss(real_labels[1:], pred_labels)
        #avg_test_loss = test_loss / steps_test

        xstrres = "Epoch {}: ".format(epoch)
        for met, dat in train_metrics.items():
            xstrres = xstrres +' Train '+met+ ' {:.5f}'.format(dat)


        xstrres = xstrres + " - "
        for met, dat in test_metrics.items():
            xstrres = xstrres + ' Test '+met+ ' {:.5f}'.format(dat)
            if met == save_on:
                met_test = dat

        print(xstrres)

        plt_confusion_matrix(real_labels, pred_labels)

        if met_test > met_test_best and SAVE_MODEL:
        #if SAVE_MODEL:

            torch.save(model.state_dict(), "model.pt")
            #xdf_dset_results = xdf_dset_test.copy()

            ## The following code creates a string to be saved as 1,2,3,3,
            ## This code will be used to validate the model
            #xfinal_pred_labels = []
            #for i in range(len(pred_labels)):
            #    joined_string = ",".join(str(int(e)) for e in pred_labels[i])
            #    xfinal_pred_labels.append(joined_string)

            #xdf_dset_results['results'] = xfinal_pred_labels

            #xdf_dset_results.to_csv('model_results.csv', index = False)
            print("The model has been saved!")
            met_test_best = met_test

    


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

    df = data[['id','target','M/F','Age','Educ','SES','eTIV','nWBV','ASF']]

    convert_dict = {'M/F': 'category',
                    'Educ': 'category',
                    'SES': 'category'}
 
    df = df.astype(convert_dict) 
    df = df[df['Age']>=60].reset_index(drop=True) # Restrict age to 60 and older

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

    return plt.show()

if __name__ == '__main__':

    FILE_NAME = 'data.csv'
    
    # Reading and filtering Excel file
    xdf_data_og = pd.read_csv(FILE_NAME)

    xdf_data = preprocess_data(xdf_data_og)

    ## Process Classes    
    class_names = process_target()

    ## Processing Train dataset

    xdf_dset, xdf_dset_val = train_test_split(xdf_data, test_size=0.30, random_state=SEED)

    xdf_dset_val, xdf_dset_test = train_test_split(xdf_data, test_size=0.50, random_state=SEED)

    xdf_dset, xdf_dset_val, xdf_dset_test = transform_data(xdf_dset, xdf_dset_val, xdf_dset_test)
    
    ## read_data creates the dataloaders

    train_ds,test_ds = read_data()

    OUTPUTS_a = len(class_names)

    list_of_metrics = ['f1_macro','acc','coh']
    list_of_agg = []

    train_and_test(train_ds, test_ds, list_of_metrics, list_of_agg, save_on='f1_macro')

    

