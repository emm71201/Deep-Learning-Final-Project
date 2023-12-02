# This is a sample Python script.
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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
BATCH_SIZE = 64
LR =  0.001

## Image processing
CHANNELS = 1
IMAGE_SIZE = 224

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SAVE_MODEL = False
        
#---- Define the model ---- #

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(CHANNELS, 16, (3, 3))
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pad1 = nn.ZeroPad2d(2)

        self.conv2 = nn.Conv2d(16, 128, (3, 3))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(128, 5)
        self.act = torch.relu

        self.linear2 = nn.Linear(7,10)
        self.linear3 = nn.Linear(10,10)
        self.linear4 = nn.Linear(10,5)

        self.linear5 = nn.Linear(10,OUTPUTS_a)

    def forward(self, x, tab): # add tabular data here
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
        if self.type_data == 'train':
            tabular = xdf_dset.iloc[0,2:].to_numpy().astype(float)
            tabular = torch.FloatTensor(tabular)
        else:
            tabular = xdf_dset.iloc[0,2:].to_numpy().astype(float)
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

def save_model(model):
    # Open the file

    print(model, file=open('summary_{}.txt'.format(NICKNAME), "w"))

def model_definition():
    # Define a Keras sequential model
    # Compile the model

    model = CNN()

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

        if met_test < met_test_best and SAVE_MODEL:
        #if SAVE_MODEL:

            torch.save(model.state_dict(), "model_{}.pt".format(NICKNAME))
            xdf_dset_results = xdf_dset_test.copy()

            ## The following code creates a string to be saved as 1,2,3,3,
            ## This code will be used to validate the model
            xfinal_pred_labels = []
            for i in range(len(pred_labels)):
                joined_string = ",".join(str(int(e)) for e in pred_labels[i])
                xfinal_pred_labels.append(joined_string)

            xdf_dset_results['results'] = xfinal_pred_labels

            xdf_dset_results.to_excel('{}_results_{}.xlsx'.format(CLASS,NICKNAME), index = False)
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


if __name__ == '__main__':

    FILE_NAME = 'train_test.csv'
    
    # Reading and filtering Excel file
    xdf_data_og = pd.read_csv(FILE_NAME, dtype=str)
    
    df = xdf_data_og[['id','target','M/F','Age','eTIV','nWBV','ASF']]
    convert_dict = {'id': str,
                'target': float,
                'M/F': 'category',
                'Age': float,
                'eTIV': float,
                'nWBV': float,
                'ASF': float
                }
 
    df = df.astype(convert_dict) 
    enc = OneHotEncoder() 
    xdf_data = df.join(pd.DataFrame(enc.fit_transform(df[['M/F']]).toarray())).drop(columns=('M/F'))

    ## Process Classes    
    class_names = process_target()

    ## Processing Train dataset

    xdf_dset, xdf_dset_val = train_test_split(xdf_data, test_size=0.30, random_state=SEED)

    xdf_dset_val, xdf_dset_test = train_test_split(xdf_data, test_size=0.50, random_state=SEED)
    
    ## read_data creates the dataloaders

    train_ds,test_ds = read_data()

    OUTPUTS_a = len(class_names)

    list_of_metrics = ['f1_macro','acc','coh']
    list_of_agg = []

    train_and_test(train_ds, test_ds, list_of_metrics, list_of_agg, save_on='f1_macro')