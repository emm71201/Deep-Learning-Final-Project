import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.preprocessing import OneHotEncoder
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random


PATH = "../oasis"
IMAGE_SIZE = 100

#######################################################################
keys = ["1000", "0100", "0010", "0001"]
encoding = {}
i = 0
for folder in os.listdir(PATH):
    encoding[folder] = keys[i]
    i += 1
def process_images(PATH):
    """ return a dataframe with path to each image and their labels
    """
    df = {"path":[], "class":[], "encoding":[], "split":[]}
    for folder in os.listdir(PATH):
        for file in os.listdir(os.path.join(PATH, folder)):
            complete_path = os.path.join(PATH, folder, file)
            df["path"].append(complete_path)
            df["class"].append(folder)
            df["encoding"].append(encoding[folder])

            if np.random.uniform(0,1) < 0.2:
                df["split"].append("test")
            else:
                df["split"].append("train")

    return pd.DataFrame(df)
class ImageDataset(Dataset):

    def __init__(self, df, transform=None):
        """Arguments:
            df (Pandas dataframe): dataframe with path and label of each image
            transformation (callable, optional): Optional transform
        """
        self.annotations = df
        self.transform = transform
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.annotations.iloc[idx]
        img_path = row["path"]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = torch.FloatTensor(image)
        image = image.unsqueeze(0)
        label = torch.from_numpy(np.array([int(item) for item in row["encoding"]]))

        return image, label

df_images = process_images(PATH)
df_images = df_images.sample(frac=1)
# data = ImageDataset(df_images)
# image, label = data[3]
# print(image)

#####################################################





