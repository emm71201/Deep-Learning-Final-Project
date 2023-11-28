import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision import models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pretrained models:
# 1. RESNET 34
def resnet_34(output_features, layers=256):

    """ create the resnet34 pretrained model"""

    resnet_model = models.resnet34(pretrained=True)
    num_features = resnet_model.fc.in_features
    # Modify the classifier for your specific case (adjust num_classes)
    resnet_model.fc = nn.Sequential(
        nn.Linear(num_features, layers),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(layers, output_features)
    )
    resnet_model = resnet_model.to(device)

    return resnet_model

# RESNET 50
def resnet_50(output_features, layers=256):

    """ create the resnet50 pretrained model"""

    resnet_model = models.resnet50(pretrained=True)
    num_features = resnet_model.fc.in_features
    # Modify the classifier for your specific case (adjust num_classes)
    resnet_model.fc = nn.Sequential(
        nn.Linear(num_features, layers),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(layers, output_features)
    )
    resnet_model = resnet_model.to(device)

    return resnet_model

# RESNET 101
def resnet_101(output_features, layers=256):

    """ create the resnet101 pretrained model"""

    resnet_model = models.resnet101(pretrained=True)
    num_features = resnet_model.fc.in_features
    # Modify the classifier for your specific case (adjust num_classes)
    resnet_model.fc = nn.Sequential(
        nn.Linear(num_features, layers),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(layers, output_features)
    )
    resnet_model = resnet_model.to(device)

    return resnet_model