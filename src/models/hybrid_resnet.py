import sys
sys.path.append('../utils/') 

from utils.model_utilities_se import SELayer, SEBottleneck
from utils.model_utilities_cbam import CBAM, CBAMBottleneck
#from torchvision.models import resnet50
import torch.nn as nn
from utils.model_utilities_se import SELayer, SEBottleneck
from torchvision.models import resnet34

class SEResNet(nn.Module):
    def __init__(self, num_classes):
        super(SEResNet, self).__init__()
        # Load a pre-trained ResNet model
        self.base_model = resnet34(pretrained=True)
        
        # Integrate SE blocks in ResNet layers
        self.base_model.layer1 = self._make_layer_with_se(SEBottleneck, 64, 3)
        # Continue integrating SE blocks in other layers as needed
        
        # Replace the classifier
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def _make_layer_with_se(self, block, planes, blocks, stride=1):
        layers = []
        for _ in range(blocks):
            layers.append(block(self.base_model.inplanes, planes, stride))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.base_model(x)


# Create an instance of the model
num_classes = 4  # Example: for a dataset with 4 classes
model = SEResNet(num_classes=num_classes)
