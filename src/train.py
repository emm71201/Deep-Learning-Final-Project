import sys
sys.path.append('../utils/') 

from utils.simple_CNN import SimpleCNN  # Import SimpleCNN
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_preprocessing import train_loader, val_loader, test_loader, class_sample_count  # Import data loaders

# Training settings
num_epochs = 10 

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, choices=["SimpleCNN"])
args = parser.parse_args()

# Define the number of classes for your dataset
num_classes = 4  # Change this based on your dataset

# Instantiate the model
model = SimpleCNN(num_classes=num_classes) if args.model == "SimpleCNN" else None
if model is None:
    raise ValueError("Invalid model selected.")

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # Move the model to the appropriate device

# Calculate class weights and initialize the loss function
class_weights = 1. / torch.tensor(class_sample_count, dtype=torch.float)
class_weights = class_weights / class_weights.sum()  # Normalize to sum to 1
class_weights = class_weights.to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# Training Function
def train_model(train_ds, criterion, optimizer, num_epochs=10, device='cpu'):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (xdata, xtarget) in enumerate(train_ds):
            xdata = xdata.unsqueeze(1).float().to(device)  # Add channel dimension if needed
            xtarget = xtarget.to(device)
            optimizer.zero_grad()  # Zero the gradients
            output = model(xdata)  # Forward pass
            loss = criterion(output, xtarget)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the model

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} completed. Avg Loss: {running_loss / len(train_ds.dataset):.5f}")

# Training loop
train_model(train_loader, criterion, optimizer, num_epochs=num_epochs, device=device)

# Validation
def validate_model(val_ds, model, criterion, device='cpu'):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xdata, xtarget in val_ds:
            xdata, xtarget = xdata.to(device), xtarget.to(device)
            output = model(xdata)
            loss = criterion(output, xtarget)
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += xtarget.size(0)
            correct += (predicted == xtarget).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Validation Loss: {val_loss / len(val_ds.dataset):.5f}, Validation Accuracy: {accuracy:.2f}%")

validate_model(val_loader, model, criterion, device=device)

# Testing
def test_model(test_ds, model, device='cpu'):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for xdata, xtarget in test_ds:
            xdata, xtarget = xdata.to(device), xtarget.to(device)
            output = model(xdata)
            _, predicted = torch.max(output.data, 1)
            total += xtarget.size(0)
            correct += (predicted == xtarget).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

test_model(test_loader, model, device=device)
