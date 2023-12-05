import torch
from torch import nn
from g_mlp_pytorch import gMLPVision
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = 128
num_classes = 4
batch_size = 64
num_epochs = 100


model = gMLPVision(
    image_size = image_size,
    patch_size = 16,
    num_classes = num_classes,
    dim = 512,
    depth = 6
)
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize((image_size,image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_path = "../train"
test_path = "../test"
train_data = datasets.ImageFolder(train_path, transform=transform)
test_data = datasets.ImageFolder(train_path, transform=transform)

train_loader = DataLoader(train_data, batch_size= batch_size, shuffle= True, pin_memory= True)
test_loader = DataLoader(test_data, batch_size= batch_size, shuffle= False, pin_memory= True)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_test_accuracy = 0.0
test_accuracies = np.array([])
losses = np.array([])

confusion_matrix_data = {f"{i}":[0 for j in range(num_classes)] for i in range(num_classes)}

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    confusion_matrix_data = {f"{i}":[0 for j in range(num_classes)] for i in range(num_classes)}
    tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
    for images, labels in tqdm_bar:

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        tqdm_bar.set_postfix(loss=running_loss / len(train_loader), test_acc=best_test_accuracy)

    losses = np.append(losses, running_loss)


    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    # Test the model
    model.eval()
    correct = 0
    incorrect = 0
    total = 0
    incorrectly_classified = {f"{ii}":0 for ii in range(num_classes)}
    
    tqdm_bar_test = tqdm(test_loader, desc=f"Testing", unit="batch")
    with torch.no_grad():
        for inputs, labels in tqdm_bar_test:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for jj in range(predicted.shape[0]):
                correct_label = labels[jj].item()
                output_label = predicted[jj].item()
                confusion_matrix_data[f"{correct_label}"][output_label] += 1
                if correct_label != output_label:
                    incorrectly_classified[f"{correct_label}"] += 1/len(test_data)
                    
                
            tqdm_bar_test.set_postfix(accuracy=correct / total)
        print(confusion_matrix_data)

    accuracy = correct / total
    test_accuracies = np.append(test_accuracies, accuracy)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    if accuracy > best_test_accuracy:
        best_test_accuracy = accuracy
        torch.save(model.state_dict(), f'gmlp.pth')
        with open("incorrectly_classified_gmlp.csv", "w") as myfile:
            for key, value in incorrectly_classified.items():
                myfile.write(f"{key},{value}\n")
        
        np.save("conf_matrix_gmlp.npy", confusion_matrix_data)
        np.save("test_accuracies_gmlp.npy", test_accuracies)
        np.save("losses_gmlp.npy", losses)
        