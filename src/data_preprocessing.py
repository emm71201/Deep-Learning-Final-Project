import os
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to the size required by your model
    transforms.ToTensor(),  # Convert the PIL Image to a tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the tensor
])

# Define your dataset directory and classes
dataset_dir = '/home/ubuntu/Deep-Learning-Final-Project/src/data/Data'
classes = ['Non Demented', 'Very mild Dementia', 'Mild Dementia', 'Moderate Dementia']

class AlzheimerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.paths = []
        self.labels = []
        self.transform = transform
        
        for label, class_name in enumerate(classes):
            class_dir = os.path.join(root_dir, class_name)
            for filename in os.listdir(class_dir):
                if filename.endswith('.jpg'):
                    self.paths.append(os.path.join(class_dir, filename))
                    self.labels.append(label)  # Assuming class labels are 0, 1, 2, 3

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image, label

# Instantiate the dataset
full_dataset = AlzheimerDataset(root_dir=dataset_dir, transform=transform)

# Get the labels from the dataset
labels = [label for _, label in full_dataset]

# Split the data
train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2, random_state=42, stratify=labels)
train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=42, stratify=[labels[i] for i in train_idx])  # 0.25 x 0.8 = 0.2

# Create data subsets
train_data = Subset(full_dataset, train_idx)
val_data = Subset(full_dataset, val_idx)
test_data = Subset(full_dataset, test_idx)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)