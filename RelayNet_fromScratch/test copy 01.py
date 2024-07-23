import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import torch.nn as nn

# Define transformation to be applied on the images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale images
])

# Custom Dataset class for images and labels
class CustomDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))])
        self.labels = sorted([f for f in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, f))])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.images[idx])
        label_name = os.path.join(self.labels_dir, self.labels[idx])
        
        image = Image.open(img_name).convert('RGB')
        label = Image.open(label_name).convert('L')
        
        if self.transform:
            image = self.transform(image)
            label = transforms.ToTensor()(label)
        
        return image, label

# Paths to the images and labels
images_dir = 'data/759x568/images'
labels_dir = 'data/759x568/labels'

# Create dataset instance
dataset = CustomDataset(images_dir, labels_dir, transform)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Example usage of dataloader
for images, labels in dataloader:
    print(f'Images batch shape: {images.size()}')
    print(f'Labels batch shape: {labels.size()}')
    break

# Define the EncoderBlock
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(7, 3), padding=(3, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x, indices = self.pool(x)
        return x, indices

# Define the DecoderBlock
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(7, 3), padding=(3, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x, indices, output_size, concat_features):
        x = self.unpool(x, indices, output_size=output_size)
        x = torch.cat((x, concat_features), dim=1)  # Concatenate along the channel dimension
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Instantiate the encoder and decoder blocks
encoder = EncoderBlock(in_channels=1, out_channels=16)
decoder = DecoderBlock(in_channels=17, out_channels=1)  # in_channels = 16 (encoded) + 1 (concat_features)

# Function to show images
def show_images(images, labels):
    fig, axes = plt.subplots(nrows=2, ncols=len(images), figsize=(12, 5))
    for i in range(len(images)):
        image = images[i].squeeze().numpy()  # Remove batch and channel dimensions
        label = labels[i].squeeze().numpy()
        axes[0, i].imshow(image, cmap='gray')
        axes[0, i].set_title('Image')
        axes[0, i].axis('off')
        axes[1, i].imshow(label, cmap='gray')
        axes[1, i].set_title('Label')
        axes[1, i].axis('off')
    plt.show()

# Example usage of dataloader and encoder/decoder blocks
for images, labels in dataloader:
    # Shape verification
    print(f'Images batch shape: {images.size()}')
    print(f'Labels batch shape: {labels.size()}')
    
    # Forward pass of the image batch through the encoder block
    encoded, indices = encoder(images)
    
    # Forward pass through the decoder block
    decoded = decoder(encoded, indices, output_size=images.size(), concat_features=images)
    
    # Shape verification of the output and indices
    print(f'Encoded shape: {encoded.shape}')
    print(f'Decoded shape: {decoded.shape}')
    
    # Show the first few images and labels
    show_images(images[:4], labels[:4])
    
    # Show the first few decoded images
    decoded_images = decoded[:4].detach().numpy()
    fig, axes = plt.subplots(1, len(decoded_images), figsize=(12, 5))
    for i in range(len(decoded_images)):
        axes[i].imshow(decoded_images[i][0], cmap='gray')
        axes[i].set_title('Decoded Image')
        axes[i].axis('off')
    plt.show()
    
    break
