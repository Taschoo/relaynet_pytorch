import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

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

# Example usage
if __name__ == "__main__":
    images_dir = 'data/759x568/images'
    labels_dir = 'data/759x568/labels'
    dataset = CustomDataset(images_dir, labels_dir, transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for images, labels in dataloader:
        print(f'Images batch shape: {images.size()}')
        print(f'Labels batch shape: {labels.size()}')
        break
