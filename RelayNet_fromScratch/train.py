import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CustomDataset, transform
from model import EncoderBlock, DecoderBlock, ClassificationBlock
from loss import WeightedMultiClassLogisticLoss, DiceLoss, class_weights
import matplotlib.pyplot as plt
import numpy as np

# Paths to the images and labels
images_dir = 'data/759x568/images'
labels_dir = 'data/759x568/labels'

# Create dataset and dataloader
dataset = CustomDataset(images_dir, labels_dir, transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Reduced batch size for testing

# Instantiate the encoder, decoder, and classification blocks
encoder = EncoderBlock(in_channels=1, out_channels=16)
decoder = DecoderBlock(in_channels=17, out_channels=16)
classifier = ClassificationBlock(in_channels=16, num_classes=9)

# Loss functions
criterion1 = WeightedMultiClassLogisticLoss(class_weights)
criterion2 = DiceLoss()

# Optimizer
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(classifier.parameters()), lr=0.001)

# Function to show images
def show_images(images, labels):
    fig, axes = plt.subplots(nrows=2, ncols=len(images), figsize=(12, 5))
    axes = axes if len(images) > 1 else np.array([[axes[0]], [axes[1]]])  # Ensure axes is 2D array
    for i in range(len(images)):
        image = images[i].squeeze().numpy()  # Remove batch and channel dimensions
        label = labels[i].squeeze().numpy()
        
        # Debugging output
        print(f"Image shape: {image.shape}")
        print(f"Label shape: {label.shape}")
        
        if image.ndim == 2:  # Ensure the image has the right number of dimensions
            axes[0, i].imshow(image, cmap='gray')
        elif image.ndim == 3:  # If there's a channel dimension, squeeze it
            axes[0, i].imshow(image[0], cmap='gray')
        axes[0, i].set_title('Image')
        axes[0, i].axis('off')
        
        if label.ndim == 2:
            axes[1, i].imshow(label, cmap='gray')
        elif label.ndim == 3:
            axes[1, i].imshow(label[0], cmap='gray')
        axes[1, i].set_title('Label')
        axes[1, i].axis('off')
    plt.show()

def show_encoded_decoded_images(encoded, decoded):
    fig, axes = plt.subplots(nrows=2, ncols=encoded.size(0), figsize=(12, 5))
    axes = axes if encoded.size(0) > 1 else np.array([[axes[0]], [axes[1]]])  # Ensure axes is 2D array
    for i in range(encoded.size(0)):
        enc_image = encoded[i][0].detach().numpy()  # Show only the first channel
        dec_image = decoded[i][0].detach().numpy()
        axes[0, i].imshow(enc_image, cmap='gray')
        axes[0, i].set_title('Encoded')
        axes[0, i].axis('off')
        axes[1, i].imshow(dec_image, cmap='gray')
        axes[1, i].set_title('Decoded')
        axes[1, i].axis('off')
    plt.show()

# Training loop
num_epochs = 5  # Reduced number of epochs for testing
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    classifier.train()

    epoch_loss = 0
    for images, labels in dataloader:
        optimizer.zero_grad()
        
        # Forward pass of the image batch through the encoder block
        encoded, indices = encoder(images)
        
        # Forward pass through the decoder block
        decoded = decoder(encoded, indices, output_size=images.size(), concat_features=images)
        
        # Forward pass through the classification block
        classified = classifier(decoded)
        
        # Compute loss
        loss1 = criterion1(classified, labels)
        loss2 = criterion2(classified, labels)
        loss = loss1 + loss2
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader)}')

    # Visualize images, encoded, and decoded images for the first batch in each epoch
    show_images(images[:2], labels[:2])
    show_encoded_decoded_images(encoded[:2], decoded[:2])

# Save the trained model weights
torch.save(encoder.state_dict(), 'encoder.pth')
torch.save(decoder.state_dict(), 'decoder.pth')
torch.save(classifier.state_dict(), 'classifier.pth')
