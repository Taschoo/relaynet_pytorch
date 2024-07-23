import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from model import EncoderBlock, DecoderBlock, ClassificationBlock

# Define transformation to be applied on the images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale images
])

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Instantiate the encoder, decoder, and classification blocks
encoder = EncoderBlock(in_channels=1, out_channels=16)
decoder = DecoderBlock(in_channels=17, out_channels=16)
classifier = ClassificationBlock(in_channels=16, num_classes=9)

# Load the trained model weights
encoder.load_state_dict(torch.load('encoder.pth'))
decoder.load_state_dict(torch.load('decoder.pth'))
classifier.load_state_dict(torch.load('classifier.pth'))

encoder.eval()
decoder.eval()
classifier.eval()

def predict(image_path):
    image = load_image(image_path)
    with torch.no_grad():
        encoded, indices = encoder(image)
        decoded = decoder(encoded, indices, output_size=image.size(), concat_features=image)
        classified = classifier(decoded)
    return classified.squeeze(0)  # Remove batch dimension

# Define colors (as RGB tuples) for each label index
label_colors_hex = {
    1: "#731300",   # Dunkles Rotbraun
    2: "#8e2c00",   # Dunkelrot
    3: "#a84900",   # Kupfer
    4: "#c75b00",   # Warmes Orange
    5: "#e0703b",   # Sanftes Orange
    6: "#b86c7f",   # Mauve
    7: "#5b5b9e",   # Grau-Blau
    8: "#004373"    # Tiefes Blau
}

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

label_colors = {label: hex_to_rgb(color) for label, color in label_colors_hex.items()}

def colorize_prediction(prediction):
    color_image = np.full((prediction.shape[0], prediction.shape[1], 3), (255, 255, 255), dtype=np.uint8)
    for label, color in label_colors.items():
        color_image[prediction == label] = color
    return color_image

def show_prediction(image_path, prediction):
    image = Image.open(image_path).convert('RGB')
    image = transforms.Grayscale(num_output_channels=1)(image)
    
    prediction = prediction.argmax(0).numpy()
    colored_prediction = colorize_prediction(prediction)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(colored_prediction)
    plt.title('Predicted Mask')
    plt.axis('off')
    
    plt.show()

# Example usage
image_path = 'data/759x568/images/Image_h_44.png'
prediction = predict(image_path)
show_prediction(image_path, prediction)
