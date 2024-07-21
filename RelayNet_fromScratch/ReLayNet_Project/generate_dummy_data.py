import os
import numpy as np
from skimage.io import imsave
from skimage.draw import random_shapes

def create_directories(base_dir):
    dirs = [
        'ReLayNet_Project/data/dummy/images', 'ReLayNet_Project/data/dummy/masks'
    ]
    for d in dirs:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)

def generate_dummy_data(image_folder, mask_folder, num_samples=10, image_size=(128, 128), num_classes=10):
    for i in range(num_samples):
        result = random_shapes(image_size, max_shapes=5, min_shapes=3, min_size=20, max_size=40)
        image = result[0]
        
        # Convert to grayscale if the image is multichannel
        if image.ndim == 3:
            image = image[:, :, 0]
        
        mask = np.random.randint(0, num_classes, image_size)
        
        imsave(os.path.join(image_folder, f'image_{i:03d}.png'), image.astype(np.uint8))
        imsave(os.path.join(mask_folder, f'mask_{i:03d}.png'), mask.astype(np.uint8))

def main():
    base_dir = '.'  # Verwenden Sie das aktuelle Verzeichnis als Basisverzeichnis
    create_directories(base_dir)
    
    # Generate dummy data for dummy images and masks
    generate_dummy_data(os.path.join(base_dir, 'ReLayNet_Project/data/dummy/images'), os.path.join(base_dir, 'ReLayNet_Project/data/dummy/masks'))

    print("\033[92mDummy data generated and stored in the appropriate directories.\033[0m")

if __name__ == "__main__":
    main()
