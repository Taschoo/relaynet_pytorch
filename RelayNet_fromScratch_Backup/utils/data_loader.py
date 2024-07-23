import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

def load_images_and_masks(base_path, target_size=(256, 256)):
    images = []
    masks = []

    for dir_name in sorted(os.listdir(base_path)):
        dir_path = os.path.join(base_path, dir_name)
        if os.path.isdir(dir_path):
            image_path = os.path.join(dir_path, 'original.jpg')
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.resize(image, target_size)
                images.append(image)

                mask_stack = []
                masks_path = os.path.join(dir_path, 'masks')
                for i in range(1, 9):
                    mask_path = os.path.join(masks_path, f'mask_{i}.png')
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        mask = cv2.resize(mask, target_size)
                        mask = (mask > 127).astype(np.uint8)  # Binarisierung
                        mask_stack.append(mask)
                masks.append(mask_stack)

    images = np.array(images) / 255.0  # Normalisieren der OCT-Bilder
    masks = np.array(masks)
    masks = np.moveaxis(masks, 1, -1)  # Verschieben der Achsen (num_images, height, width, num_masks)
    masks = masks / 255.0  # Normalisieren der Masken

    # Annahme: Masken haben binäre Werte 0 und 1 und insgesamt 8 Schichten
    num_classes = 2
    masks = masks[..., np.newaxis]  # Hinzufügen der Kanalachse für One-Hot-Encoding
    masks = to_categorical(masks, num_classes=num_classes).reshape(masks.shape[0], masks.shape[1], masks.shape[2], -1)

    return images, masks
