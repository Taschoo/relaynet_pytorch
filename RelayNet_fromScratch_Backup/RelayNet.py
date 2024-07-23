import os

# Bereinigen der Kommandozeile
if os.name == 'nt':  # Windows
    os.system('cls')
else:  # Unix-basierte Systeme (Linux, macOS)
    os.system('clear')

import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Überprüfen Sie, ob eine GPU verfügbar ist
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

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

    return np.array(images), np.array(masks)

base_path = 'data'
oct_images, masks = load_images_and_masks(base_path)

# Normalisieren der OCT-Bilder
oct_images = oct_images / 255.0

# Reshape und One-Hot-Encoding der Masken
masks = np.moveaxis(masks, 1, -1)  # Verschieben der Achsen (num_images, height, width, num_masks)
masks = masks / 255.0  # Normalisieren der Masken

# Annahme: Masken haben binäre Werte 0 und 1 und insgesamt 8 Schichten
num_classes = 2
masks = masks[..., np.newaxis]  # Hinzufügen der Kanalachse für One-Hot-Encoding
masks = to_categorical(masks, num_classes=num_classes).reshape(masks.shape[0], masks.shape[1], masks.shape[2], -1)

# Aufteilen der Daten in Trainings- und Test-Sets
X_train, X_test, y_train, y_test = train_test_split(oct_images, masks, test_size=0.2, random_state=42)

# Erstellung des CNN-Modells
def create_cnn_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(16, (1, 1), activation='sigmoid')  # Ausgabe: 8 Schichten * 2 Klassen
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

input_shape = (256, 256, 3)
model = create_cnn_model(input_shape)
model.summary()

# Training des Modells
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Evaluation des Modells
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
