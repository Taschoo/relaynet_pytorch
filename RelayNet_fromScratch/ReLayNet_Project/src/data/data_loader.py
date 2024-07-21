import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize

def load_images_from_folder(folder, target_size):
    if not os.path.exists(folder):
        print(f"\033[91mError: Folder '{folder}' does not exist.\033[0m")
        return np.array([])  # Rückgabe eines leeren Arrays im Fehlerfall

    images = []
    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = imread(img_path, as_gray=True)
            img_resized = resize(img, target_size, mode='constant', preserve_range=True)
            images.append(img_resized)
    return np.array(images)

def prepare_data(image_folder, mask_folder, target_size=(128, 128)):
    X = load_images_from_folder(image_folder, target_size)
    Y = load_images_from_folder(mask_folder, target_size)
    
    if X.size == 0 or Y.size == 0:
        print("\033[91mError: Failed to load images or masks.\033[0m")
        return None, None
    
    # Reshape for Keras [samples, height, width, channels]
    X = X[..., np.newaxis]
    Y = Y[..., np.newaxis]
    
    return X, Y

def main():
    raw_data_dir = 'ReLayNet_Project/data/dummy'
    processed_data_dir = 'ReLayNet_Project/data/processed'
    target_size = (128, 128)
    
    if not os.path.exists(raw_data_dir):
        print(f"\033[91mError: Raw data directory '{raw_data_dir}' does not exist.\033[0m")
        return
    
    X_data, Y_data = prepare_data(os.path.join(raw_data_dir, 'images'), 
                                  os.path.join(raw_data_dir, 'masks'), 
                                  target_size)
    
    if X_data is None or Y_data is None:
        return
    
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    
    # Split into training and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
    
    # Save the prepared data as .npy files
    np.save(os.path.join(processed_data_dir, 'X_data.npy'), X_train)
    np.save(os.path.join(processed_data_dir, 'Y_data.npy'), Y_train)
    np.save(os.path.join(processed_data_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(processed_data_dir, 'Y_test.npy'), Y_test)
    
    print("\033[92mData preparation completed and saved as .npy files\033[0m")

if __name__ == "__main__":
    main()
