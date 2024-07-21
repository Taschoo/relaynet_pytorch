from skimage.transform import resize

def preprocess_data(images, masks, target_size=(128, 128)):
    X_prep = []
    Y_prep = []

    for img, mask in zip(images, masks):
        img_resized = resize(img, target_size, mode='constant', preserve_range=True)
        mask_resized = resize(mask, target_size, mode='constant', preserve_range=True, order=0)

        X_prep.append(img_resized)
        Y_prep.append(mask_resized)
    
    X_prep = np.array(X_prep)
    Y_prep = np.array(Y_prep)

    # Reshape for Keras [samples, height, width, channels]
    X_prep = X_prep.reshape(-1, target_size[0], target_size[1], 1)
    Y_prep = Y_prep.reshape(-1, target_size[0], target_size[1], 1)
    
    return X_prep, Y_prep
