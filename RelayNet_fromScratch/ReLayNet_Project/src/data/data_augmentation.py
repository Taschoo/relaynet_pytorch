from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(X_train, Y_train):
    data_gen_args = dict(horizontal_flip=True)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Fit the generators to the data
    image_datagen.fit(X_train, augment=True, seed=42)
    mask_datagen.fit(Y_train, augment=True, seed=42)

    # Create generators
    image_generator = image_datagen.flow(X_train, batch_size=32, seed=42)
    mask_generator = mask_datagen.flow(Y_train, batch_size=32, seed=42)

    # Combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)
    return train_generator
