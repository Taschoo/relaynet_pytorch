import os
from sklearn.model_selection import train_test_split
from utils.data_loader import load_images_and_masks
from models.simple_cnn import create_cnn_model

def main():
    base_path = 'data'
    oct_images, masks = load_images_and_masks(base_path)

    # Aufteilen der Daten in Trainings- und Test-Sets
    X_train, X_test, y_train, y_test = train_test_split(oct_images, masks, test_size=0.2, random_state=42)

    input_shape = (256, 256, 3)
    model = create_cnn_model(input_shape)
    model.summary()

    # Training des Modells
    history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

    # Speichern des Modells
    model.save('saved_models/saved_model.keras')

    # Evaluation des Modells
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()
