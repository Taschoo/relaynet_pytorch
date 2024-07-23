import tensorflow as tf
from utils.data_loader import load_images_and_masks
from sklearn.model_selection import train_test_split

def main():
    base_path = 'data'
    oct_images, masks = load_images_and_masks(base_path)

    # Aufteilen der Daten in Trainings- und Test-Sets
    X_train, X_test, y_train, y_test = train_test_split(oct_images, masks, test_size=0.2, random_state=42)

    # Laden des gespeicherten Modells
    model = tf.keras.models.load_model('saved_models/saved_model.keras')

    # Kompilieren des Modells (optional, falls erforderlich)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Evaluation des Modells
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()
