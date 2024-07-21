import os
import numpy as np
from tensorflow.keras.optimizers import SGD
from src.models.relaynet import relaynet, combined_loss
from src.data.data_augmentation import get_data_generators
from sklearn.model_selection import train_test_split

def main():
    # Datenpfade
    processed_data_dir = 'data/processed'
    model_dir = 'results'
    X_data_path = os.path.join(processed_data_dir, 'X_data.npy')
    Y_data_path = os.path.join(processed_data_dir, 'Y_data.npy')
    model_save_path = os.path.join(model_dir, 'model.h5')

    # Daten laden und aufteilen
    if not os.path.exists(X_data_path) or not os.path.exists(Y_data_path):
        print(f"\033[91mError: Processed data files '{X_data_path}' or '{Y_data_path}' not found.\033[0m")
        return

    X_data = np.load(X_data_path)  # Laden Sie Ihre B-Scan-Daten
    Y_data = np.load(Y_data_path)  # Laden Sie Ihre Annotationen

    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

    # Datenaugmentation
    train_generator = get_data_generators(X_train, Y_train)

    # Modell kompilieren und trainieren
    input_shape = (128, 128, 1)  # Beispielgröße, muss an Ihren Datensatz angepasst werden
    num_classes = 10  # Anzahl der zu segmentierenden Klassen
    model = relaynet(input_shape, num_classes)
    model.compile(optimizer=SGD(lr=0.1, momentum=0.9),
                  loss=combined_loss,
                  metrics=['accuracy'])

    model.fit(train_generator, epochs=100, validation_data=(X_test, Y_test))

    # Modell speichern
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model.save(model_save_path)
    print(f"Model saved at {model_save_path}")

if __name__ == "__main__":
    main()
