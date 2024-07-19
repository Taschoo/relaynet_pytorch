import numpy as np
import h5py
import torch

# Parameter
num_examples = 100  # Anzahl der Beispiele
height = 512
width = 512
num_classes = 9

# Dummy-Bilddaten erstellen
data = np.random.rand(num_examples, 1, height, width).astype(np.float32)

# Dummy-Labels und Gewichtungen erstellen
labels = np.random.randint(0, num_classes, (num_examples, height, width)).astype(np.float32)
weights = np.random.rand(num_examples, height, width).astype(np.float32)

# Überprüfen der Label-Werte
assert np.all(labels >= 0) and np.all(labels < num_classes), "Labels außerhalb des Bereichs"

# Kombiniertes Label- und Gewichtungsarray
label_and_weight = np.stack([labels, weights], axis=1)  # Shape: (num_examples, 2, height, width)

# Dummy-Set erstellen (1 für Training, 3 für Test)
set_array = np.random.choice([1, 3], num_examples).astype(np.int32)

# Daten in HDF5-Dateien speichern
with h5py.File('datasets/Data.h5', 'w') as f:
    f.create_dataset('data', data=data)

with h5py.File('datasets/label.h5', 'w') as f:
    f.create_dataset('label', data=label_and_weight)

with h5py.File('datasets/set.h5', 'w') as f:
    f.create_dataset('set', data=set_array)

print("Dummy-Daten wurden erstellt und gespeichert.")

# Überprüfen der Dimensionen
print(f'Datenform: {data.shape}')
print(f'Label- und Gewichtungsform: {label_and_weight.shape}')
print(f'Set-Array-Form: {set_array.shape}')