import numpy as np
from sklearn.metrics import jaccard_score
from src.models.relaynet import relaynet

# Daten laden
X_test = np.load('path_to_data/X_test.npy')  # Laden Sie Ihre Testdaten
Y_test = np.load('path_to_data/Y_test.npy')  # Laden Sie Ihre Testlabels

# Modell laden
input_shape = (128, 128, 1)  # Beispielgröße, muss an Ihren Datensatz angepasst werden
num_classes = 10  # Anzahl der zu segmentierenden Klassen
model = relaynet(input_shape, num_classes)
model.load_weights('path_to_model_weights.h5')

# Vorhersagen auf Testdaten
Y_pred = model.predict(X_test)

# Berechnung des Dice-Koeffizienten
def dice_coefficient(y_true, y_pred, smooth=1e-5):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

dice_scores = [dice_coefficient(Y_test[i], Y_pred[i]) for i in range(len(Y_test))]
mean_dice_score = np.mean(dice_scores)
print(f'Mean Dice Score: {mean_dice_score}')
