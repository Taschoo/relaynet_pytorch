import numpy as np

def dice_coefficient(y_true, y_pred, smooth=1e-5):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
