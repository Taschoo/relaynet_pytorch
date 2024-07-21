import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Input, BatchNormalization, Activation
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def encoder_block(inputs, filters, kernel_size=(7, 3), pool_size=(2, 2)):
    x = Conv2D(filters, kernel_size, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
    return x, x

def decoder_block(inputs, skip_features, filters, kernel_size=(7, 3), pool_size=(2, 2)):
    x = UpSampling2D(size=pool_size)(inputs)
    x = Concatenate()([x, skip_features])
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def relaynet(input_shape, num_classes):
    inputs = Input(input_shape)
    
    # Encoder
    e1, p1 = encoder_block(inputs, 64)
    e2, p2 = encoder_block(p1, 128)
    e3, p3 = encoder_block(p2, 256)
    
    # Decoder
    d1 = decoder_block(e3, e2, 128)
    d2 = decoder_block(d1, e1, 64)
    
    # Classification block
    outputs = Conv2D(num_classes, (1, 1), padding='same')(d2)
    outputs = Activation('softmax')(outputs)
    
    model = Model(inputs, outputs)
    return model

# Verlustfunktionen
def weighted_log_loss(y_true, y_pred):
    weights = tf.constant([1.0, 2.0, 0.5, 1.5, 1.0, 2.5, 0.5, 3.0, 1.0, 2.0])  # Beispielwerte
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = -K.sum(y_true * K.log(y_pred) * weights, axis=-1)
    return K.mean(loss)

def dice_loss(y_true, y_pred):
    numerator = 2 * K.sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = K.sum(y_true + y_pred, axis=(1, 2, 3))
    return 1 - numerator / denominator

def combined_loss(y_true, y_pred):
    return weighted_log_loss(y_true, y_pred) + dice_loss(y_true, y_pred)
