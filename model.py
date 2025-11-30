import tensorflow as tf
from tensorflow.keras import layers, models

def cnns_model():

    height = 100
    width = 100
    danse_info = 1024

    model = tf.keras.models.Sequential([

        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, 1)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(height, width, 1)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(height, width, 1)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(height, width, 1)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(height, width, 1)),


        layers.Flatten(),
        layers.Dense(danse_info, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(danse_info, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(29, activation='softmax')


        
    ])

    return model