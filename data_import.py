import tensorflow as tf

def data_import():

    directory = r'C:\\Users\\Karol\\Documents\\nauka_ai\\Computer Vision\\Sign language\\Sign-language-with-camera-view\\dataset\\asl_alphabet_train\\asl_alphabet_train'

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=0.2,
        subset='training',
        seed = 100,
        image_size=(150,180),
        color_mode='grayscale',
        batch_size=32
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=0.2,
        subset='validation',
        seed = 100,
        image_size=(150,180),
        color_mode='grayscale',
        batch_size=32
    )

    return train_ds, test_ds

