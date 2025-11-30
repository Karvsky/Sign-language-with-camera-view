import tensorflow as tf

def data_import():
    directory = r'C:\\Users\\Karol\\Documents\\nauka_ai\\Computer Vision\\Sign language\\Sign-language-with-camera-view\\dataset\\asl_alphabet_train\\asl_alphabet_train'

    img_height = 100
    img_width = 100
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=0.2,
        subset='training',
        seed=100,
        image_size=(img_height, img_width), 
        color_mode='grayscale',
        batch_size=32
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=0.2,
        subset='validation',
        seed=100,
        image_size=(img_height, img_width), 
        color_mode='grayscale',
        batch_size=32
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, test_ds

