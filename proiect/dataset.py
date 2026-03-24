# dataset.py
import tensorflow as tf
import config

# randomize photo (augmentation)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomBrightness(0.15),
    tf.keras.layers.RandomContrast(0.15),
])

def prepare_data(ds, augment=False):
    AUTOTUNE = tf.data.AUTOTUNE
    
    # from 0-255 to 0.0-1.0 (float)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=AUTOTUNE)
    
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
        
    return ds.prefetch(buffer_size=AUTOTUNE)

def load_datasets():
    print("load train data")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{config.DATA_DIR}/train",
        image_size=config.IMG_SIZE,
        color_mode=config.COLOR_MODE,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        seed=42
    )

    print("load test data")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        f"{config.DATA_DIR}/test",
        image_size=config.IMG_SIZE,
        color_mode=config.COLOR_MODE,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    train_ds = prepare_data(train_ds, augment=True)
    test_ds = prepare_data(test_ds, augment=False)

    return train_ds, test_ds

if __name__ == "__main__":
    train, test = load_datasets()
    print('dataset-uri incarcate si optimizate')