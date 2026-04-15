# model_4blocks.py
import tensorflow as tf
import config

def build_model():
    model = tf.keras.models.Sequential([
        # Input: Poze 64x64, alb-negru (1 singur canal)
        tf.keras.layers.Input(shape=(config.IMG_SIZE[0], config.IMG_SIZE[1], 1)),
        
        # block 1
        tf.keras.layers.Conv2D(32, (3,3), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(32, (3,3), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        
        # block 2
        tf.keras.layers.Conv2D(64, (3,3), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        
        # block 3
        tf.keras.layers.Conv2D(128, (3,3), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        
        # --- PIESA NOUA: block 4 (stage 2 tuning pentru putere bruta) ---
        tf.keras.layers.Conv2D(256, (3,3), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.3),
        
        # head (pooling2d)    
        tf.keras.layers.GlobalAveragePooling2D(), 
        
        # marit carburatorul la 256
        tf.keras.layers.Dense(256, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5), # frana de mana mai puternica anti-overfitting
        
        # output (0: happy 1: neutral 2: stressed)
        tf.keras.layers.Dense(config.NUM_CLASSES, activation='softmax')
    ])
    
    return model

if __name__ == "__main__":
    motor = build_model()
    motor.summary()