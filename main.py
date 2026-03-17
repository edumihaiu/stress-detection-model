import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

img_size = (64, 64)
batch_size = 64

train_dataset = tf.keras.utils.image_dataset_from_directory(
    "fer2013/train",
    image_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    "fer2013/test",
    image_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
)

class_names = train_dataset.class_names
print("Class names:", class_names)

stress_level_map = {
    "neutral": 0,
    "happy": 0,
    "surprise": 0,
    "sad": 1,
    "angry": 1,
    "fear": 1,
    "disgust": 1,   
}

stress_per_class = tf.constant([stress_level_map[name] for name in class_names],
                               dtype=tf.int32
                               )

print(stress_per_class)

def emotion_to_stress(images, labels):
    stress_labels = tf.gather(stress_per_class, labels)
    images = tf.cast(images, tf.float32) / 255.0
    return images, stress_labels

print(train_dataset)
print(test_dataset)

train_dataset = train_dataset.map(emotion_to_stress).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.map(emotion_to_stress).prefetch(tf.data.AUTOTUNE)

print(train_dataset)
print(test_dataset)

num_stress_classes = 3  # low / medium / high

model = keras.Sequential([
    layers.Input(shape=img_size + (1,)),  # 64x64x1

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_stress_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=15
)

# Grafic loss
plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')
plt.savefig('loss.png')

# Grafic accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy')
plt.savefig('accuracy.png')