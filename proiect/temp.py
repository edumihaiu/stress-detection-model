import tensorflow as tf
import config
from dataset import load_datasets
import numpy as np

train_ds, test_ds = load_datasets()

# verificam ce labeluri primeste modelul
print("=== SAMPLE LABELURI ===")
for images, labels in test_ds.take(3):
    unique, counts = np.unique(labels.numpy(), return_counts=True)
    print(f"Clase găsite: {unique}, Counts: {counts}")

# verificam distributia pe tot test set
all_labels = []
for images, labels in test_ds:
    all_labels.extend(labels.numpy())

all_labels = np.array(all_labels)
print(f"\n=== DISTRIBUTIE TEST SET ===")
for i in range(3):
    count = np.sum(all_labels == i)
    print(f"Clasa {i}: {count} ({count/len(all_labels)*100:.1f}%)")

# verificam daca modelul prezice mereu aceeasi clasa
from model import build_model
model = build_model()
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

preds = []
for images, labels in test_ds.take(5):
    p = model(images, training=False)
    preds.extend(np.argmax(p.numpy(), axis=1))

print(f"\n=== PREDICTII MODEL NEANTRENAT ===")
unique, counts = np.unique(preds, return_counts=True)
print(f"Clase prezise: {unique}, Counts: {counts}")