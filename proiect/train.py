# train.py
import os
import zipfile
import tensorflow as tf
import config
from dataset import load_datasets
from model import build_model

def extract_data():
    # extract zip in colab
    zip_path = "/content/drive/MyDrive/dataset_curat.zip"
    extract_dir = "/content/dataset_curat"
    
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("/content/")
            
    # update config path
    config.DATA_DIR = extract_dir

def calculate_alpha_weights(data_dir):
    # calcul dinamic pentru ponderile claselor (alpha) in functie de distributia reala a datelor
    counts = []
    for i in range(config.NUM_CLASSES):
        class_path = os.path.join(data_dir, "train", str(i))
        counts.append(len(os.listdir(class_path)))
    
    total_samples = sum(counts)
    # formula standard: pondere invers proportionala cu frecventa clasei
    alpha = [total_samples / (config.NUM_CLASSES * count) for count in counts]
    print(f"[*] distributie fisiere: {counts}")
    print(f"[*] ponderi alpha calculate: {alpha}")
    return alpha

# focal loss (handles class imbalance) cu suport pentru alpha ca array (multi-class)
def sparse_focal_loss(gamma=2.0, alpha_weights=None):
    alpha_tensor = tf.constant(alpha_weights, dtype=tf.float32)
    
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=config.NUM_CLASSES)
        
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # calcul standard cross-entropy
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        
        # probabilitatea prezisa pentru clasa corecta
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1, keepdims=True)
        
        # aplicare factor de focalizare: (1 - p_t)^gamma
        focal_factor = tf.pow(1.0 - p_t, gamma)
        
        # calcul pierdere finala ponderata cu alpha
        loss = y_true_one_hot * alpha_tensor * focal_factor * cross_entropy
        
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    
    return focal_loss_fn

def train():
    extract_data()
    calculated_alpha = calculate_alpha_weights(config.DATA_DIR)
    train_ds, test_ds = load_datasets()
    model = build_model()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),  # schimbat
        loss=sparse_focal_loss(gamma=2.0, alpha_weights=calculated_alpha),
        metrics=['accuracy']
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="/content/drive/MyDrive/best_model.keras",
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
        verbose=1
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=config.EPOCHS,
        callbacks=[checkpoint, early_stop, reduce_lr]  # toate 3 aici
    )

if __name__ == "__main__":
    train()