import cv2
import numpy as np
import tensorflow as tf
import os

# 0. Curatam terminalul de erori inutile
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def focal_loss_fn(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    y_true = tf.cast(y_true, tf.float32)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    cross_entropy = -y_true * tf.math.log(y_pred)
    weight = alpha * y_true * tf.math.pow((1.0 - y_pred), gamma)
    loss = weight * cross_entropy
    return tf.math.reduce_sum(loss, axis=1)

# 1. Configurarile tale de baza
MODEL_PATH = "../80epoci.keras"
IMG_SIZE = (64, 64)
CLASSES = ['relaxat', 'neutru', 'stresat']

# 2. Incarcare Model
print("[*] Se monteaza filtrul CLAHE si se porneste motorul...")
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'focal_loss_fn': focal_loss_fn})

# 3. Initializare filtru de lumina (CLAHE)
# Asta face minuni cu umbrele de pe fata!
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

print("[OK] Testam acum cu lumina reglata automat. Apasa 'q' pentru stop.")

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # --- AICI E MANEVRA A ---
    # Aplicam filtrul pe toata imaginea gri inainte de detectie
    gray_balanced = clahe.apply(gray)
    
    faces = face_cascade.detectMultiScale(gray_balanced, 1.3, 5)

    for (x, y, w, h) in faces:
        # Decupam din imaginea filtrata, nu din aia bruta!
        roi = gray_balanced[y:y+h, x:x+w]
        roi = cv2.resize(roi, IMG_SIZE)
        
        # Normalizare standard
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        prediction = model.predict(roi, verbose=0)
        
        # Strategie de "barosan": Daca e prea aproape de Neutru, alegem Neutru!
        # Daca Stresat are 0.51 si Neutru 0.49, nu vrem sa sara la rosu imediat.
        probs = prediction[0]
        idx = np.argmax(probs)
        
        # Daca diferenta intre Stresat si restul e mica, il tinem pe Neutru
        if idx == 2 and probs[2] < 0.6: # Daca Stresat e sub 60%, suntem precauti
             if probs[1] > 0.2: # Daca avem si ceva Neutru in rezervor
                idx = 1 

        label = CLASSES[idx]
        conf = probs[idx] * 100

        color = (0, 255, 0) if label == 'relaxat' else (0, 255, 255)
        if label == 'stresat': color = (0, 0, 255)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{label} ({conf:.1f}%)", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Afisam imaginea balansata intr-o fereastra mica sa vezi diferenta
    cv2.imshow('Ce vede AI-ul (Filtrat)', cv2.resize(gray_balanced, (300, 200)))
    cv2.imshow('Rezultat Live', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()