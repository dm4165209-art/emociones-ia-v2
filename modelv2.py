import cv2
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array  # <-- cambio aquí
import os

# -----------------------------
# RUTAS DE ARCHIVOS
# -----------------------------
model_path = "C:/Users/dm416/OneDrive/Desktop/emotion_project/Emotion-Detection-with-Face/best_model.h5"
haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# -----------------------------
# CARGAR MODELO
# -----------------------------
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No se encontró el archivo del modelo: {model_path}")

model = load_model(model_path)
print("Modelo cargado correctamente.")

# -----------------------------
# CARGAR HAAR CASCADE
# -----------------------------
face_cascade = cv2.CascadeClassifier(haar_path)
if face_cascade.empty():
    raise ValueError("No se pudo cargar Haar Cascade. Verifica la ruta.")

# Lista de emociones
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# -----------------------------
# INICIALIZAR CAMARA
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise ValueError("No se pudo acceder a la cámara. Verifica que esté conectada.")

print("=== Sistema de Detección de Emociones en Tiempo Real ===")
print("Presiona 'q' para salir.\n")

# -----------------------------
# BUCLE PRINCIPAL
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi = frame[y:y+h, x:x+w]
        roi = cv2.resize(roi, (224, 224))
        img_pixels = img_to_array(roi)  # <-- aquí usamos la versión correcta
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels = img_pixels.astype('float32') / 255.0  # normalización

        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotions[max_index]

        cv2.putText(frame, predicted_emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow('Emotion Detector', cv2.resize(frame, (1000, 700)))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Programa finalizado.")
