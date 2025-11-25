# Detección de emociones faciales

Análisis de emociones en tiempo real a partir de la cámara usando deep learning. El modelo se carga desde un archivo Keras preentrenado (por ejemplo, `best_model.h5`) y se utiliza OpenCV para captura y detección de rostros.

## Requisitos

Instalar las librerías necesarias:

```bash
pip install numpy opencv-python keras matplotlib
```

Además, descarga estos archivos y colócalos junto al script o proporciona rutas absolutas:

- `best_model.h5` — modelo Keras preentrenado.
- `haarcascade_frontalface_default.xml` — clasificador Haar Cascade (disponible en el repositorio de OpenCV).

## Uso

1. Coloca `emotion_detection.py`, `best_model.h5` y `haarcascade_frontalface_default.xml` en la misma carpeta (o ajusta las rutas en el script).
2. En el script, carga el modelo así:

```python
from keras.models import load_model
model = load_model("best_model.h5")
```

3. Ejecuta:

```bash
python emotion_detection.py
```

4. Pulsa `q` para cerrar la ventana y detener la aplicación.

## Descripción del funcionamiento

- Se captura vídeo desde la cámara con OpenCV.
- Se detectan rostros con el clasificador Haar Cascade.
- Para cada rostro detectado se recorta la región de interés (ROI), se preprocesa (redimensionar, normalizar, conversión a escala de grises si aplica) y se pasa al modelo para predecir la emoción.
- La emoción predicha se dibuja como texto sobre el fotograma en tiempo real.

## Emociones reconocidas

- Angry (Enfadado)  
- Disgust (Asco)  
- Fear (Miedo)  
- Happy (Feliz)  
- Sad (Triste)  
- Surprise (Sorpresa)  
- Neutral (Neutral)

## Solución de problemas

- Error al cargar el modelo: verifica que `best_model.h5` exista y la ruta sea correcta.
- Problemas con la cámara: confirma que esté conectada y que otra aplicación no la esté usando.
- Dependencias: reinstala las librerías o crea un entorno virtual limpio.
- Haar Cascade no detecta rostros: prueba con otra versión del archivo XML o ajusta parámetros de detección (escala, vecinos).

Si necesitas adaptar el script a otro modelo, revisa las dimensiones de entrada y la normalización que espera tu modelo. ¡Feliz codificación!
