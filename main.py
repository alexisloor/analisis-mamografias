import io
import base64
import numpy as np
import cv2
import pydicom
import tensorflow as tf
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model


# Crear instancia de FastAPI
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


# Configurar Jinja2 para usar la carpeta 'templates'
templates = Jinja2Templates(directory="templates")

# Cargar del modelo preentrenado (archivo.h5)
model = load_model("model/eficcientnetB6.h5")

# Función para cargar y preprocesar la imagen DICOM
def load_dicom_image_from_bytes(file_bytes):
    ds = pydicom.dcmread(io.BytesIO(file_bytes))
    img = ds.pixel_array.astype(np.float32)

    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)

    img = cv2.resize(img, (299, 299))

    if np.max(img) > 0:
        img = img / np.max(img)
    return img

# Función que hace la predicción y aplica la técnica grad-cam
def apply_gradcam(model, img_array, original_image, img_size=(299, 299), intensity=0.5):
    # Realizar la predicción
    preds = model.predict(img_array)
    pred_class = np.argmax(preds[0])
    confidence = preds[0][pred_class] * 100

    print(f"Clase predicha: {pred_class}")

    # Identificar la última capa Conv2D del modelo
    last_conv_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_name = layer.name
            break
    print("Última capa Conv2D:", last_conv_name)

    # Crear un modelo que devuelve tanto la salida como la última capa convolucional
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.output, model.get_layer(last_conv_name).output]
    )

    with tf.GradientTape() as tape:
        predictions, conv_outputs = grad_model(img_array)
        chosen_class = predictions[:, pred_class]
        grads = tape.gradient(chosen_class, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0] * pooled_grads
    heatmap = tf.reduce_mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)

    # Normalizar el heatmap a [0, 1]
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    # Redimensionar el heatmap al tamaño de la imagen original
    heatmap_resized = cv2.resize(heatmap, img_size)
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # Preparar la imagen original para la superposición
    # Se asume que la imagen original está en [0, 1]
    orig_img_uint8 = np.uint8(original_image * 255)

    # Superponer el heatmap sobre la imagen original
    superimposed_img = cv2.addWeighted(orig_img_uint8, 1 - intensity, heatmap_color, intensity, 0)

    return pred_class, confidence, heatmap_color, superimposed_img

# Función auxiliar para convertir un array de imagen a una cadena Base64 (formato PNG)
def array_to_base64(img_array):
    # Convertir de BGR a RGB para que los colores se muestren correctamente
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    retval, buffer = cv2.imencode('.png', img_rgb)
    img_bytes = buffer.tobytes()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_b64

# Endpoint GET: muestra la página principal con el formulario
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    # Se renderiza la plantilla sin resultados (variables vacías)
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint POST: procesa la imagen dicom cargada y devuelve: predicción, nivel de confianza, imagen original y Grad-CAM
@app.post("/", response_class=HTMLResponse)
async def analyze(request: Request, dicom_file: UploadFile = File(...)):
    # Leer el archivo subido
    file_bytes = await dicom_file.read()

    # Cargar y preprocesar la imagen DICOM
    image = load_dicom_image_from_bytes(file_bytes)
    img_array = np.expand_dims(image, axis=0)  

    # Realizar la predicción y aplicar Grad-CAM 
    pred_class, confidence, heatmap, superimposed_img = apply_gradcam(
        model, img_array, image
    )

    # Convertir la predicción a texto
    label_dict = {0: "Maligno", 1: "Benigno"}
    pred_label_text = label_dict.get(pred_class, "Desconocido")

    # Convertir la predicción a texto y analizar la confianza
    label_dict = {0: "Maligno", 1: "Benigno"}
    pred_label_text = label_dict.get(pred_class, "Desconocido")

    # Evaluar el nivel de confianza para Benigno
    confidence_msg = ""
    if pred_class == 1:
        if confidence >= 85:
            confidence_msg = "Clasificación Benigna con alta confianza."
        elif 65 <= confidence < 85:
            confidence_msg = "⚠️ Clasificación Benigna con baja confianza. Se recomienda análisis adicional para confirmación."
        else:
            confidence_msg = "🚨 Clasificación Benigna con muy baja confianza. Se recomienda evaluación médica inmediata para confirmación."
    elif pred_class == 0:
        confidence_msg = "Clasificación Maligna. Evaluación clínica prioritaria sugerida."


    # Convertir las imágenes a Base64 para integrarlas en el HTML
    original_b64 = array_to_base64(np.uint8(image * 255))
    heatmap_b64 = array_to_base64(heatmap)
    superimposed_b64 = array_to_base64(superimposed_img)

    # Renderizar la plantilla index.html con los resultados
    return templates.TemplateResponse("index.html", {
        "request": request,
        "original_b64": original_b64,
        "heatmap_b64": heatmap_b64,
        "superimposed_b64": superimposed_b64,
        "pred_label_text": pred_label_text,
        "confidence": f"{confidence:.2f}",
        "confidence_msg": confidence_msg
    })