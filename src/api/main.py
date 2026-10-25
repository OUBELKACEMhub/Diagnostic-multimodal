from fastapi import FastAPI, UploadFile, File
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from ultralytics import YOLO

app = FastAPI(title="Brain Tumor API")

# Chargement des modèles
clf_model = tf.keras.models.load_model("models/inception_brain_tumor.keras")
det_model = YOLO("models/yolov8_best.pt")
classes = ["glioma", "meningioma", "no_tumor", "pituitary"]

@app.post("/predict/classify")
async def classify_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    img_array = np.array(image)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    preds = clf_model.predict(img_array)
    pred_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_idx])

    return {"class": classes[pred_idx], "confidence": round(confidence * 100, 2)}

@app.post("/predict/detect")
async def detect_tumor(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    results = det_model(image)
    
    detections = []
    for box in results[0].boxes:
        coords = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        detections.append({
            "bbox": coords,
            "confidence": round(conf * 100, 2),
            "class": det_model.names[cls]
        })

    return {"detections": detections}