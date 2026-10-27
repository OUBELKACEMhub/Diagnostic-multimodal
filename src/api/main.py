from fastapi import FastAPI, UploadFile, File
import numpy as np
from PIL import Image
import io
import json
from pathlib import Path
import tensorflow as tf
from ultralytics import YOLO

app = FastAPI(title="Brain Tumor API")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
clf_model = tf.keras.models.load_model(MODELS_DIR / "final_model.keras")
with (MODELS_DIR / "class_names.json").open(encoding="utf-8") as file:
    classes = json.load(file)

_det_model = None


def get_detection_model():
    global _det_model
    if _det_model is None:
        model_path = MODELS_DIR / "yolov8_best.pt"
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Poids YOLO manquants : {model_path}. Entraînez d'abord le modèle de détection."
            )
        _det_model = YOLO(str(model_path))
    return _det_model

@app.post("/predict/classify")
async def classify_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    img_array = np.asarray(image, dtype=np.float32)
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
    det_model = get_detection_model()
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