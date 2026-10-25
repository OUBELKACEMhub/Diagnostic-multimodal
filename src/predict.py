
import cv2
import numpy as np
from PIL import Image


def predict_image(path, model, le, taille_image=224):
    """
    Predis la classe d'une image depuis son chemin.
    Exemple:
        label, conf = predict_image("data/raw/glioma/image.jpg", model, le)
        print(label, conf)  # "glioma", 92.5
    """
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Impossible de lire : {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (taille_image, taille_image))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # ajoute dimension batch

    pred = model.predict(img, verbose=0)[0]
    idx = int(np.argmax(pred))
    confiance = float(np.max(pred) * 100)
    label = str(le.inverse_transform([idx])[0])
    return label, confiance


def predict_from_pil(pil_image, model, le, taille_image=224):
    """
    Predis depuis une image PIL (utilise par Streamlit).
    Retourne (label, confiance, {classe: proba}).
    """
    img = np.array(pil_image.convert("RGB"))
    img = cv2.resize(img, (taille_image, taille_image))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0]
    idx = int(np.argmax(pred))
    confiance = float(np.max(pred) * 100)
    label = str(le.inverse_transform([idx])[0])
    proba_dict = {str(le.classes_[i]): float(pred[i] * 100) for i in range(len(le.classes_))}

    return label, confiance, proba_dict
