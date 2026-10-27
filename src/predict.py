
import numpy as np
from PIL import Image
from tensorflow.keras.applications.inception_v3 import preprocess_input


def _prepare_image(image, taille_image):
    image = image.convert("RGB").resize((taille_image, taille_image))
    image_array = np.asarray(image, dtype=np.float32)
    image_array = preprocess_input(image_array)
    return np.expand_dims(image_array, axis=0)


def predict_image(path, model, le, taille_image=224):
    """
    Prédit la classe d'une image depuis son chemin.
    Exemple:
        label, conf = predict_image("data/raw/Train/Glioma/images/image.jpg", model, le)
        print(label, conf)  # "Glioma", 92.5
    """
    with Image.open(path) as image:
        img = _prepare_image(image, taille_image)

    pred = model.predict(img, verbose=0)[0]
    idx = int(np.argmax(pred))
    confiance = float(np.max(pred) * 100)
    label = str(le.inverse_transform([idx])[0])
    return label, confiance


def predict_from_pil(pil_image, model, le, taille_image=224):
    """
    Prédit depuis une image PIL (utilisé par Streamlit).
    Retourne (label, confiance, {classe: proba}).
    """
    img = _prepare_image(pil_image, taille_image)

    pred = model.predict(img, verbose=0)[0]
    idx = int(np.argmax(pred))
    confiance = float(np.max(pred) * 100)
    label = str(le.inverse_transform([idx])[0])
    proba_dict = {str(le.classes_[i]): float(pred[i] * 100) for i in range(len(le.classes_))}

    return label, confiance, proba_dict
