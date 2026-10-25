import os
import shutil
import random
import numpy as np
from PIL import Image, ImageFilter
import tensorflow as tf

def clean_and_verify_images(source_dir):
    """Vérifie l'intégrité de chaque image avec try-except et supprime les corrompues."""
    valid_extensions = {".jpg", ".jpeg", ".png"}
    corrupted_count = 0
    for root, _, files in os.walk(source_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            file_path = os.path.join(root, file)
            if ext not in valid_extensions:
                os.remove(file_path)
                continue
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception:
                corrupted_count += 1
                os.remove(file_path)
    print(f"[INFO] Nettoyage termine. Images corrompues supprimees: {corrupted_count}")

def split_dataset(raw_dir, split_dir, train_ratio=0.7, val_ratio=0.15):
    """Divise les données en Train (70%), Validation (15%), Test (15%)."""
    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    for split in ["train", "valid", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(split_dir, split, cls), exist_ok=True)

    for cls in classes:
        cls_dir = os.path.join(raw_dir, cls)
        images = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for img in train_imgs:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(split_dir, "train", cls, img))
        for img in val_imgs:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(split_dir, "valid", cls, img))
        for img in test_imgs:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(split_dir, "test", cls, img))

def augment_image(image_path, save_path_prefix):
    """Applique Flou, Bruit et Retournement sur une image."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            # 1. Flip horizontal
            flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            flipped.save(f"{save_path_prefix}_flip.jpg")

            # 2. Blur (Flou)
            blurred = img.filter(ImageFilter.GaussianBlur(radius=1.5))
            blurred.save(f"{save_path_prefix}_blur.jpg")

            # 3. Noise (Bruit gaussien)
            arr = np.array(img).astype(np.float32)
            noise = np.random.normal(0, 15, arr.shape)
            noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
            Image.fromarray(noisy).save(f"{save_path_prefix}_noise.jpg")
    except Exception as e:
        print(f"Erreur d'augmentation sur {image_path}: {e}")

def augment_train_data(train_dir):
    """Applique l'augmentation uniquement sur le dossier d'entraînement."""
    for root, _, files in os.walk(train_dir):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")) and not any(k in file for k in ["_flip", "_blur", "_noise"]):
                img_path = os.path.join(root, file)
                prefix = os.path.splitext(img_path)[0]
                augment_image(img_path, prefix)

def load_datasets(data_dir, img_size=(224, 224), batch_size=32):
    """Charge les datasets avec imagedatasetfrom_directory et applique la normalisation InceptionV3."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "valid"),
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "test"),
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    # Normalisation selon InceptionV3 (pixels de [-1, 1])
    normalization_layer = tf.keras.applications.inception_v3.preprocess_input
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, val_ds, test_ds