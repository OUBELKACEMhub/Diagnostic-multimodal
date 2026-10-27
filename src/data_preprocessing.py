import os
import random
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLIT_NAMES = ("train", "validation", "test")


def inspecter_dataset_images(source_dir):
    """Retourne les chemins d'images lisibles et les fichiers invalides sans modifier les sources."""
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Dossier de données introuvable : {source_dir}")

    images = []
    fichiers_invalides = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            if "images" in path.parts:
                fichiers_invalides.append(
                    {"path": str(path), "reason": "extension non prise en charge"}
                )
            continue

        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.convert("RGB").load()
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
            fichiers_invalides.append({"path": str(path), "reason": str(error)})
            continue

        parts = path.relative_to(source_dir).parts
        if "images" in parts:
            image_folder_index = parts.index("images")
            if image_folder_index == 0:
                fichiers_invalides.append(
                    {"path": str(path), "reason": "classe introuvable dans le chemin"}
                )
                continue
            label = parts[image_folder_index - 1]
        else:
            class_index = 1 if parts[0].lower() in {"train", "val", "valid", "test"} else 0
            if len(parts) <= class_index + 1:
                fichiers_invalides.append(
                    {"path": str(path), "reason": "classe introuvable dans le chemin"}
                )
                continue
            label = parts[class_index]

        images.append({"path": str(path), "label": label})

    return pd.DataFrame(images, columns=["path", "label"]), pd.DataFrame(
        fichiers_invalides, columns=["path", "reason"]
    )


def repartir_images(images, taille_test=0.15, taille_validation=0.15, seed=42):
    """Effectue un split stratifié train/validation/test à partir des chemins valides."""
    if not 0 < taille_test < 1 or not 0 < taille_validation < 1:
        raise ValueError("Les proportions test et validation doivent être comprises entre 0 et 1.")
    if taille_test + taille_validation >= 1:
        raise ValueError("La somme des proportions test et validation doit rester inférieure à 1.")
    if images.empty or not {"path", "label"}.issubset(images.columns):
        raise ValueError("Le tableau d'images doit contenir au moins une ligne et les colonnes path/label.")
    if images["label"].value_counts().min() < 3:
        raise ValueError("Chaque classe doit contenir au moins trois images pour le split stratifié.")

    train_validation, test = train_test_split(
        images,
        test_size=taille_test,
        random_state=seed,
        stratify=images["label"],
    )
    validation_fraction = taille_validation / (1 - taille_test)
    train, validation = train_test_split(
        train_validation,
        test_size=validation_fraction,
        random_state=seed,
        stratify=train_validation["label"],
    )

    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def _link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _augment_train(images, labels):
    import tensorflow as tf

    images = tf.image.random_flip_left_right(images)
    channels = tf.shape(images)[-1]
    blur_kernel = tf.ones(tf.stack([3, 3, channels, 1]), dtype=images.dtype) / 9
    images = tf.cond(
        tf.random.uniform(()) < 0.3,
        lambda: tf.nn.depthwise_conv2d(
            images,
            blur_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        ),
        lambda: images,
    )
    noise = tf.random.normal(tf.shape(images), stddev=0.03, dtype=images.dtype)
    images = tf.clip_by_value(images + noise, 0.0, 1.0)
    return images, labels


def creer_datasets_images(
    repartitions,
    taille_image=224,
    taille_lot=32,
    seed=42,
    augmenter=True,
    equilibrer=True,
):
    """Crée les datasets TensorFlow; augmentation et rééquilibrage ne touchent que Train."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    if set(repartitions) != set(SPLIT_NAMES):
        raise ValueError(f"Les splits requis sont : {', '.join(SPLIT_NAMES)}.")

    class_names = sorted(
        set().union(*(set(split["label"].unique()) for split in repartitions.values()))
    )
    if not class_names:
        raise ValueError("Aucune classe trouvée dans les splits.")

    workspace = tempfile.TemporaryDirectory(prefix="brain_images_")
    workspace_path = Path(workspace.name)
    effectifs = {split: len(repartitions[split]) for split in SPLIT_NAMES}
    train_rows = repartitions["train"].to_dict("records")
    if equilibrer:
        rows_by_class = {
            name: [row for row in train_rows if row["label"] == name]
            for name in class_names
        }
        if any(not rows for rows in rows_by_class.values()):
            workspace.cleanup()
            raise ValueError("Chaque classe doit être présente dans le split d'entraînement.")
        target_count = max(len(rows) for rows in rows_by_class.values())
        rng = random.Random(seed)
        train_rows = []
        for name in class_names:
            rows = rows_by_class[name]
            train_rows.extend(rows)
            train_rows.extend(rng.choices(rows, k=target_count - len(rows)))
        rng.shuffle(train_rows)

    rows_by_split = {
        "train": train_rows,
        "validation": repartitions["validation"].to_dict("records"),
        "test": repartitions["test"].to_dict("records"),
    }
    datasets = {}
    try:
        for split in SPLIT_NAMES:
            split_root = workspace_path / split
            for class_name in class_names:
                (split_root / class_name).mkdir(parents=True, exist_ok=True)

            for index, row in enumerate(rows_by_split[split]):
                source = Path(row["path"])
                if not source.is_file():
                    raise FileNotFoundError(f"Image du split introuvable : {source}")
                destination = split_root / row["label"] / f"{index:08d}{source.suffix.lower()}"
                _link_or_copy(source, destination)

            dataset = tf.keras.utils.image_dataset_from_directory(
                split_root,
                labels="inferred",
                label_mode="int",
                class_names=class_names,
                image_size=(taille_image, taille_image),
                batch_size=taille_lot,
                shuffle=(split == "train"),
                seed=seed,
            )

            def normalize_inception(images, labels):
                images = tf.keras.applications.inception_v3.preprocess_input(
                    tf.cast(images, tf.float32)
                )
                return images, labels

            if split == "train" and augmenter:

                def scale_for_augmentation(images, labels):
                    return tf.cast(images, tf.float32) / 255.0, labels

                def normalize_augmented(images, labels):
                    return (
                        tf.keras.applications.inception_v3.preprocess_input(images * 255.0),
                        labels,
                    )

                dataset = dataset.map(
                    scale_for_augmentation, num_parallel_calls=tf.data.AUTOTUNE
                )
                dataset = dataset.map(_augment_train, num_parallel_calls=tf.data.AUTOTUNE)
                dataset = dataset.map(
                    normalize_augmented, num_parallel_calls=tf.data.AUTOTUNE
                )
            else:
                dataset = dataset.map(
                    normalize_inception, num_parallel_calls=tf.data.AUTOTUNE
                )
            datasets[split] = dataset.prefetch(tf.data.AUTOTUNE)
    except Exception:
        workspace.cleanup()
        raise

    return datasets, workspace, class_names, effectifs


def clean_and_verify_images(source_dir):
    """Affiche le nombre d'images valides/invalides sans supprimer les fichiers source."""
    images, fichiers_invalides = inspecter_dataset_images(source_dir)
    print(
        f"[INFO] Vérification terminée : {len(images)} images valides, "
        f"{len(fichiers_invalides)} fichiers ignorés."
    )
    return images, fichiers_invalides
