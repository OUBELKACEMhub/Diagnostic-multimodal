import json
from pathlib import Path

import mlflow
import mlflow.keras
import tensorflow as tf

from src.data_preprocessing import (
    creer_datasets_images,
    inspecter_dataset_images,
    repartir_images,
)
from src.evaluate import evaluate_classifier
from src.model_architecture import build_inception_model, prepare_for_fine_tuning


def run_training():
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    images, invalid_files = inspecter_dataset_images(raw_dir)
    if images.empty:
        raise ValueError(f"Aucune image valide trouvée dans {raw_dir}.")

    repartitions = repartir_images(images, taille_test=0.15, taille_validation=0.15, seed=42)
    datasets, split_workspace, class_names, split_counts = creer_datasets_images(
        repartitions,
        taille_image=224,
        taille_lot=32,
        seed=42,
    )
    try:
        models_dir = project_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        with (models_dir / "class_names.json").open("w", encoding="utf-8") as file:
            json.dump(class_names, file, ensure_ascii=False, indent=2)

        mlflow.set_experiment("Brain_Tumor_InceptionV3")
        with mlflow.start_run(run_name="InceptionV3_Two_Phase_Training"):
            mlflow.log_params(
                {
                    "lr_feature_extraction": 1e-3,
                    "epochs_feature_extraction": 10,
                    "lr_fine_tuning": 1e-5,
                    "epochs_fine_tuning": 10,
                    "batch_size": 32,
                    "class_count": len(class_names),
                    "invalid_files": len(invalid_files),
                }
            )
            mlflow.log_metrics(
                {f"{split}_images": count for split, count in split_counts.items()}
            )

            model, base_model = build_inception_model(len(class_names))
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            print("[INFO] Lancement Phase 1 : extraction des caractéristiques...")
            model.fit(datasets["train"], validation_data=datasets["validation"], epochs=10)

            model = prepare_for_fine_tuning(model, base_model, unfreeze_from_layer=249)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            print("[INFO] Lancement Phase 2 : fine-tuning...")
            model.fit(datasets["train"], validation_data=datasets["validation"], epochs=10)

            metrics = evaluate_classifier(model, datasets["test"], class_names)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact("confusion_matrix.png")

            model_path = models_dir / "final_model.keras"
            model.save(model_path)
            mlflow.keras.log_model(model, "model")
            print(f"[INFO] Modèle sauvegardé : {model_path}")
    finally:
        split_workspace.cleanup()


if __name__ == "__main__":
    run_training()
