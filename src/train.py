import os
import mlflow
import mlflow.keras
import tensorflow as tf
from src.data_preprocessing import load_datasets
from src.model_architecture import build_inception_model, prepare_for_fine_tuning
from src.evaluate import evaluate_classifier

def run_training():
    mlflow.set_experiment("Brain_Tumor_InceptionV3")
    
    data_dir = "data/split_data"
    batch_size = 32
    train_ds, val_ds, test_ds = load_datasets(data_dir, batch_size=batch_size)
    class_names = os.listdir(os.path.join(data_dir, "train"))
    num_classes = len(class_names)

    with mlflow.start_run(run_name="InceptionV3_Two_Phase_Training"):
        # Phase 1: Feature Extraction
        lr_fe = 1e-3
        epochs_fe = 10
        mlflow.log_params({"lr_feature_extraction": lr_fe, "epochs_fe": epochs_fe, "batch_size": batch_size})

        model, base_model = build_inception_model(num_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_fe),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        print("[INFO] Lancement Phase 1 : Feature Extraction...")
        model.fit(train_ds, validation_data=val_ds, epochs=epochs_fe)

        # Phase 2: Fine-Tuning
        lr_ft = 1e-5
        epochs_ft = 10
        mlflow.log_params({"lr_fine_tuning": lr_ft, "epochs_ft": epochs_ft})

        model = prepare_for_fine_tuning(model, base_model, unfreeze_from_layer=249)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_ft),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        print("[INFO] Lancement Phase 2 : Fine-Tuning...")
        model.fit(train_ds, validation_data=val_ds, epochs=epochs_ft)

        # Évaluation finale sur test
        metrics = evaluate_classifier(model, test_ds, class_names)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("confusion_matrix.png")

        # Sauvegarde
        os.makedirs("models", exist_ok=True)
        model.save("models/inception_brain_tumor.keras")
        mlflow.keras.log_model(model, "model")
        print("[INFO] Modèle sauvegardé avec succès.")

if __name__ == "__main__":
    run_training()