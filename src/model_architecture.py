

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionV3


def conception_modele_cnn(input_shape=(224, 224, 3), nb_classes=4):
    model = Sequential([
        # Bloc 1 : 32 filtres
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Bloc 2 : 64 filtres
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Bloc 3 : 128 filtres
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        # Bloc 4 : 256 filtres
        Conv2D(256, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        # Tete du modele
        GlobalAveragePooling2D(),       # moyenne au lieu de Flatten
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(nb_classes, activation="softmax"),  # 4 classes
    ])
    return model


def compiler_modele(model, learning_rate=0.0001):
    
    from tensorflow.keras.optimizers import Adam

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    return model


def build_inception_model(num_classes, input_shape=(224, 224, 3)):
    """Construit un classificateur InceptionV3 prêt pour l'extraction de caractéristiques."""
    base_model = InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    features = base_model(inputs, training=False)
    features = GlobalAveragePooling2D()(features)
    features = Dropout(0.3)(features)
    outputs = Dense(num_classes, activation="softmax")(features)
    model = Model(inputs=inputs, outputs=outputs)

    return model, base_model


def prepare_for_fine_tuning(model, base_model, unfreeze_from_layer=249):
    """Dégèle la fin du réseau InceptionV3 en laissant les couches BatchNormalization figées."""
    if not 0 <= unfreeze_from_layer < len(base_model.layers):
        raise ValueError(
            f"unfreeze_from_layer doit être compris entre 0 et {len(base_model.layers) - 1}."
        )

    base_model.trainable = True
    for index, layer in enumerate(base_model.layers):
        layer.trainable = (
            index >= unfreeze_from_layer and not isinstance(layer, BatchNormalization)
        )

    return model
