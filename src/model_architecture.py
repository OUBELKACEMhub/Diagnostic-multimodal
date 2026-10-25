

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, GlobalAveragePooling2D, BatchNormalization


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
