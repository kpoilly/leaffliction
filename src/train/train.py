import tensorflow as tf
from keras import layers, models
from keras.api.callbacks import EarlyStopping


def create_model(nb_outputs, nb_filters=64, dropout=0.5):
    model = models.Sequential([
        layers.Rescaling(1.0 / 255),
        layers.BatchNormalization(),
        layers.Conv2D(nb_filters, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.BatchNormalization(),
        layers.Conv2D(nb_filters, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.BatchNormalization(),
        layers.Conv2D(32, (1, 1), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(256, activation="relu"),
        layers.Dense(nb_outputs, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return model


def train(df, df_val, nb_filters=64, dropout=0.5, epochs=5, patience=2):
    print(f"Starting model's training with settings:\n{epochs} epochs\n\
Convolution filters: {nb_filters}\nDropout: {dropout}")

    model = create_model(len(df.class_names), nb_filters, dropout)
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )

    model.fit(df,
              epochs=epochs,
              validation_data=df_val,
              callbacks=[early_stop])

    loss, accuracy = model.evaluate(df_val)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    return model
