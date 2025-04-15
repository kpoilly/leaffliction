import os
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import layers, models
from keras.api.callbacks import EarlyStopping
from keras.api.utils import image_dataset_from_directory


def load_split_dataset(path: str, batch_size=128):
    """
    Loads directory of images and split it into 2 tf.Datasets
    (train and validation)
    """
    try:
        df_train, df_val = image_dataset_from_directory(path,
                                                        image_size=(128, 128),
                                                        batch_size=batch_size,
                                                        validation_split=0.2,
                                                        subset='both',
                                                        shuffle=True,
                                                        seed=42
                                                        )
    except FileNotFoundError:
        raise AssertionError(f"file {path} not found.")
    print(
        f"Loaded df_train: {df_train.element_spec} - {len(df_train)} elements.")
    print(f"Loaded df_val: {df_val.element_spec} - {len(df_val)} elements.")
    return df_train, df_val

def draw_training(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plot_path = os.path.join("metrics", 'training_history.png')
    plt.savefig(plot_path)
    print(f"Training history plot saved to '{plot_path}'")

def create_model(nb_outputs, nb_filters=64, dropout=0.5):
    model = models.Sequential([
        layers.Rescaling(1.0 / 255),
        layers.BatchNormalization(),
        layers.SeparableConv2D(nb_filters, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.BatchNormalization(),
        layers.SeparableConv2D(nb_filters, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.BatchNormalization(),
        layers.SeparableConv2D(32, (1, 1), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(128, activation="relu"),
        layers.Dense(nb_outputs, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return model


def train(df, df_val, nb_filters=48, dropout=0.3, epochs=10, patience=3):
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

    history = model.fit(df,
              epochs=epochs,
              validation_data=df_val,
              callbacks=[early_stop])

    loss, accuracy = model.evaluate(df_val)
    print(f"Val Loss: {loss:.4f}")
    print(f"Val Accuracy: {accuracy:.4f}")

    draw_training(history)
    print(model.summary())

    return model
