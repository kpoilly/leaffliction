import os
import numpy as np

from keras.models import load_model
from PIL import Image


def predict(image_path, model_path="model/model.keras"):
    try:
        model = load_model(model_path)
        print(f"Model '{model_path}' loaded successfully.")
        print(model.summary())
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((128, 128))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

    class_names = ["Apple (Black Rot)",
                   "Apple (Healthy)",
                   "Apple (Rust)",
                   "Apple (Scab)",
                   "Grape (Black Rot)",
                   "Grape (Esca)",
                   "Grape (Healthy)",
                   "Grape (Spot)"]

    predictions = model.predict(img_array)
    prediction = np.argmax(predictions[0])

    return class_names[prediction]

