import tensorflow as tf
from keras.models import load_model
import numpy as np
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

    class_names = ["Apple_Black_rot",
                "Apple_healthy",
                "Apple_rust",
                "Apple_scab",
                "Grape_Black_rot",
                "Grape_Esca",
                "Grape_healthy",
                "Grape_spot"]

    predictions = model.predict(img_array)
    prediction = np.argmax(predictions[0])

    return class_names[prediction]
