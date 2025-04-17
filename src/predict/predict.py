import os
from typing import List
import numpy as np
import cv2
from keras.models import load_model
from keras import Model
from transformation import transformation


def get_image(img_path: str):
    transformations = transformation(
        img_path, "", None, {"original", "mask"})
    original = transformations["original"]
    mask = transformations["mask"]
    img_array = np.array(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    img_array = np.expand_dims(img_array, axis=0)
    mask_array = np.array(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    mask_array = np.expand_dims(mask_array, axis=0)
    return (img_array, mask_array)


def predict_images(img_tuple: any, models: List[Model]):
    class_names = ["Apple_Black_rot",
                   "Apple_healthy",
                   "Apple_rust",
                   "Apple_scab",
                   "Grape_Black_rot",
                   "Grape_Esca",
                   "Grape_healthy",
                   "Grape_spot"]

    images, filename = img_tuple
    print(f"Processing {filename}...")
    predictions = [model.predict(image)
                   for model, image in zip(models, images)]
    print(predictions)
    predictions = np.sum(predictions, axis=0)
    print(predictions)
    prediction = np.argmax(predictions[0])
    print(
        f"{filename} | {class_names[prediction]}")
    predicted_class = class_names[prediction].lower()
    isValid = predicted_class in filename.lower()
    print(
        f"Validation: {'✓' if isValid else '✗'}"
    )

    return isValid


def predict(path: str):
    models_path = ['model/model_original.keras', 'model/model_mask.keras']
    try:
        models: List[Model] = [load_model(model_path)
                               for model_path in models_path]
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file '{path}' not found.")

        predictions = []
        if os.path.isdir(path):
            for filename in os.listdir(path):
                if filename.endswith(('.JPG', '.jpeg', '.png')):
                    images = get_image(os.path.join(path, filename))
                    predictions.append(predict_images(
                        (images, filename), models))
        elif os.path.isfile(path):
            images = get_image(path)
            predictions.append(predict_images(
                (images, os.path.basename(path)), models))
        else:
            raise ValueError("Invalid path. Must be a file or directory.")
        return predictions
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
