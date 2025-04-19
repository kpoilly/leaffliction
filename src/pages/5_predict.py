import os
import cv2
import streamlit as st
import numpy as np

from predict import predict
from transformation import transformation_from_img

st.set_page_config(page_title="Prediction", page_icon="")

st.markdown("# Prediction")
st.sidebar.header("Prediction")
st.write(
    """This Page is used to predict the class of a leaf."""
)

def view():
    imgFile = st.file_uploader(
        "image", type=["jpg", "jpeg", "JPG"])

    if imgFile:
        file_bytes = np.frombuffer(imgFile.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        images = transformation_from_img(img, st.pyplot)

        cols = st.columns(3)
        image_list = list(images.items())
        for i, (key, img) in enumerate(image_list):
            with cols[i % 3]:
                st.image(img, caption=key, use_container_width=True)

        if st.button("Predict"):
            with st.spinner("Predicting..."):
                model_path = "model/model.keras"
                if not os.path.exists(model_path):
                    st.error(f"Model file '{model_path}' not found. Make sure to train the model first!")
                else:
                    predicted_class = predict(imgFile, model_path)
                    st.success(f"Predicted class: {predicted_class}")


view()
