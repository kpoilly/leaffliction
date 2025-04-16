import os
import PIL
import streamlit as st

from predict import predict

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
        st.write("### Original image")
        st.image(imgFile, caption="Original Image", use_container_width=True)

        if st.button("Predict"):
            with st.spinner("Predicting..."):
                model_path = "model/model.keras"
                if not os.path.exists(model_path):
                    st.error(f"Model file '{model_path}' not found. Make sure to train the model first!")
                else:
                    predicted_class = predict(imgFile, model_path)
                    st.success("Prediction done!")
                    st.write(f"Predicted class: {predicted_class}")
                os.remove("tmp.jpg")


view()
