import streamlit as st
from augmentation import augmentation_from_img
import PIL


st.set_page_config(page_title="Augmentation", page_icon="")

st.markdown("# Augmentation")
st.sidebar.header("Augmentation")
st.write(
    """This Page is used to visualize the augmentation process."""
)


if "imgUploaded" not in st.session_state:
    st.session_state["imgUploaded"] = False


def onChangeCB():
    st.session_state["imgUploaded"] = True


def onClickCB():
    st.session_state["imgAugmented"] = True


def view():
    imgFile = st.file_uploader(
        "image", type=["jpg", "jpeg", "JPG"], on_change=onChangeCB)

    if st.session_state["imgUploaded"]:
        st.write("### Original image")
        st.image(imgFile, caption="Original Image", use_container_width=True)

        if st.button("Show Augmented images"):
            with st.spinner("Generating Augmented images..."):
                img = PIL.Image.open(imgFile)
                images = augmentation_from_img(
                    img, imgFile.name)
                st.success("Images generated!")
                for img in images:
                    st.write(img[0])
                    st.image(img[1], caption=img[0],
                             use_container_width=True)


view()
