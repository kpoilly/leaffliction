import streamlit as st

st.set_page_config(page_title="Augmentation", page_icon="")

st.markdown("# Augmentation")
st.sidebar.header("Augmentation")
st.write(
    """This Page is used to visualize the augmentation process."""
)


def view():
    def display_augmented_img():
        st.write("### Augmented images")

    if st.button("Show Augmented images"):
        with st.spinner("Generating Augmented images..."):
            display_augmented_img()
        st.success("Images generated!")


view()
