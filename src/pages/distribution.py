from distribution import distribution
import streamlit as st

st.set_page_config(page_title="Distribution", page_icon="📊")

st.markdown("# Distribution")
st.sidebar.header("Distribution")
st.write(
    """This Page is used to visualize the distribution of the data."""
)


distribution('dsd')
