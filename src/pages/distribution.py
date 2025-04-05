from distribution import distribution, plot_distribution
import streamlit as st

st.set_page_config(page_title="Distribution", page_icon="📊")

st.markdown("# Distribution")
st.sidebar.header("Distribution")
st.write(
    """This Page is used to visualize the distribution of the data."""
)


def view():
    def display_plot():
        dist_map = distribution("data/")
        st.write("### Distribution of Images by directory")
        plot_distribution(
            dist_map,
            displayfunc=st.pyplot,
        )

    if st.button("Show Distribution"):
        with st.spinner("Generating distribution plot..."):
            display_plot()
        st.success("Plot generated!")


view()
