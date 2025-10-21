import streamlit as st
from streamlit_extras.switch_page_button import switch_page

# Trick: Use empty container, no widgets, no sidebar rendering
st.set_page_config(page_title="Redirecting...", layout="wide")

# Hide all UI (so this page never appears in the sidebar)
st.markdown("""
    <style>
    [data-testid="stSidebarNav"] {visibility: hidden;}
    .stApp {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Redirect immediately
switch_page("Home")


# Hide this default "streamlit app" page by redirecting to Home
st.switch_page("pages/0_Home.py")

