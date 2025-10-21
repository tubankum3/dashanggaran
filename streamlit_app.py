import streamlit as st

# Hide the "streamlit_app" link in the sidebar navigation
hide_main_page = """
    <style>
    [data-testid="stSidebarNav"] li a[href*="streamlit_app"] {
        display: none !important;
    }
    </style>
"""
st.markdown(hide_main_page, unsafe_allow_html=True)

# Hide this default "streamlit app" page by redirecting to Home
st.switch_page("pages/1_Home.py")

