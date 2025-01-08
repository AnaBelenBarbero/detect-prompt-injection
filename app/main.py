import pandas as pd
import streamlit as st

from utils import load_css, predict

st.set_page_config(
    page_title="Detect Prompt Injection",
    page_icon="🔎",
    menu_items={
        "Get Help": "https://www.linkedin.com/in/ana-barbero-castejon/",
    },
)


css = load_css()
st.markdown(css, unsafe_allow_html=True)

prompt = st.text_input("Enter a prompt to detect prompt injection")

if prompt:
    result = predict(prompt)
    st.write(result)

