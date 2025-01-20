import streamlit as st
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.predict import detect_prompt_injection, load_model


@st.cache_data
def load_css() -> str:
    return """<style>
    .st-dn {
        border-color: rgba(49, 51, 63, 0.2);
    }
    .streamlit-expanderHeader p{
        font-weight: bold;
    }
    .streamlit-expanderHeader {
        background: #a8eba495;
    }
    .streamlit-expanderContent {
        background: #c1e9be2e;
    }</style>
    """


@st.cache_data
def load_cache_model() -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    model, tokenizer = load_model(
        model_path="ana-contrasto-ai/detect-prompt-injection",
        tokenizer_path="ana-contrasto-ai/detect-prompt-injection",
    )
    return model, tokenizer


def predict(prompt: str) -> dict[str, str | float]:
    logger.info(f"Received prompt: {prompt}")
    model, tokenizer = load_cache_model()
    return detect_prompt_injection(model, tokenizer, prompt)
