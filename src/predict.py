from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from huggingface_hub import login
from dotenv import load_dotenv

import torch
import torch.nn.functional as F
from loguru import logger

load_dotenv()


def load_model(model_path: str, tokenizer_path: str):
    login(token=os.getenv("HF_DETECTOR_TOKEN"))
    model_fine_tuned = AutoModelForSequenceClassification.from_pretrained(
        model_path, use_auth_token=True
    )
    tokenizer_fine_tuned = AutoTokenizer.from_pretrained(
        tokenizer_path, use_auth_token=True
    )
    return model_fine_tuned, tokenizer_fine_tuned


def detect_prompt_injection(
    model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, prompt: str
) -> dict[str, str | float]:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    predicted_index = torch.argmax(probs, dim=1).item()
    predicted_prob = probs[0][predicted_index].item()
    labels = model.config.id2label
    predicted_label = labels[predicted_index]
    return {"label": predicted_label, "probability": predicted_prob}


app = FastAPI()


@app.get("/")
def ping():
    return {"message": "pong"}


@app.get("/predict")
def predict(prompt: str) -> dict[str, str | float]:
    logger.info(f"Received prompt: {prompt}")
    model, tokenizer = load_model(
        model_path="lawincode/detect-prompt-injection",
        tokenizer_path="lawincode/detect-prompt-injection",
    )
    return detect_prompt_injection(model, tokenizer, prompt)
