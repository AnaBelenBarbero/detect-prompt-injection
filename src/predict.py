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
    model_fine_tuned = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer_fine_tuned = AutoTokenizer.from_pretrained(tokenizer_path)
    return model_fine_tuned, tokenizer_fine_tuned


def detect_prompt_injection(model, tokenizer, prompt):
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    # Move each tensor to the model's device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    predicted_index = torch.argmax(probs, dim=1).item()
    predicted_prob = probs[0][predicted_index].item()
    labels = model.config.id2label
    predicted_label = labels[predicted_index]
    return predicted_label, predicted_prob



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
