from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import torch
import torch.nn.functional as F


def load_model(model_path: str, tokenizer_path: str):
    model_fine_tuned = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer_fine_tuned = AutoTokenizer.from_pretrained(tokenizer_path)
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
    model, tokenizer = load_model(
        model_path="prompt_injection_detector_fine_tuned",
        tokenizer_path="prompt_injection_detector_fine_tuned",
    )
    return detect_prompt_injection(model, tokenizer, prompt)
