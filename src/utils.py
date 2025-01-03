import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def detect_prompt_injection(
    model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, prompt: str
) -> tuple[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    predicted_index = torch.argmax(probs, dim=1).item()
    predicted_prob = probs[0][predicted_index].item()
    labels = model.config.id2label
    predicted_label = labels[predicted_index]
    return predicted_label, predicted_prob
