import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    TrainingArguments, 
    Trainer
)
from huggingface_hub import login
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import evaluate
from torch.utils.data import Dataset
from typing import Dict, Tuple
from pathlib import Path

class PromptDataset(Dataset):
    """Dataset class for prompt classification"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_datasets(english_path: str, 
                 spanish_jailbreak_path: str, 
                 spanish_benign_path: str) -> pd.DataFrame:
    """
    Load and combine all datasets
    
    Args:
        english_path: Path to English dataset
        spanish_jailbreak_path: Path to Spanish jailbreak dataset
        spanish_benign_path: Path to Spanish benign dataset
    
    Returns:
        Combined DataFrame
    """
    # Load English dataset
    dataset = pd.read_parquet(english_path)
    
    # Load Spanish datasets
    spanish_df_jailbreak = pd.read_csv(spanish_jailbreak_path)
    spanish_df_benign = pd.read_csv(spanish_benign_path)
    
    # Prepare Spanish datasets
    spanish_df_jailbreak.rename(columns={'texto': 'text', 'tipo': 'label'}, inplace=True)
    spanish_df_jailbreak['label'] = 1
    spanish_df_benign.rename(columns={'texto': 'text', 'tipo': 'label'}, inplace=True)
    spanish_df_benign['label'] = 0
    
    # Combine datasets
    return pd.concat([dataset, spanish_df_jailbreak, spanish_df_benign], ignore_index=True)

def prepare_datasets(df: pd.DataFrame, 
                    tokenizer: AutoTokenizer, 
                    train_size: float = 0.8) -> Tuple[Dict, Dict, pd.DataFrame, pd.DataFrame]:
    """
    Prepare and split datasets
    
    Args:
        df: Input DataFrame
        tokenizer: HuggingFace tokenizer
        train_size: Fraction of data for training
    
    Returns:
        Tuple of (train_encodings, test_encodings, train_df, test_df)
    """
    # Rename columns for consistency
    df.rename(columns={'label': 'type', 'text': 'prompt'}, inplace=True)
    
    # Split data
    df_train = df.sample(frac=train_size, random_state=42)
    df_test = df.drop(df_train.index)
    
    # Tokenize
    train_encodings = tokenizer(df_train["prompt"].tolist(), 
                              padding="max_length", 
                              truncation=True, 
                              return_tensors="pt")
    test_encodings = tokenizer(df_test["prompt"].tolist(), 
                             padding="max_length", 
                             truncation=True, 
                             return_tensors="pt")
    
    # Convert to numpy for dataset creation
    train_encodings = {key: val.numpy() for key, val in train_encodings.items()}
    test_encodings = {key: val.numpy() for key, val in test_encodings.items()}
    
    return train_encodings, test_encodings, df_train, df_test

def compute_metrics(eval_pred) -> Dict:
    """
    Compute metrics for model evaluation
    
    Args:
        eval_pred: Tuple of predictions and labels
    
    Returns:
        Dictionary of metric scores
    """
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def setup_model_and_tokenizer(model_name: str) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Setup the model and tokenizer
    
    Args:
        model_name: Name or path of the model
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    return model, tokenizer

def create_trainer(model: AutoModelForSequenceClassification,
                  train_dataset: PromptDataset,
                  eval_dataset: PromptDataset,
                  training_args: Dict | None = None) -> Trainer:
    """
    Create and return a trainer instance
    
    Args:
        model: The model to train
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        training_args: Optional training arguments
    
    Returns:
        Trainer instance
    """
    if training_args is None:
        training_args = TrainingArguments(
            output_dir="test_trainer",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
        )
    
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

def main():
    """Main training pipeline"""
    print("Starting training pipeline...")
    
    # Load environment variables and login
    print("Loading environment variables...")
    load_dotenv()
    login(token=os.getenv('HF_TOKEN'))
    print("Successfully logged in to Hugging Face")
    
    # Setup model and tokenizer
    print("Setting up model and tokenizer...")
    model_name = "madhurjindal/Jailbreak-Detector"
    model, tokenizer = setup_model_and_tokenizer(model_name)
    print(f"Model and tokenizer loaded from {model_name}")
    
    # Load datasets
    print("Loading datasets...")
    df = load_datasets(
        english_path='hf://datasets/deepset/prompt-injections/data/train-00000-of-00001-9564e8b05b4757ab.parquet',
        spanish_jailbreak_path="data/jailbreak_detection_dataset.csv",
        spanish_benign_path="data/benign_prompts_dataset.csv"
    )
    print(f"Loaded {len(df)} total examples")
    
    # Prepare datasets
    print("Preparing and splitting datasets...")
    train_encodings, test_encodings, df_train, df_test = prepare_datasets(df, tokenizer)
    print(f"Split into {len(df_train)} training and {len(df_test)} test examples")
    
    # Create dataset objects
    print("Creating dataset objects...")
    train_dataset = PromptDataset(train_encodings, df_train["type"].tolist())
    test_dataset = PromptDataset(test_encodings, df_test["type"].tolist())
    print("Dataset objects created successfully")
    
    # Create and start trainer
    print("Initializing trainer...")
    trainer = create_trainer(model, train_dataset, test_dataset)
    
    # Train the model
    print("Starting training...")
    trainer.train()
    print("Training completed!")
    
    # Save the final model
    print("Saving model...")
    trainer.save_model("final_model")
    print("Training completed and model saved to 'final_model' directory!")

if __name__ == "__main__":
    main()