from typing import Dict, Tuple

import evaluate
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)
from transformers.tokenization_utils_base import BatchEncoding

load_dotenv()


class PromptDataset(Dataset):
    """Dataset class for prompt classification"""

    def __init__(self, encodings, labels):
        self.encodings: BatchEncoding | dict = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_datasets() -> pd.DataFrame:
    """
    Load and combine all datasets

    Args:
        english_path: Path to English dataset
        spanish_jailbreak_path: Path to Spanish jailbreak dataset
        spanish_benign_path: Path to Spanish benign dataset

    Returns:
        Combined DataFrame
    """

    # english
    df_3k_neg = pd.read_csv("../data/negative_3k_en_prompts.csv")
    df_3k_ben = pd.read_csv("../data/benign_2k_en_prompts.csv")

    english_dataset = pd.concat([df_3k_neg, df_3k_ben], ignore_index=True)
    english_dataset.dropna(inplace=True)

    # spanish
    df_3k_neg_esp = pd.read_csv("../data/negative_3k_es_prompts.csv")
    df_3k_ben_esp = pd.read_csv("../data/benign_2k_es_prompts.csv")
    df_3k_neg_esp.rename(columns={"tipo": "type"}, inplace=True)
    df_3k_ben_esp.rename(columns={"tipo": "type"}, inplace=True)

    spanish_dataset = pd.concat([df_3k_ben_esp, df_3k_neg_esp], ignore_index=True)
    spanish_dataset.dropna(inplace=True)

    # Combine datasets
    return pd.concat([english_dataset, spanish_dataset], ignore_index=True)


def prepare_datasets(
    df: pd.DataFrame, tokenizer: AutoTokenizer, train_size: float = 0.8
) -> Tuple[Dict, Dict, pd.DataFrame, pd.DataFrame]:
    """
    Prepare and split datasets

    Args:
        df: Input DataFrame
        tokenizer: HuggingFace tokenizer
        train_size: Fraction of data for training

    Returns:
        Tuple of (train_encodings, test_encodings, train_df, test_df)
    """
    df_train = df.sample(frac=train_size, random_state=42)
    df_test = df.drop(df_train.index)

    # tokenize
    train_encodings: BatchEncoding | dict = tokenizer(
        df_train["prompt"].tolist(),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    test_encodings: BatchEncoding | dict = tokenizer(
        df_test["prompt"].tolist(),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # convert to numpy for dataset creation
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


def setup_model_and_tokenizer(
    model_name: str,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Setup the model and tokenizer

    Args:
        model_name: Name or path of the model

    Returns:
        Tuple of (model, tokenizer)
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, use_auth_token=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    return model, tokenizer


def create_trainer(
    model: AutoModelForSequenceClassification,
    train_dataset: PromptDataset,
    eval_dataset: PromptDataset,
    training_args: Dict | None = None,
) -> Trainer:
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
    print("Loading environment variables...")

    # base model
    print("Setting up model and tokenizer...")
    model_name = "madhurjindal/Jailbreak-Detector"
    model, tokenizer = setup_model_and_tokenizer(model_name)
    print(f"Model and tokenizer loaded from {model_name}")

    # load fine tuning sets
    print("Loading datasets...")
    df = load_datasets()
    print(f"Loaded {len(df)} total examples")

    print("Preparing and splitting dataset encodings...")
    train_encodings, test_encodings, df_train, df_test = prepare_datasets(df, tokenizer)
    print(f"Split into {len(df_train)} training and {len(df_test)} test examples")

    print("Creating dataset objects...")
    train_dataset = PromptDataset(train_encodings, df_train["type"].tolist())
    test_dataset = PromptDataset(test_encodings, df_test["type"].tolist())
    print("Dataset objects created successfully")

    # create trainer and train
    print("Initializing trainer...")
    trainer = create_trainer(model, train_dataset, test_dataset)

    print("Starting training...")
    trainer.train()
    print("Training completed!")

    print("Saving model...")
    trainer.save_model("final_model")
    print("Training completed and model saved to 'final_model' directory!")


if __name__ == "__main__":
    main()
