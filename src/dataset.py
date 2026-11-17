"""
Dataset loading, tokenization, and stratified split utilities for Bengali toxicity data.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizerBase

from src.preprocessor import BanglaPreprocessor


class BanglaToxicityDataset(Dataset):
    """PyTorch Dataset for Bengali Toxic Comment Classification."""

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        preprocessor: Optional[BanglaPreprocessor] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessor = preprocessor or BanglaPreprocessor()

        # Preprocess text
        self.texts = [self.preprocessor.preprocess(t) for t in texts]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def load_presplit_data(
    train_path: str,
    val_path: str,
    test_path: str,
    text_col: str = "sentence",
    label_col: str = "hate speech",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-split train, val, and test CSV or TSV files directly."""
    train_df = pd.read_csv(train_path).dropna(subset=[text_col, label_col])
    val_df = pd.read_csv(val_path).dropna(subset=[text_col, label_col])
    test_df = pd.read_csv(test_path).dropna(subset=[text_col, label_col])

    train_df[label_col] = train_df[label_col].astype(int)
    val_df[label_col] = val_df[label_col].astype(int)
    test_df[label_col] = test_df[label_col].astype(int)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_and_split_data(
    file_path: str,
    text_col: str = "comment",
    label_col: str = "label",
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load dataset from CSV/TSV and generate stratified train, validation, and test splits.
    """
    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_csv(file_path, sep="\t")
    df = df.dropna(subset=[text_col, label_col])
    df[label_col] = df[label_col].astype(int)

    # First split off test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=random_state,
    )

    # Then split train and validation
    val_relative_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        stratify=train_val_df[label_col],
        random_state=random_state,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def calculate_class_weights(labels: List[int]) -> torch.Tensor:
    """
    Compute inverse frequency class weights to combat severe class imbalance.
    Returns: torch.Tensor of shape (num_classes,)
    """
    unique_classes, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    num_classes = len(unique_classes)

    weights = total_samples / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float)
