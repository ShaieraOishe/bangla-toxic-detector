"""
Training pipeline for fine-tuning BanglaBERT on toxic comment datasets.
"""

import argparse
import os
from typing import Dict
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.dataset import (
    BanglaToxicityDataset,
    calculate_class_weights,
    load_and_split_data,
    load_presplit_data,
)
from src.model import DEFAULT_PRETRAINED_MODEL, build_model, get_tokenizer
from src.preprocessor import BanglaPreprocessor


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Calculate accuracy, precision, recall, and F1 scores."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "macro_precision": precision,
        "macro_recall": recall,
        "weighted_f1": weighted_f1,
    }


class WeightedLossTrainer(Trainer):
    """Custom Trainer with weighted cross-entropy loss to handle class imbalance."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BanglaBERT for toxic comment detection.")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing train.csv, val.csv, and test.csv")
    parser.add_argument("--data_path", type=str, default=None, help="Path to single CSV dataset to auto-split")
    parser.add_argument("--text_col", type=str, default="sentence", help="Name of comment text column (e.g., sentence or comment)")
    parser.add_argument("--label_col", type=str, default="hate speech", help="Name of label column (e.g., hate speech or label)")
    parser.add_argument("--model_name", type=str, default=DEFAULT_PRETRAINED_MODEL, help="Pretrained model identifier")
    parser.add_argument("--output_dir", type=str, default="./results/banglabert_toxic", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--use_weighted_loss", action="store_true", help="Apply class-weighted loss")

    args = parser.parse_args()

    if args.data_dir:
        print(f"Loading pre-split data from directory: {args.data_dir}...")
        train_df, val_df, test_df = load_presplit_data(
            train_path=os.path.join(args.data_dir, "train.csv"),
            val_path=os.path.join(args.data_dir, "val.csv"),
            test_path=os.path.join(args.data_dir, "test.csv"),
            text_col=args.text_col,
            label_col=args.label_col,
        )
    elif args.data_path:
        print(f"Loading data from {args.data_path}...")
        train_df, val_df, test_df = load_and_split_data(
            file_path=args.data_path,
            text_col=args.text_col,
            label_col=args.label_col,
        )
    else:
        raise ValueError("Must provide either --data_dir or --data_path.")
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    num_labels = train_df[args.label_col].nunique()
    print(f"Number of classes detected: {num_labels}")

    preprocessor = BanglaPreprocessor()
    tokenizer = get_tokenizer(args.model_name)
    model = build_model(model_name_or_path=args.model_name, num_labels=num_labels)

    train_dataset = BanglaToxicityDataset(
        texts=train_df[args.text_col].tolist(),
        labels=train_df[args.label_col].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
        preprocessor=preprocessor,
    )

    val_dataset = BanglaToxicityDataset(
        texts=val_df[args.text_col].tolist(),
        labels=val_df[args.label_col].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
        preprocessor=preprocessor,
    )

    test_dataset = BanglaToxicityDataset(
        texts=test_df[args.text_col].tolist(),
        labels=test_df[args.label_col].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
        preprocessor=preprocessor,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=300,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
    )

    class_weights = None
    if args.use_weighted_loss:
        class_weights = calculate_class_weights(train_df[args.label_col].tolist())
        print(f"Computed Class Weights: {class_weights}")
        trainer = WeightedLossTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )

    print("Starting training...")
    trainer.train()

    print("\nEvaluating on holdout test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Results: {test_results}")

    # Save best model artifacts
    final_model_dir = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"Model and tokenizer saved to {final_model_dir}")


if __name__ == "__main__":
    main()
