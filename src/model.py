"""
BanglaBERT Model Architecture and Configuration.

Wraps Hugging Face transformers AutoModelForSequenceClassification
with preconfigured settings for Bengali pretrained checkpoints.
"""

from typing import Dict, List, Optional

try:
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    AutoConfig = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    HAS_TRANSFORMERS = False

# Fine-tuned BanglaBERT model published on Hugging Face Hub
DEFAULT_PRETRAINED_MODEL = "SS009/banglabert-toxic-detector"

# Standard label mapping for Toxic and Harassment detection
DEFAULT_ID2LABEL: Dict[int, str] = {
    0: "Non-Toxic",
    1: "Toxic / Harassment",
}
DEFAULT_LABEL2ID: Dict[str, int] = {v: k for k, v in DEFAULT_ID2LABEL.items()}

# Detailed multi-category label mapping for fine-grained classification
FINE_GRAINED_ID2LABEL: Dict[int, str] = {
    0: "Safe / Neutral",
    1: "Abusive / Insult",
    2: "Harassment / Cyberbullying",
    3: "Sexual Harassment / Obscene",
    4: "Threat / Violence",
}
FINE_GRAINED_LABEL2ID: Dict[str, int] = {v: k for k, v in FINE_GRAINED_ID2LABEL.items()}


def get_tokenizer(model_name_or_path: str = DEFAULT_PRETRAINED_MODEL) -> AutoTokenizer:
    """Load tokenizer with fallback handling."""
    return AutoTokenizer.from_pretrained(model_name_or_path)


def build_model(
    model_name_or_path: str = DEFAULT_PRETRAINED_MODEL,
    num_labels: int = 2,
    id2label: Optional[Dict[int, str]] = None,
    label2id: Optional[Dict[str, int]] = None,
    dropout_rate: float = 0.2,
) -> AutoModelForSequenceClassification:
    """
    Constructs a Sequence Classification model initialized from a pretrained checkpoint.

    Args:
        model_name_or_path: HuggingFace model repo ID or local checkpoint path.
        num_labels: Number of output classes (default: 2 for binary toxic detection).
        id2label: Dictionary mapping class index to label name.
        label2id: Dictionary mapping label name to class index.
        dropout_rate: Dropout probability for classification head.

    Returns:
        Configured AutoModelForSequenceClassification instance.
    """
    if id2label is None:
        id2label = DEFAULT_ID2LABEL if num_labels == 2 else FINE_GRAINED_ID2LABEL
    if label2id is None:
        label2id = {v: k for k, v in id2label.items()}

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        classifier_dropout=dropout_rate,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
    )
    return model
