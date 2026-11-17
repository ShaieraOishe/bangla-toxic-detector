"""
Inference engine for Bangla toxic and harassment comment detection.
Supports confidence scores, thresholding, and keyword span highlighting.
"""

from typing import Any, Dict, List, Optional, Union

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    torch = None
    F = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    HAS_TORCH = False

from src.model import DEFAULT_ID2LABEL, DEFAULT_PRETRAINED_MODEL
from src.preprocessor import BanglaPreprocessor


# Common Bengali abusive/harassment keywords for span highlighting / rule-based fallback
KNOWN_TOXIC_KEYWORDS = {
    "শালা", "শালি", "মাগি", "বেশ্যা", "খানকি", "শুয়োরের", "কুত্তা",
    "হারামি", "বদমাইশ", "মর", "মারব", "ফালতু", "হতচ্ছাড়া", "পাজী"
}


class ToxicCommentDetector:
    """End-to-end detector for Bengali toxic and harassment comments."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_PRETRAINED_MODEL,
        device: Optional[str] = None,
        threshold: float = 0.5,
        lazy_load: bool = False,
    ):
        self.model_name_or_path = model_name_or_path
        self.threshold = threshold
        self.preprocessor = BanglaPreprocessor()

        if device is None:
            if HAS_TORCH and torch.cuda.is_available():
                self.device = "cuda"
            elif HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.tokenizer = None
        self.model = None

        if not lazy_load:
            self._load_model()

    def _load_model(self):
        """Load tokenizer and model weights onto target device."""
        if self.tokenizer is None or self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path
            )
            self.model.to(self.device)
            self.model.eval()

    def find_flagged_spans(self, text: str) -> List[str]:
        """Detect known abusive/harassment spans in text for explainability."""
        words = text.split()
        flagged = [w.strip("।,!?") for w in words if any(kw in w for kw in KNOWN_TOXIC_KEYWORDS)]
        return list(dict.fromkeys(flagged))  # preserve order & unique

    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Analyze a single comment for toxicity.

        Returns:
            Dictionary with cleaned text, prediction label, confidence,
            toxicity score, and flagged spans.
        """
        cleaned_text = self.preprocessor.preprocess(text)
        if not cleaned_text:
            return {
                "original_text": text,
                "cleaned_text": "",
                "is_toxic": False,
                "label": "Non-Toxic",
                "toxicity_score": 0.0,
                "confidence": 1.0,
                "all_scores": {"Non-Toxic": 1.0, "Toxic / Harassment": 0.0},
                "flagged_spans": [],
            }

        flagged_spans = self.find_flagged_spans(cleaned_text)

        # If model is loaded, run neural inference
        if self.model is not None and self.tokenizer is not None:
            inputs = self.tokenizer(
                cleaned_text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

            id2label = getattr(self.model.config, "id2label", DEFAULT_ID2LABEL)
            # Ensure keys are int
            id2label = {int(k): v for k, v in id2label.items()}

            # Toxicity score is prob of class 1 (or max toxic class)
            toxic_prob = probs[1] if len(probs) > 1 else probs[0]
            is_toxic = toxic_prob >= self.threshold

            label_name = id2label.get(1 if is_toxic else 0, "Toxic" if is_toxic else "Non-Toxic")

            all_scores = {
                id2label.get(i, f"Class {i}"): round(p, 4) for i, p in enumerate(probs)
            }

            return {
                "original_text": text,
                "cleaned_text": cleaned_text,
                "is_toxic": is_toxic,
                "label": label_name,
                "toxicity_score": round(toxic_prob, 4),
                "confidence": round(max(probs), 4),
                "all_scores": all_scores,
                "flagged_spans": flagged_spans,
            }

        # Fallback heuristic mode if weights aren't loaded yet
        has_flagged = len(flagged_spans) > 0
        score = 0.85 if has_flagged else 0.15
        is_toxic = score >= self.threshold
        return {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "is_toxic": is_toxic,
            "label": "Toxic / Harassment" if is_toxic else "Non-Toxic",
            "toxicity_score": score,
            "confidence": score if is_toxic else (1 - score),
            "all_scores": {
                "Non-Toxic": round(1 - score, 4),
                "Toxic / Harassment": round(score, 4),
            },
            "flagged_spans": flagged_spans,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Batch prediction over a list of texts."""
        return [self.predict_single(t) for t in texts]
