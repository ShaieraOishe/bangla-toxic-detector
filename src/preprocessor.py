"""
Bangla Text Preprocessing and Normalization Module.

Handles Unicode canonicalization, zero-width characters (ZWJ/ZWNJ),
URL removal, mention cleaning, and whitespace normalization tailored
for Bengali NLP pipelines.
"""

import re
import unicodedata
from typing import List, Optional


class BanglaPreprocessor:
    """Preprocessor for raw Bengali text before feeding into transformer tokenizers."""

    def __init__(self, keep_emojis: bool = True):
        self.keep_emojis = keep_emojis

        # Bengali Unicode range: \u0980-\u09FF
        # Common Bengali punctuation: । (Dari: \u0964), ॥ (\u0965)
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.mention_pattern = re.compile(r"@[\w\u0980-\u09FF_]+")
        self.hashtag_pattern = re.compile(r"#([\w\u0980-\u09FF_]+)")
        self.repeated_punct_pattern = re.compile(r"([!?,.।]){2,}")
        self.whitespace_pattern = re.compile(r"\s+")

        # Zero-width characters often corrupt tokenization if misplaced
        self.zwnj = "\u200C"
        self.zwj = "\u200D"

    def normalize_unicode(self, text: str) -> str:
        """Apply Unicode NFC normalization to compose decomposed characters."""
        return unicodedata.normalize("NFC", text)

    def clean_zero_width_chars(self, text: str) -> str:
        """
        Clean unneeded zero-width characters while preserving valid Bengali conjuncts.
        Removes repeated or isolated ZWJ/ZWNJ characters.
        """
        # Replace multiple consecutive zero-width chars with a single one or clean dangling ones
        text = re.sub(rf"[{self.zwnj}{self.zwj}]+", "", text)
        return text

    def remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return self.url_pattern.sub("", text)

    def remove_mentions(self, text: str) -> str:
        """Remove @mentions (common in social media comments)."""
        return self.mention_pattern.sub("", text)

    def normalize_hashtags(self, text: str) -> str:
        """Convert #hashtag to hashtag."""
        return self.hashtag_pattern.sub(r"\1", text)

    def normalize_punctuation(self, text: str) -> str:
        """Compress repeated punctuation like '????' or '!!!!' to a single mark."""
        return self.repeated_punct_pattern.sub(r"\1", text)

    def normalize_whitespace(self, text: str) -> str:
        """Collapse multiple spaces, tabs, and newlines into a single space."""
        return self.whitespace_pattern.sub(" ", text).strip()

    def preprocess(self, text: Optional[str]) -> str:
        """
        Full preprocessing pipeline for a single Bengali text string.

        Args:
            text: Input raw text string.

        Returns:
            Cleaned and normalized Bengali text.
        """
        if not text or not isinstance(text, str):
            return ""

        # Step 1: Unicode Normalization
        text = self.normalize_unicode(text)

        # Step 2: Clean URLs and Mentions
        text = self.remove_urls(text)
        text = self.remove_mentions(text)
        text = self.normalize_hashtags(text)

        # Step 3: Zero-width joiner handling
        text = self.clean_zero_width_chars(text)

        # Step 4: Punctuation and Whitespace
        text = self.normalize_punctuation(text)
        text = self.normalize_whitespace(text)

        return text

    def batch_preprocess(self, texts: List[str]) -> List[str]:
        """Preprocess a list of text strings."""
        return [self.preprocess(t) for t in texts]


if __name__ == "__main__":
    preprocessor = BanglaPreprocessor()
    sample = "এই মেয়েটা একদম ফালতু!!! @user123 https://fb.com/test #bangla"
    cleaned = preprocessor.preprocess(sample)
    print(f"Original: {sample}")
    print(f"Cleaned : {cleaned}")
