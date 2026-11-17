---
language:
- bn
license: mit
library_name: transformers
tags:
- text-classification
- toxicity-detection
- cyberbullying-detection
- hate-speech
- banglabert
- bengali
- electra
pipeline_tag: text-classification
widget:
- text: "আপনার ভিডিওটি অনেক তথ্যবহুল ও চমৎকার ছিল!"
- text: "তুই একটা আস্ত বদমাইশ ও বাটপাড়!"
- text: "এই মেয়েটাকে দেখলেই মেজাজ খারাপ হয়!"
---

# 🛡️ BanglaBERT Toxic & Harassment Comment Detector

Fine-tuned transformer model for detecting **toxicity, cyberbullying, and harassment in Bengali social media comments**, with particular focus on safeguarding online spaces and protecting women against digital harassment.

- **Base Architecture:** `csebuetnlp/banglabert` (ELECTRA)
- **Developer:** Shaiera Sultana Oishe
- **Task:** Binary Toxicity Classification (`0: Non-Toxic`, `1: Toxic / Hate Speech`)
- **Dataset:** 50,281 real Bengali social media comments

---

## 📊 Benchmark Results

Evaluated on unseen holdout test set (5,029 samples):

| Metric | Score |
| :--- | :---: |
| **Accuracy** | **90.91%** |
| **Macro F1** | **90.91%** |
| **Macro Precision** | **90.95%** |
| **Macro Recall** | **91.00%** |

---

## 🚀 How to Use

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

MODEL_NAME = "SS009/banglabert-toxic-detector"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

text = "আপনার কাজটি অনেক প্রশংসনীয়!"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)

labels = ["Non-Toxic", "Toxic / Harassment"]
predicted_idx = torch.argmax(probs, dim=-1).item()
print(f"Prediction: {labels[predicted_idx]} ({probs[0][predicted_idx].item()*100:.2f}%)")
```

---

## ⚖️ Intended Use & Limitations
- **Intended Use:** Assistive moderation for Bengali social media platforms, comment sections, and browser safety extensions.
- **Limitations:** Colloquial dialect words, satire, and sarcasm may occasionally produce false positives. Best used in conjunction with human moderation workflows.
