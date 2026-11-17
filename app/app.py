"""
Hugging Face Space Gradio Application for Bangla Toxic & Harassment Comment Detector.
"""

import gradio as gr
from src.inference import ToxicCommentDetector

# Instantiate detector
detector = ToxicCommentDetector(lazy_load=True)


def analyze_comment(text: str, threshold: float):
    """Callback function for Gradio interface."""
    if not text or not text.strip():
        return (
            "⚠️ অনুগ্রহ করে একটি মন্তব্য লিখুন (Please enter a comment).",
            {"Non-Toxic": 1.0, "Toxic": 0.0},
            "None",
            "",
        )

    detector.threshold = threshold
    res = detector.predict_single(text)

    # Status badge
    if res["is_toxic"]:
        status = f"🚨 **FLAGGED: {res['label']}** (স্কোর: {res['toxicity_score']*100:.1f}%)"
    else:
        status = f"✅ **SAFE: {res['label']}** (টক্সিক স্কোর: {res['toxicity_score']*100:.1f}%)"

    scores = res["all_scores"]
    flagged = ", ".join(res["flagged_spans"]) if res["flagged_spans"] else "কোনো আপত্তিকর শব্দ পাওয়া যায়নি (None)"
    cleaned = res["cleaned_text"]

    return status, scores, flagged, cleaned


# Curated Bengali sample comments representing different toxicity levels
EXAMPLES = [
    ["আপনার উপস্থাপনা এবং কাজ খুবই প্রশংসনীয়! শুভকামনা রইল।", 0.5],
    ["আমি আপনার মতামতের সাথে একমত নই, তবে যুক্তিটা ভালো ছিল।", 0.5],
    ["এই ফালতু মেয়েটাকে দেখলেই মেজাজ খারাপ হয়! একে সোশ্যাল মিডিয়া থেকে ব্যান করা উচিত।", 0.5],
    ["তুই একটা আস্ত বদমাইশ ও বাটপাড়, তোকে রাস্তায় পেলে মারব!", 0.5],
    ["চমৎকার একটি পোস্ট করেছেন! অনেক কিছু জানতে পারলাম।", 0.5],
]

demo = gr.Interface(
    fn=analyze_comment,
    inputs=[
        gr.Textbox(
            label="বাংলা মন্তব্য লিখুন বা পেস্ট করুন (Enter Bengali Comment)",
            placeholder="উদাহরণ: এই কাজটি অনেক সুন্দর হয়েছে...",
            lines=3,
        ),
        gr.Slider(
            minimum=0.1,
            maximum=0.9,
            value=0.5,
            step=0.05,
            label="Toxicity Detection Sensitivity Threshold (সংবেদনশীলতার মাত্রা)",
        ),
    ],
    outputs=[
        gr.Markdown(label="Detection Result"),
        gr.Label(label="Prediction Probabilities"),
        gr.Textbox(label="Flagged Abusive Spans / Keywords (চিহ্নিত আপত্তিকর শব্দসমূহ)"),
        gr.Textbox(label="Normalized Bengali Text (পরিশোধিত টেক্সট)"),
    ],
    title="🛡️ Bangla Toxic and Harassment Comment Detector",
    description=(
        "**BanglaBERT-powered moderation engine** designed to safeguard online spaces and protect women against "
        "cyberbullying, harassment, and abusive comments in Bengali. Fine-tuned on authentic social media datasets."
    ),
    examples=EXAMPLES,
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
