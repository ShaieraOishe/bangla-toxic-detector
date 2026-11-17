"""
FastAPI REST API for Bangla Toxic and Harassment Comment Detector.
Enables backend integration for Chrome extensions, bots, and moderation dashboards.
"""

from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference import ToxicCommentDetector

app = FastAPI(
    title="Bangla Toxic Comment Detector API",
    description="Production-ready REST API for detecting toxic and abusive Bengali comments.",
    version="1.0.0",
)

# Enable CORS for browser extensions and external frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector (lazy loaded on start or fallback)
detector = ToxicCommentDetector(lazy_load=True)


class CommentRequest(BaseModel):
    text: str = Field(..., example="আপনার ভিডিওটি খুব সুন্দর হয়েছে!")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)


class BatchCommentRequest(BaseModel):
    texts: List[str] = Field(..., example=["ভালো কাজ!", "এই ফালতু লোকটাকে ব্যান করা হোক"])
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)


class CommentResponse(BaseModel):
    original_text: str
    cleaned_text: str
    is_toxic: bool
    label: str
    toxicity_score: float
    confidence: float
    all_scores: Dict[str, float]
    flagged_spans: List[str]


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "Bangla Toxic Comment Detector"}


@app.post("/predict", response_model=CommentResponse)
def predict_comment(request: CommentRequest):
    """Analyze a single comment for toxicity."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Comment text cannot be empty.")

    detector.threshold = request.threshold or 0.5
    result = detector.predict_single(request.text)
    return result


@app.post("/predict_batch", response_model=List[CommentResponse])
def predict_batch(request: BatchCommentRequest):
    """Analyze multiple comments in a single batch request."""
    if not request.texts:
        raise HTTPException(status_code=400, detail="Text list cannot be empty.")

    detector.threshold = request.threshold or 0.5
    results = detector.predict_batch(request.texts)
    return results
