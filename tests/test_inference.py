"""
Unit tests for toxic comment inference engine.
Compatible with both pytest and python -m unittest.
"""

import unittest
from src.inference import ToxicCommentDetector


class TestToxicCommentDetector(unittest.TestCase):
    def setUp(self):
        # Instantiate in lazy mode for quick unit testing
        self.detector = ToxicCommentDetector(lazy_load=True)

    def test_empty_string_prediction(self):
        res = self.detector.predict_single("")
        self.assertFalse(res["is_toxic"])
        self.assertEqual(res["label"], "Non-Toxic")
        self.assertEqual(res["toxicity_score"], 0.0)

    def test_non_toxic_comment(self):
        text = "আপনার এই ভিডিওটি দেখে অনেক ভালো লাগল, ধন্যবাদ!"
        res = self.detector.predict_single(text)
        self.assertIn("cleaned_text", res)
        self.assertIn("toxicity_score", res)
        self.assertIn("all_scores", res)
        self.assertFalse(res["is_toxic"])

    def test_toxic_comment_flagging(self):
        text = "তুই একটা আস্ত বদমাইশ, দূর হ এখান থেকে!"
        res = self.detector.predict_single(text)
        self.assertTrue(res["is_toxic"])
        self.assertIn("বদমাইশ", res["flagged_spans"])

    def test_batch_prediction_shape(self):
        texts = [
            "খুব সুন্দর পোস্ট!",
            "এই ফালতু লোকটাকে এখান থেকে বের করো।",
        ]
        results = self.detector.predict_batch(texts)
        self.assertEqual(len(results), 2)
        self.assertFalse(results[0]["is_toxic"])
        self.assertTrue(results[1]["is_toxic"])

    def test_threshold_sensitivity(self):
        self.detector.threshold = 0.95
        # Even if score is 0.85, high threshold of 0.95 classifies as non-toxic
        res = self.detector.predict_single("ফালতু একটা কাজ")
        self.assertFalse(res["is_toxic"])


if __name__ == "__main__":
    unittest.main()

