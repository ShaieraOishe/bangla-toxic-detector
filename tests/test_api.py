"""
Integration tests for FastAPI endpoints.
Compatible with both pytest and python -m unittest.
"""

import unittest

try:
    from fastapi.testclient import TestClient
    from app.api import app
    HAS_FASTAPI = True
except ImportError:
    TestClient = None
    app = None
    HAS_FASTAPI = False


class TestFastAPIEndpoints(unittest.TestCase):
    def setUp(self):
        if not HAS_FASTAPI:
            self.skipTest("FastAPI or dependencies not installed in test environment")
        self.client = TestClient(app)

    def test_health_check_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")

    def test_predict_endpoint_valid(self):
        payload = {"text": "আজকের দিনটি সত্যিই অনেক সুন্দর।", "threshold": 0.5}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("is_toxic", data)
        self.assertIn("toxicity_score", data)
        self.assertFalse(data["is_toxic"])

    def test_predict_endpoint_toxic(self):
        payload = {"text": "তুই একটা বদমাইশ!", "threshold": 0.5}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["is_toxic"])
        self.assertIn("বদমাইশ", data["flagged_spans"])

    def test_predict_endpoint_empty_fails(self):
        payload = {"text": "   ", "threshold": 0.5}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 400)

    def test_batch_predict_endpoint(self):
        payload = {
            "texts": ["চমৎকার উপস্থাপনা!", "ফালতু কন্টেন্ট!"],
            "threshold": 0.5,
        }
        response = self.client.post("/predict_batch", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 2)


if __name__ == "__main__":
    unittest.main()

