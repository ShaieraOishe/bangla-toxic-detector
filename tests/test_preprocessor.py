"""
Unit tests for Bengali text preprocessor.
Compatible with both pytest and python -m unittest.
"""

import unittest
from src.preprocessor import BanglaPreprocessor


class TestBanglaPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = BanglaPreprocessor()

    def test_empty_and_none_input(self):
        self.assertEqual(self.preprocessor.preprocess(""), "")
        self.assertEqual(self.preprocessor.preprocess(None), "")
        self.assertEqual(self.preprocessor.preprocess("   "), "")

    def test_url_removal(self):
        text = "এই ভিডিওটি দেখুন https://youtu.be/xyz123 এবং জানান"
        cleaned = self.preprocessor.preprocess(text)
        self.assertNotIn("https://youtu.be/xyz123", cleaned)
        self.assertIn("এই ভিডিওটি দেখুন", cleaned)

    def test_mention_removal(self):
        text = "সাবাশ @sakib_75 দারুণ খেলেছ!"
        cleaned = self.preprocessor.preprocess(text)
        self.assertNotIn("@sakib_75", cleaned)
        self.assertIn("দারুণ খেলেছ", cleaned)

    def test_hashtag_normalization(self):
        text = "সবাই ভালো থাকুন #bangladesh #peace"
        cleaned = self.preprocessor.preprocess(text)
        self.assertNotIn("#", cleaned)
        self.assertIn("bangladesh peace", cleaned)

    def test_zero_width_character_handling(self):
        text = "বাংলা\u200Cভাষা\u200D"
        cleaned = self.preprocessor.preprocess(text)
        self.assertNotIn("\u200C", cleaned)
        self.assertNotIn("\u200D", cleaned)

    def test_repeated_punctuation_normalization(self):
        text = "সত্যিই কি তাই???? দারুণ!!!!"
        cleaned = self.preprocessor.preprocess(text)
        self.assertNotIn("????", cleaned)
        self.assertEqual(cleaned, "সত্যিই কি তাই? দারুণ!")

    def test_batch_preprocess(self):
        texts = ["কমেন্ট ১ https://fb.com", "কমেন্ট ২ @user"]
        cleaned = self.preprocessor.batch_preprocess(texts)
        self.assertEqual(len(cleaned), 2)
        self.assertNotIn("https", cleaned[0])
        self.assertNotIn("@user", cleaned[1])


if __name__ == "__main__":
    unittest.main()

