"""
Tests for text preprocessing and candidate generation.
"""

import unittest
from keyword_extractor.preprocessor import TextPreprocessor


class TestTextPreprocessor(unittest.TestCase):

    def setUp(self):
        self.prep = TextPreprocessor()

    def test_clean_text(self):
        raw = "Check this link https://example.com/nlp and contact user@domain.com   with   spaces."
        cleaned = self.prep.clean_text(raw)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("user@domain.com", cleaned)
        self.assertNotIn("   ", cleaned)

    def test_split_sentences(self):
        text = "Deep learning is powerful. NLP enables machines to read! Does it work well?"
        sentences = self.prep.split_sentences(text)
        self.assertEqual(len(sentences), 3)

    def test_tokenize_words(self):
        text = "State-of-the-art Natural Language Processing in 2026."
        tokens = self.prep.tokenize_words(text)
        self.assertIn("natural", tokens)
        self.assertIn("language", tokens)
        self.assertIn("processing", tokens)

    def test_generate_ngrams(self):
        tokens = ["deep", "neural", "network", "architectures"]
        ngrams = self.prep.generate_ngrams(tokens, ngram_range=(1, 2))
        self.assertIn("deep", ngrams)
        self.assertIn("deep neural", ngrams)
        self.assertIn("neural network", ngrams)

    def test_custom_stopwords(self):
        prep = TextPreprocessor(custom_stopwords=["customstop", "boilerplate"])
        self.assertIn("customstop", prep.stopwords)
        self.assertIn("boilerplate", prep.stopwords)


if __name__ == "__main__":
    unittest.main()
