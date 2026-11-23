"""
Unit tests for all keyword extraction algorithms.
"""

import unittest
from keyword_extractor.algorithms.tfidf import TFIDFExtractor
from keyword_extractor.algorithms.rake import RAKEExtractor
from keyword_extractor.algorithms.textrank import TextRankExtractor
from keyword_extractor.algorithms.yake import YAKEExtractor
from keyword_extractor.algorithms.semantic import SemanticExtractor
from keyword_extractor.algorithms.ensemble import EnsembleExtractor


SAMPLE_TEXT = """
Machine learning and artificial intelligence are revolutionizing modern computer science.
Deep learning uses artificial neural networks with multiple layers to model complex representations.
Supervised machine learning algorithms learn mapping functions from input data to output labels.
"""


class TestExtractors(unittest.TestCase):

    def test_tfidf_extractor(self):
        ext = TFIDFExtractor()
        results = ext.extract_keywords(SAMPLE_TEXT, top_k=5)
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 5)
        self.assertTrue(all(0.0 <= r.score <= 1.0 for r in results))

    def test_rake_extractor(self):
        ext = RAKEExtractor()
        results = ext.extract_keywords(SAMPLE_TEXT, top_k=5)
        self.assertGreater(len(results), 0)
        self.assertTrue(all(r.ngram_size >= 1 for r in results))

    def test_textrank_extractor(self):
        ext = TextRankExtractor(use_position_bias=False)
        results = ext.extract_keywords(SAMPLE_TEXT, top_k=5)
        self.assertGreater(len(results), 0)

    def test_positionrank_extractor(self):
        ext = TextRankExtractor(use_position_bias=True)
        results = ext.extract_keywords(SAMPLE_TEXT, top_k=5)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].metadata.get("method"), "PositionRank")

    def test_yake_extractor(self):
        ext = YAKEExtractor()
        results = ext.extract_keywords(SAMPLE_TEXT, top_k=5)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].score, 1.0)

    def test_semantic_extractor(self):
        ext = SemanticExtractor(diversity=0.4)
        results = ext.extract_keywords(SAMPLE_TEXT, top_k=5)
        self.assertGreater(len(results), 0)

    def test_ensemble_extractor(self):
        ext = EnsembleExtractor()
        results = ext.extract_keywords(SAMPLE_TEXT, top_k=5)
        self.assertGreater(len(results), 0)
        # Verify metadata has consensus votes
        self.assertIn("supporting_methods", results[0].metadata)
        self.assertGreater(results[0].metadata["vote_count"], 0)


if __name__ == "__main__":
    unittest.main()
