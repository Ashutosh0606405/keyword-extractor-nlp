"""
Tests for KeywordExtractorPipeline facade, batch processing, and comparisons.
"""

import unittest
from keyword_extractor.pipeline import KeywordExtractorPipeline


class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = KeywordExtractorPipeline()
        self.text = "Natural language processing enables intelligent computers to extract key phrases and concepts."

    def test_pipeline_default_ensemble(self):
        res = self.pipeline.extract(self.text, top_k=5)
        self.assertGreater(len(res), 0)
        self.assertLessEqual(len(res), 5)

    def test_pipeline_method_override(self):
        for method in ["tfidf", "rake", "textrank", "yake", "semantic"]:
            res = self.pipeline.extract(self.text, top_k=3, method=method)
            self.assertGreater(len(res), 0)

    def test_pipeline_compare(self):
        comp = self.pipeline.compare(self.text, top_k=3)
        self.assertIn("EnsembleHybrid", comp)
        self.assertIn("TF-IDF", comp)
        self.assertIn("RAKE", comp)

    def test_pipeline_batch(self):
        texts = [
            "Deep neural networks learn hierarchical representations from images.",
            "Quantum computers manipulate qubits using quantum superposition and entanglement."
        ]
        batch_res = self.pipeline.extract_batch(texts, top_k=3)
        self.assertEqual(len(batch_res), 2)
        self.assertGreater(len(batch_res[0]), 0)
        self.assertGreater(len(batch_res[1]), 0)

    def test_pipeline_to_dataframe(self):
        res = self.pipeline.extract(self.text, top_k=5)
        df = self.pipeline.to_dataframe(res)
        self.assertEqual(len(df), len(res))
        self.assertIn("Keyword", df.columns)
        self.assertIn("Score", df.columns)


if __name__ == "__main__":
    unittest.main()
