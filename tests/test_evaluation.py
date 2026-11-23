"""
Tests for evaluation metrics and benchmark datasets.
"""

import unittest
from keyword_extractor.base import KeywordResult
from keyword_extractor.evaluation import evaluate_predictions, run_benchmark
from keyword_extractor.pipeline import KeywordExtractorPipeline


class TestEvaluation(unittest.TestCase):

    def test_evaluate_predictions_exact(self):
        preds = [
            KeywordResult(keyword="transformer", score=1.0),
            KeywordResult(keyword="attention mechanism", score=0.9),
            KeywordResult(keyword="random word", score=0.5)
        ]
        golds = ["transformer", "attention mechanism", "neural networks"]

        scores = evaluate_predictions(preds, golds, top_k=3)
        self.assertAlmostEqual(scores.precision, 2 / 3, places=2)
        self.assertAlmostEqual(scores.recall, 2 / 3, places=2)
        self.assertEqual(scores.mrr, 1.0)

    def test_evaluate_predictions_soft(self):
        preds = [
            KeywordResult(keyword="transformer model", score=1.0),
            KeywordResult(keyword="attention", score=0.9)
        ]
        golds = ["transformer", "attention mechanism"]

        scores = evaluate_predictions(preds, golds, top_k=2)
        self.assertEqual(scores.soft_precision, 1.0)
        self.assertEqual(scores.soft_recall, 1.0)

    def test_run_benchmark(self):
        pipeline = KeywordExtractorPipeline()
        bench = run_benchmark(pipeline, methods=["ensemble", "tfidf"], top_k=5)
        self.assertIn("ensemble", bench)
        self.assertIn("tfidf", bench)
        self.assertIn("f1@k", bench["ensemble"])


if __name__ == "__main__":
    unittest.main()
