"""
KeySense NLP - Production-grade Keyword & Keyphrase Extractor.
"""

from keyword_extractor.base import BaseKeywordExtractor, KeywordResult
from keyword_extractor.pipeline import KeywordExtractorPipeline
from keyword_extractor.preprocessor import TextPreprocessor
from keyword_extractor.algorithms.tfidf import TFIDFExtractor
from keyword_extractor.algorithms.rake import RAKEExtractor
from keyword_extractor.algorithms.textrank import TextRankExtractor
from keyword_extractor.algorithms.yake import YAKEExtractor
from keyword_extractor.algorithms.semantic import SemanticExtractor
from keyword_extractor.algorithms.ensemble import EnsembleExtractor
from keyword_extractor.evaluation import evaluate_predictions, run_benchmark

__version__ = "1.0.0"

__all__ = [
    "KeywordResult",
    "BaseKeywordExtractor",
    "KeywordExtractorPipeline",
    "TextPreprocessor",
    "TFIDFExtractor",
    "RAKEExtractor",
    "TextRankExtractor",
    "YAKEExtractor",
    "SemanticExtractor",
    "EnsembleExtractor",
    "evaluate_predictions",
    "run_benchmark",
]
