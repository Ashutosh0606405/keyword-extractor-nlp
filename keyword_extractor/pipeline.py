"""
Unified KeywordExtractor Pipeline facade.
Provides easy switching between methods, multi-method comparison, batch processing,
and formatting utilities.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from keyword_extractor.base import BaseKeywordExtractor, KeywordResult
from keyword_extractor.algorithms.tfidf import TFIDFExtractor
from keyword_extractor.algorithms.rake import RAKEExtractor
from keyword_extractor.algorithms.textrank import TextRankExtractor
from keyword_extractor.algorithms.yake import YAKEExtractor
from keyword_extractor.algorithms.semantic import SemanticExtractor
from keyword_extractor.algorithms.ensemble import EnsembleExtractor
from keyword_extractor.preprocessor import TextPreprocessor


class KeywordExtractorPipeline:
    """
    High-level facade for Keyword Extraction NLP.

    Supported methods:
        - 'ensemble' (default): Multi-algorithm consensus using Reciprocal Rank Fusion
        - 'tfidf': TF-IDF weighting (single-doc pseudo-corpus or background corpus)
        - 'rake': Rapid Automatic Keyword Extraction
        - 'textrank': Co-occurrence graph PageRank
        - 'positionrank': Graph PageRank biased towards early occurrences
        - 'yake': Statistical feature scoring (casing, position, dispersion)
        - 'semantic': Embedding space cosine similarity + MMR diversity
    """

    AVAILABLE_METHODS = {
        "ensemble", "tfidf", "rake", "textrank", "positionrank", "yake", "semantic"
    }

    def __init__(
        self,
        method: str = "ensemble",
        custom_stopwords: Optional[List[str]] = None,
        diversity: float = 0.5,
        corpus: Optional[List[str]] = None
    ):
        self.method_name = method.lower()
        self.preprocessor = TextPreprocessor(custom_stopwords=custom_stopwords)
        self.diversity = diversity
        self.corpus = corpus

        # Cache of initialized extractors
        self._extractors: Dict[str, BaseKeywordExtractor] = {}
        self._init_extractor(self.method_name)

    def _init_extractor(self, method: str) -> BaseKeywordExtractor:
        """Instantiate and cache extractor by name."""
        method = method.lower()
        if method in self._extractors:
            return self._extractors[method]

        if method == "tfidf":
            ext = TFIDFExtractor(preprocessor=self.preprocessor, corpus=self.corpus)
        elif method == "rake":
            ext = RAKEExtractor(preprocessor=self.preprocessor)
        elif method == "textrank":
            ext = TextRankExtractor(preprocessor=self.preprocessor, use_position_bias=False)
        elif method == "positionrank":
            ext = TextRankExtractor(preprocessor=self.preprocessor, use_position_bias=True)
        elif method == "yake":
            ext = YAKEExtractor(preprocessor=self.preprocessor)
        elif method == "semantic":
            ext = SemanticExtractor(preprocessor=self.preprocessor, diversity=self.diversity)
        elif method == "ensemble":
            ext = EnsembleExtractor(preprocessor=self.preprocessor)
        else:
            raise ValueError(
                f"Unknown extraction method '{method}'. Available: {sorted(list(self.AVAILABLE_METHODS))}"
            )

        self._extractors[method] = ext
        return ext

    def extract(
        self,
        text: str,
        top_k: int = 10,
        ngram_range: Tuple[int, int] = (1, 2),
        method: Optional[str] = None,
        **kwargs
    ) -> List[KeywordResult]:
        """
        Extract ranked keywords from a text document.

        Args:
            text: Input text string.
            top_k: Number of keywords to return.
            ngram_range: (min_n, max_n) token length.
            method: Override the default extraction method.

        Returns:
            List of KeywordResult objects.
        """
        active_method = (method or self.method_name).lower()
        extractor = self._init_extractor(active_method)
        return extractor.extract_keywords(
            text, top_k=top_k, ngram_range=ngram_range, **kwargs
        )

    def extract_batch(
        self,
        texts: List[str],
        top_k: int = 10,
        ngram_range: Tuple[int, int] = (1, 2),
        method: Optional[str] = None,
        **kwargs
    ) -> List[List[KeywordResult]]:
        """
        Batch keyword extraction.
        """
        active_method = (method or self.method_name).lower()
        extractor = self._init_extractor(active_method)
        return extractor.extract_batch(
            texts, top_k=top_k, ngram_range=ngram_range, **kwargs
        )

    def compare(
        self,
        text: str,
        top_k: int = 5,
        ngram_range: Tuple[int, int] = (1, 2),
        methods: Optional[List[str]] = None
    ) -> Dict[str, List[KeywordResult]]:
        """
        Run multiple extraction algorithms on the same text side-by-side.

        Returns:
            Dictionary mapping method name to its list of KeywordResults.
        """
        target_methods = methods or ["ensemble", "tfidf", "rake", "positionrank", "yake", "semantic"]
        comparison: Dict[str, List[KeywordResult]] = {}

        for m in target_methods:
            try:
                ext = self._init_extractor(m)
                comparison[ext.name] = ext.extract_keywords(
                    text, top_k=top_k, ngram_range=ngram_range
                )
            except Exception as e:
                comparison[m] = []

        return comparison

    def to_dataframe(self, results: List[KeywordResult]):
        """Convert a list of KeywordResults to a Pandas DataFrame."""
        try:
            import pandas as pd
            records = [
                {
                    "Keyword": r.keyword,
                    "Score": r.score,
                    "N-Gram": r.ngram_size,
                    "Frequency": r.frequency,
                    "Method": r.metadata.get("method", self.method_name)
                }
                for r in results
            ]
            return pd.DataFrame(records)
        except ImportError:
            raise ImportError("pandas is required for to_dataframe(). Install pandas or use r.to_dict()")
