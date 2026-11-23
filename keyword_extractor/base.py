"""
Base classes and data models for Keyword Extractor NLP.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class KeywordResult:
    """
    Structured representation of an extracted keyword or keyphrase.
    """
    keyword: str
    score: float
    ngram_size: int = 1
    frequency: int = 1
    positions: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to standard dictionary."""
        return asdict(self)

    def __repr__(self) -> str:
        return f"KeywordResult(keyword='{self.keyword}', score={self.score:.4f}, ngram={self.ngram_size}, freq={self.frequency})"


class BaseKeywordExtractor(ABC):
    """
    Abstract base class for all keyword extraction algorithms.
    """

    def __init__(self, name: str = "BaseExtractor"):
        self.name = name

    @abstractmethod
    def extract_keywords(
        self,
        text: str,
        top_k: int = 10,
        ngram_range: Tuple[int, int] = (1, 2),
        **kwargs
    ) -> List[KeywordResult]:
        """
        Extract ranked keywords from the input text.

        Args:
            text: Input text string.
            top_k: Maximum number of keywords to return.
            ngram_range: Tuple of (min_n, max_n) for keyword phrase lengths.
            **kwargs: Algorithm-specific parameters.

        Returns:
            List of KeywordResult objects sorted descending by score.
        """
        pass

    def extract_batch(
        self,
        texts: List[str],
        top_k: int = 10,
        ngram_range: Tuple[int, int] = (1, 2),
        **kwargs
    ) -> List[List[KeywordResult]]:
        """
        Extract keywords from a batch of texts.

        Args:
            texts: List of input text strings.
            top_k: Number of keywords per document.
            ngram_range: Tuple of (min_n, max_n).

        Returns:
            List of keyword results for each document.
        """
        return [
            self.extract_keywords(text, top_k=top_k, ngram_range=ngram_range, **kwargs)
            for text in texts
        ]
