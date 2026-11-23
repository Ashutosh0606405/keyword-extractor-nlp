"""
Ensemble / Hybrid Keyword Extractor.
Combines multiple keyword extraction algorithms using Reciprocal Rank Fusion (RRF)
and consensus weighting to produce the most reliable, comprehensive keywords.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from keyword_extractor.base import BaseKeywordExtractor, KeywordResult
from keyword_extractor.algorithms.tfidf import TFIDFExtractor
from keyword_extractor.algorithms.rake import RAKEExtractor
from keyword_extractor.algorithms.textrank import TextRankExtractor
from keyword_extractor.algorithms.yake import YAKEExtractor
from keyword_extractor.algorithms.semantic import SemanticExtractor
from keyword_extractor.preprocessor import TextPreprocessor


class EnsembleExtractor(BaseKeywordExtractor):
    """
    Hybrid consensus keyword extractor combining:
    - TF-IDF (Frequency/IDF)
    - RAKE (Co-occurrence degree)
    - TextRank / PositionRank (Graph centrality)
    - YAKE (Statistical features)
    - SemanticMMR (Vector space similarity + diversity)
    """

    def __init__(
        self,
        preprocessor: Optional[TextPreprocessor] = None,
        extractors: Optional[List[BaseKeywordExtractor]] = None,
        weights: Optional[Dict[str, float]] = None,
        rrf_k: int = 60
    ):
        super().__init__(name="EnsembleHybrid")
        self.preprocessor = preprocessor or TextPreprocessor()
        self.rrf_k = rrf_k

        if extractors is not None:
            self.extractors = extractors
        else:
            self.extractors = [
                TFIDFExtractor(preprocessor=self.preprocessor),
                RAKEExtractor(preprocessor=self.preprocessor),
                TextRankExtractor(preprocessor=self.preprocessor, use_position_bias=True),
                YAKEExtractor(preprocessor=self.preprocessor),
                SemanticExtractor(preprocessor=self.preprocessor, diversity=0.5)
            ]

        # Weights per method
        self.weights = weights or {
            "TF-IDF": 1.0,
            "RAKE": 1.0,
            "PositionRank": 1.2,
            "TextRank": 1.0,
            "YAKE": 1.1,
            "SemanticMMR": 1.2
        }

    def extract_keywords(
        self,
        text: str,
        top_k: int = 10,
        ngram_range: Tuple[int, int] = (1, 2),
        **kwargs
    ) -> List[KeywordResult]:
        """
        Run all configured extractors and aggregate with Reciprocal Rank Fusion (RRF).
        """
        clean_text = self.preprocessor.clean_text(text)
        if not clean_text:
            return []

        # Run extractors with an expanded candidate pool (e.g., top 2 * top_k)
        pool_size = max(top_k * 2, 20)
        method_runs: Dict[str, List[KeywordResult]] = {}

        for ext in self.extractors:
            try:
                res = ext.extract_keywords(
                    clean_text, top_k=pool_size, ngram_range=ngram_range, **kwargs
                )
                method_runs[ext.name] = res
            except Exception:
                # If an individual extractor fails on edge-case text, continue gracefully
                continue

        if not method_runs:
            return []

        # Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = defaultdict(float)
        keyword_methods: Dict[str, List[str]] = defaultdict(list)
        keyword_positions: Dict[str, List[int]] = {}
        keyword_freqs: Dict[str, int] = {}

        for method_name, results in method_runs.items():
            weight = self.weights.get(method_name, 1.0)
            for rank_0, res in enumerate(results):
                kw = res.keyword.lower().strip()
                rank_1 = rank_0 + 1
                rrf_scores[kw] += weight / (self.rrf_k + rank_1)
                keyword_methods[kw].append(method_name)
                if kw not in keyword_positions and res.positions:
                    keyword_positions[kw] = res.positions
                if kw not in keyword_freqs or res.frequency > keyword_freqs[kw]:
                    keyword_freqs[kw] = res.frequency

        # Sort aggregated keywords
        sorted_kws = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        max_rrf = sorted_kws[0][1] if sorted_kws else 1.0

        results: List[KeywordResult] = []
        for kw, raw_score in sorted_kws:
            positions = keyword_positions.get(kw) or self.preprocessor.get_token_positions(text, kw)
            freq = keyword_freqs.get(kw, len(positions) or 1)
            norm_score = round(raw_score / max_rrf, 4) if max_rrf > 0 else 0.0

            results.append(
                KeywordResult(
                    keyword=kw,
                    score=norm_score,
                    ngram_size=len(kw.split()),
                    frequency=freq,
                    positions=positions,
                    metadata={
                        "method": "EnsembleHybrid",
                        "raw_rrf": round(raw_score, 6),
                        "vote_count": len(keyword_methods[kw]),
                        "supporting_methods": keyword_methods[kw]
                    }
                )
            )

            if len(results) >= top_k:
                break

        return results
