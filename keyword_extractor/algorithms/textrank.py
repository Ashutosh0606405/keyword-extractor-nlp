"""
Graph-based keyword extraction: TextRank and PositionRank algorithms.
Uses co-occurrence graph centrality and PageRank scoring.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx

from keyword_extractor.base import BaseKeywordExtractor, KeywordResult
from keyword_extractor.preprocessor import TextPreprocessor


class TextRankExtractor(BaseKeywordExtractor):
    """
    TextRank algorithm for keyword extraction.
    Constructs a word co-occurrence graph and scores nodes using PageRank.
    Includes PositionRank option to bias ranking towards terms appearing earlier in the text.
    """

    def __init__(
        self,
        preprocessor: Optional[TextPreprocessor] = None,
        window_size: int = 3,
        use_position_bias: bool = False,
        damping: float = 0.85,
        max_iter: int = 100
    ):
        name = "PositionRank" if use_position_bias else "TextRank"
        super().__init__(name=name)
        self.preprocessor = preprocessor or TextPreprocessor()
        self.window_size = window_size
        self.use_position_bias = use_position_bias
        self.damping = damping
        self.max_iter = max_iter

    def extract_keywords(
        self,
        text: str,
        top_k: int = 10,
        ngram_range: Tuple[int, int] = (1, 2),
        **kwargs
    ) -> List[KeywordResult]:
        """
        Extract ranked keywords/keyphrases using PageRank on the co-occurrence graph.
        """
        clean_text = self.preprocessor.clean_text(text)
        if not clean_text:
            return []

        tokens = self.preprocessor.tokenize_words(clean_text)
        if not tokens:
            return []

        # POS tagging: filter to nouns and adjectives
        tagged = self.preprocessor.pos_tag(tokens)
        valid_pos = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"}

        # Candidate word tokens and their first occurrence positions
        filtered_tokens: List[Tuple[str, int]] = []
        first_positions: Dict[str, int] = {}

        for idx, (token, tag) in enumerate(tagged):
            w = token.lower()
            if tag in valid_pos and w not in self.preprocessor.stopwords:
                filtered_tokens.append((w, idx))
                if w not in first_positions:
                    first_positions[w] = idx + 1  # 1-indexed for reciprocal

        if not filtered_tokens:
            return []

        # Build co-occurrence graph
        graph = nx.Graph()
        for w, _ in filtered_tokens:
            graph.add_node(w)

        num_tokens = len(filtered_tokens)
        for i in range(num_tokens):
            w1, _ = filtered_tokens[i]
            for j in range(i + 1, min(i + self.window_size, num_tokens)):
                w2, _ = filtered_tokens[j]
                if w1 != w2:
                    if graph.has_edge(w1, w2):
                        graph[w1][w2]["weight"] += 1.0
                    else:
                        graph.add_edge(w1, w2, weight=1.0)

        # Personalization vector for PositionRank
        personalization = None
        if self.use_position_bias:
            personalization = {}
            total_weight = 0.0
            for node in graph.nodes():
                pos = first_positions.get(node, 1)
                weight = 1.0 / pos
                personalization[node] = weight
                total_weight += weight

            if total_weight > 0:
                personalization = {k: v / total_weight for k, v in personalization.items()}
            else:
                personalization = None

        try:
            word_scores: Dict[str, float] = nx.pagerank(
                graph,
                alpha=self.damping,
                max_iter=self.max_iter,
                weight="weight",
                personalization=personalization
            )
        except Exception:
            # Fallback uniform scores if graph is disconnected or disconnected components fail
            word_scores = {node: 1.0 / len(graph.nodes()) for node in graph.nodes()}

        # Reconstruct multi-word keyphrases from original sequence
        min_n, max_n = ngram_range
        candidate_phrases: Dict[str, float] = {}
        phrase_freqs: Dict[str, int] = defaultdict(int)

        # 1. Unigram candidates
        if min_n <= 1:
            for w, s in word_scores.items():
                candidate_phrases[w] = s
                phrase_freqs[w] = sum(1 for tok, _ in filtered_tokens if tok == w)

        # 2. Multi-word phrases: find adjacent qualifying words in token stream
        if max_n >= 2:
            current_chain: List[str] = []
            last_idx = -2

            for w, orig_idx in filtered_tokens:
                if orig_idx == last_idx + 1:
                    current_chain.append(w)
                else:
                    if len(current_chain) >= 2:
                        self._collect_ngram_phrases(
                            current_chain, word_scores, min_n, max_n, candidate_phrases, phrase_freqs
                        )
                    current_chain = [w]
                last_idx = orig_idx

            if len(current_chain) >= 2:
                self._collect_ngram_phrases(
                    current_chain, word_scores, min_n, max_n, candidate_phrases, phrase_freqs
                )

        if not candidate_phrases:
            return []

        sorted_candidates = sorted(
            candidate_phrases.items(),
            key=lambda item: item[1],
            reverse=True
        )

        max_score = sorted_candidates[0][1] if sorted_candidates else 1.0
        results: List[KeywordResult] = []

        for phrase, raw_score in sorted_candidates:
            words = phrase.split()
            n = len(words)
            if not (min_n <= n <= max_n):
                continue

            positions = self.preprocessor.get_token_positions(text, phrase)
            frequency = phrase_freqs.get(phrase, len(positions) or 1)
            norm_score = round(raw_score / max_score, 4) if max_score > 0 else 0.0

            results.append(
                KeywordResult(
                    keyword=phrase,
                    score=norm_score,
                    ngram_size=n,
                    frequency=frequency,
                    positions=positions,
                    metadata={"method": self.name, "raw_score": raw_score}
                )
            )

            if len(results) >= top_k:
                break

        return results

    def _collect_ngram_phrases(
        self,
        chain: List[str],
        word_scores: Dict[str, float],
        min_n: int,
        max_n: int,
        candidate_phrases: Dict[str, float],
        phrase_freqs: Dict[str, int]
    ) -> None:
        """Extract multi-word sub-phrases from a contiguous chain of valid tokens."""
        chain_len = len(chain)
        for n in range(max(2, min_n), min(max_n + 1, chain_len + 1)):
            for i in range(chain_len - n + 1):
                sub = chain[i : i + n]
                phrase = " ".join(sub)
                phrase_freqs[phrase] += 1
                # Phrase score is sum of constituent word scores
                score = sum(word_scores.get(w, 0.0) for w in sub)
                candidate_phrases[phrase] = max(candidate_phrases.get(phrase, 0.0), score)
