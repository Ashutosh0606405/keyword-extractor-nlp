"""
Semantic Keyword Extractor (KeyBERT-style with Maximal Marginal Relevance).
Extracts keywords using vector space representations and cosine similarity,
with MMR diversity optimization to prevent duplicate or repetitive keywords.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from keyword_extractor.base import BaseKeywordExtractor, KeywordResult
from keyword_extractor.preprocessor import TextPreprocessor


class SemanticExtractor(BaseKeywordExtractor):
    """
    Semantic embedding-based keyword extractor.
    Extracts candidate phrases, calculates cosine similarity against document embedding,
    and diversifies results using Maximal Marginal Relevance (MMR).
    """

    def __init__(
        self,
        preprocessor: Optional[TextPreprocessor] = None,
        diversity: float = 0.5,
        model_name: Optional[str] = None
    ):
        super().__init__(name="SemanticMMR")
        self.preprocessor = preprocessor or TextPreprocessor()
        self.diversity = max(0.0, min(1.0, diversity))
        self.model_name = model_name
        self._st_model = None

        if model_name:
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(model_name)
            except Exception:
                self._st_model = None

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute dense vector embeddings for input strings.
        Uses SentenceTransformer if loaded; otherwise uses a robust character+word
        sublinear TF-IDF vectorizer projected into normalized metric space.
        """
        if self._st_model is not None:
            return self._st_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        # High-dimensional subword & word n-gram feature representation
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True
        )
        matrix = vec.fit_transform(texts).toarray()
        # L2-normalize vectors
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _maximal_marginal_relevance(
        self,
        doc_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        candidates: List[str],
        top_k: int = 10,
        diversity: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Select candidates using MMR:
        MMR = argmax_{c} [ (1 - diversity) * sim(c, doc) - diversity * max_{s} sim(c, s) ]
        """
        # doc_embedding shape: (1, dim), candidate_embeddings: (num_cands, dim)
        doc_sims = cosine_similarity(candidate_embeddings, doc_embedding).reshape(-1)
        cand_sims = cosine_similarity(candidate_embeddings, candidate_embeddings)

        selected_indices: List[int] = []
        unselected_indices = list(range(len(candidates)))

        # Pick the most similar candidate first
        best_first = int(np.argmax(doc_sims))
        selected_indices.append(best_first)
        unselected_indices.remove(best_first)

        top_k = min(top_k, len(candidates))

        while len(selected_indices) < top_k and unselected_indices:
            mmr_scores = []
            for idx in unselected_indices:
                rel = doc_sims[idx]
                redundancy = max(cand_sims[idx, s_idx] for s_idx in selected_indices)
                mmr = (1.0 - diversity) * rel - (diversity * redundancy)
                mmr_scores.append((mmr, idx))

            mmr_scores.sort(key=lambda x: x[0], reverse=True)
            chosen_idx = mmr_scores[0][1]
            selected_indices.append(chosen_idx)
            unselected_indices.remove(chosen_idx)

        return [(candidates[idx], float(doc_sims[idx])) for idx in selected_indices]

    def extract_keywords(
        self,
        text: str,
        top_k: int = 10,
        ngram_range: Tuple[int, int] = (1, 2),
        diversity: Optional[float] = None,
        **kwargs
    ) -> List[KeywordResult]:
        """
        Extract semantically relevant and diverse keywords using MMR.
        """
        clean_text = self.preprocessor.clean_text(text)
        if not clean_text:
            return []

        # Candidate generation: POS-pattern or n-grams
        tokens = self.preprocessor.tokenize_words(clean_text)
        if not tokens:
            return []

        candidates = self.preprocessor.extract_candidates_by_pos(
            clean_text, ngram_range=ngram_range
        )
        if not candidates:
            candidates = self.preprocessor.generate_ngrams(
                tokens, ngram_range=ngram_range
            )

        # Deduplicate candidates while preserving order
        unique_cands: List[str] = []
        seen = set()
        for c in candidates:
            if c not in seen and len(c) >= self.preprocessor.min_word_length:
                seen.add(c)
                unique_cands.append(c)

        if not unique_cands:
            return []

        # Compute embeddings for document and candidates
        all_texts = [clean_text] + unique_cands
        embeddings = self._get_embeddings(all_texts)
        doc_emb = embeddings[0:1]
        cand_embs = embeddings[1:]

        div = self.diversity if diversity is None else max(0.0, min(1.0, diversity))

        mmr_results = self._maximal_marginal_relevance(
            doc_embedding=doc_emb,
            candidate_embeddings=cand_embs,
            candidates=unique_cands,
            top_k=top_k,
            diversity=div
        )

        results: List[KeywordResult] = []
        max_score = mmr_results[0][1] if mmr_results and mmr_results[0][1] > 0 else 1.0

        for phrase, sim_score in mmr_results:
            positions = self.preprocessor.get_token_positions(text, phrase)
            norm_score = round(max(0.0, sim_score / max_score), 4)

            results.append(
                KeywordResult(
                    keyword=phrase,
                    score=norm_score,
                    ngram_size=len(phrase.split()),
                    frequency=len(positions) or 1,
                    positions=positions,
                    metadata={
                        "method": "SemanticMMR",
                        "raw_similarity": round(sim_score, 4),
                        "diversity": div
                    }
                )
            )

        return results
