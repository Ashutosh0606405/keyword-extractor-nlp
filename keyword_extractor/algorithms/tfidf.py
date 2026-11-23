"""
TF-IDF (Term Frequency - Inverse Document Frequency) Keyword Extractor.
Supports both single-document pseudo-corpus mode and multi-document corpus mode.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from keyword_extractor.base import BaseKeywordExtractor, KeywordResult
from keyword_extractor.preprocessor import TextPreprocessor


class TFIDFExtractor(BaseKeywordExtractor):
    """
    Extracts keywords using TF-IDF weighting.
    When a background corpus is provided, uses true corpus IDF.
    When operating on a single document, decomposes the document into sentences
    to build a local pseudo-corpus, ensuring meaningful IDF discrimination.
    """

    def __init__(
        self,
        corpus: Optional[List[str]] = None,
        preprocessor: Optional[TextPreprocessor] = None,
        sublinear_tf: bool = True,
        max_df: float = 0.95,
        min_df: Union[int, float] = 1
    ):
        super().__init__(name="TF-IDF")
        self.preprocessor = preprocessor or TextPreprocessor()
        self.sublinear_tf = sublinear_tf
        self.max_df = max_df
        self.min_df = min_df
        self.corpus = corpus
        self.vectorizer: Optional[TfidfVectorizer] = None

        if corpus:
            self._fit_corpus(corpus)

    def _fit_corpus(self, corpus: List[str]) -> None:
        """Fit vectorizer on a user-provided background corpus."""
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words=list(self.preprocessor.stopwords),
            sublinear_tf=self.sublinear_tf,
            max_df=self.max_df,
            min_df=self.min_df
        )
        self.vectorizer.fit(corpus)

    def extract_keywords(
        self,
        text: str,
        top_k: int = 10,
        ngram_range: Tuple[int, int] = (1, 2),
        **kwargs
    ) -> List[KeywordResult]:
        """
        Extract top-k keywords from text.
        """
        clean_text = self.preprocessor.clean_text(text)
        if not clean_text:
            return []

        # If a pre-fitted vectorizer exists with matching ngram_range, use it
        if self.vectorizer is not None:
            vec = self.vectorizer
            tfidf_matrix = vec.transform([clean_text])
            feature_names = vec.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
        else:
            # Build pseudo-corpus from sentences
            sentences = self.preprocessor.split_sentences(clean_text)
            if len(sentences) < 2:
                # Fallback to chunking into smaller token windows
                tokens = self.preprocessor.tokenize_words(clean_text)
                chunk_size = 15
                sentences = [
                    " ".join(tokens[i : i + chunk_size])
                    for i in range(0, len(tokens), chunk_size)
                ]

            if not sentences:
                sentences = [clean_text]

            vec = TfidfVectorizer(
                ngram_range=ngram_range,
                stop_words=list(self.preprocessor.stopwords),
                sublinear_tf=self.sublinear_tf
            )
            try:
                tfidf_matrix = vec.fit_transform(sentences)
                feature_names = vec.get_feature_names_out()
                # Aggregate scores across all sentences
                scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
            except ValueError:
                # Text contains only stop words or no valid tokens
                return []

        # Filter by ngram_range if pre-fitted vectorizer was used
        ranked_indices = np.argsort(scores)[::-1]
        results: List[KeywordResult] = []
        min_n, max_n = ngram_range

        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue

            phrase = str(feature_names[idx])
            words = phrase.split()
            n = len(words)

            if n < min_n or n > max_n:
                continue

            # Skip phrases made solely of very short tokens
            if all(len(w) < self.preprocessor.min_word_length for w in words):
                continue

            positions = self.preprocessor.get_token_positions(clean_text, phrase)
            frequency = len(positions) or 1

            results.append(
                KeywordResult(
                    keyword=phrase,
                    score=score,
                    ngram_size=n,
                    frequency=frequency,
                    positions=positions,
                    metadata={"method": "TF-IDF"}
                )
            )

            if len(results) >= top_k:
                break

        # Normalize scores to 0-1 range for consistency
        if results and results[0].score > 0:
            max_s = results[0].score
            for res in results:
                res.score = round(res.score / max_s, 4)

        return results
