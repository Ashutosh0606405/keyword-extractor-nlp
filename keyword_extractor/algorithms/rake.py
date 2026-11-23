"""
Rapid Automatic Keyword Extraction (RAKE) algorithm.
Domain-independent keyword extraction using word co-occurrence within candidate phrases.
"""

from collections import Counter, defaultdict
import re
from typing import Dict, List, Optional, Set, Tuple

from keyword_extractor.base import BaseKeywordExtractor, KeywordResult
from keyword_extractor.preprocessor import TextPreprocessor


class RAKEExtractor(BaseKeywordExtractor):
    """
    RAKE extracts multi-word keywords by using stop words and punctuation
    as natural phrase delimiters, then scoring words based on their co-occurrence degrees.
    """

    def __init__(
        self,
        preprocessor: Optional[TextPreprocessor] = None,
        min_char_length: int = 3,
        max_words_length: int = 4,
        min_keyword_frequency: int = 1
    ):
        super().__init__(name="RAKE")
        self.preprocessor = preprocessor or TextPreprocessor()
        self.min_char_length = min_char_length
        self.max_words_length = max_words_length
        self.min_keyword_frequency = min_keyword_frequency

    def _split_into_candidate_phrases(self, text: str) -> List[str]:
        """
        Split text into candidate phrases using punctuation and stopwords as delimiters.
        """
        clean_text = self.preprocessor.clean_text(text)
        # Sentence splitting pattern
        sentence_delimiters = r"[.!?,;:\"\t\n\r\(\)]+"
        sentences = re.split(sentence_delimiters, clean_text)

        # Build stopword regex pattern
        stopwords = sorted(self.preprocessor.stopwords, key=len, reverse=True)
        escaped_stopwords = [r"\b" + re.escape(sw) + r"\b" for sw in stopwords]
        stopword_pattern = re.compile("|".join(escaped_stopwords), re.IGNORECASE)

        candidate_phrases: List[str] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # Replace stopwords with a delimiter character '|'
            phrases = stopword_pattern.split(sentence)
            for phrase in phrases:
                phrase = phrase.strip().lower()
                # Clean stray punctuation and whitespace
                phrase = re.sub(r"^[^\w]+|[^\w]+$", "", phrase)
                phrase = re.sub(r"\s+", " ", phrase)
                if phrase and len(phrase) >= self.min_char_length:
                    words = phrase.split()
                    if 1 <= len(words) <= self.max_words_length:
                        candidate_phrases.append(phrase)

        return candidate_phrases

    def _calculate_word_scores(
        self, candidate_phrases: List[str]
    ) -> Tuple[Dict[str, float], Dict[str, int]]:
        """
        Calculate word degree and frequency to compute word scores: degree / frequency.
        """
        word_freq: Dict[str, int] = defaultdict(int)
        word_degree: Dict[str, int] = defaultdict(int)

        for phrase in candidate_phrases:
            words = phrase.split()
            phrase_length = len(words)
            phrase_degree = phrase_length - 1

            for word in words:
                word_freq[word] += 1
                word_degree[word] += phrase_degree

        for word in word_freq:
            word_degree[word] = word_degree[word] + word_freq[word]

        word_scores: Dict[str, float] = {}
        for word in word_freq:
            word_scores[word] = word_degree[word] / float(word_freq[word])

        return word_scores, word_freq

    def extract_keywords(
        self,
        text: str,
        top_k: int = 10,
        ngram_range: Tuple[int, int] = (1, 2),
        **kwargs
    ) -> List[KeywordResult]:
        """
        Extract top-k ranked keywords using RAKE.
        """
        if not text or not text.strip():
            return []

        candidate_phrases = self._split_into_candidate_phrases(text)
        if not candidate_phrases:
            return []

        word_scores, _ = self._calculate_word_scores(candidate_phrases)

        # Score candidate phrases
        phrase_freq = Counter(candidate_phrases)
        phrase_scores: Dict[str, float] = {}

        min_n, max_n = ngram_range

        for phrase, freq in phrase_freq.items():
            if freq < self.min_keyword_frequency:
                continue
            words = phrase.split()
            n = len(words)
            if not (min_n <= n <= max_n):
                continue

            score = sum(word_scores.get(w, 0.0) for w in words)
            phrase_scores[phrase] = score

        # Sort candidates descending
        sorted_candidates = sorted(
            phrase_scores.items(), key=lambda item: (item[1], phrase_freq[item[0]]), reverse=True
        )

        results: List[KeywordResult] = []
        max_score = sorted_candidates[0][1] if sorted_candidates else 1.0

        for phrase, raw_score in sorted_candidates:
            positions = self.preprocessor.get_token_positions(text, phrase)
            norm_score = round(raw_score / max_score, 4) if max_score > 0 else 0.0

            results.append(
                KeywordResult(
                    keyword=phrase,
                    score=norm_score,
                    ngram_size=len(phrase.split()),
                    frequency=phrase_freq[phrase],
                    positions=positions,
                    metadata={"method": "RAKE", "raw_score": raw_score}
                )
            )

            if len(results) >= top_k:
                break

        return results
