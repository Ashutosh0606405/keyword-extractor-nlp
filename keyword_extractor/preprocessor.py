"""
Text preprocessing and candidate generation utilities for Keyword Extraction.
"""

import re
import string
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

# Clean stop words of apostrophes and contractions for sklearn compatibility
RAW_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn",
    "did", "didn", "do", "does", "doesn", "doing", "don", "down", "during", "each",
    "few", "for", "from", "further", "had", "hadn", "has", "hasn", "have", "haven",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "isn", "it", "its", "itself", "let",
    "me", "more", "most", "mustn", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
    "over", "own", "same", "shan", "she", "should", "shouldn", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn", "we", "were", "weren", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "won", "would", "wouldn", "you", "your",
    "yours", "yourself", "yourselves", "also", "using", "use", "used", "via", "often",
    "may", "might", "well", "within", "one", "two", "three", "first", "second"
}

DEFAULT_STOPWORDS: Set[str] = set(RAW_STOPWORDS)

# Try loading NLTK stopwords if available
try:
    from nltk.corpus import stopwords as nltk_stopwords
    for w in nltk_stopwords.words("english"):
        # Strip punctuation
        clean_w = re.sub(r"[^\w\s]", "", w).lower()
        if clean_w:
            DEFAULT_STOPWORDS.add(clean_w)
except Exception:
    pass


class TextPreprocessor:
    """
    NLP Text preprocessing engine providing cleaning, tokenization,
    POS tagging, and candidate n-gram generation.
    """

    def __init__(
        self,
        stopwords: Optional[Set[str]] = None,
        custom_stopwords: Optional[List[str]] = None,
        lowercase: bool = True,
        min_word_length: int = 2
    ):
        self.stopwords = set(stopwords or DEFAULT_STOPWORDS)
        if custom_stopwords:
            self.stopwords.update(w.lower() for w in custom_stopwords)
        self.lowercase = lowercase
        self.min_word_length = min_word_length

    def clean_text(self, text: str) -> str:
        """Strip URLs, abnormal whitespaces, and unsupported characters."""
        if not text:
            return ""
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        # Remove email addresses
        text = re.sub(r"\S+@\S+", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using punctuation delimiters."""
        clean = self.clean_text(text)
        if not clean:
            return []
        # Split on sentence boundaries (. ! ? followed by whitespace or end)
        sentences = re.split(r"(?<=[.!?])\s+", clean)
        return [s.strip() for s in sentences if s.strip()]

    def tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into alphanumeric words/tokens."""
        if self.lowercase:
            text = text.lower()
        # Find alphanumeric words (including hyphenated compound words)
        words = re.findall(r"\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*\b", text)
        return [w for w in words if len(w) >= self.min_word_length]

    def pos_tag(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Tag tokens with Part-of-Speech labels.
        Uses NLTK pos_tag if available; otherwise falls back to a rule-based heuristic.
        """
        try:
            import nltk
            return nltk.pos_tag(tokens)
        except Exception:
            # Simple rule-based heuristic fallback
            tagged = []
            for t in tokens:
                lower = t.lower()
                if lower in self.stopwords:
                    tagged.append((t, "DT"))
                elif t.endswith(("ing", "ed")):
                    tagged.append((t, "VBG"))
                elif t.endswith(("ly",)):
                    tagged.append((t, "RB"))
                elif t.endswith(("able", "ible", "al", "ful", "ic", "ish", "ive", "less", "ous")):
                    tagged.append((t, "JJ"))
                else:
                    tagged.append((t, "NN"))
            return tagged

    def generate_ngrams(
        self,
        tokens: List[str],
        ngram_range: Tuple[int, int] = (1, 2)
    ) -> List[str]:
        """Generate n-grams from a list of tokens within ngram_range."""
        min_n, max_n = ngram_range
        ngrams: List[str] = []
        n_tokens = len(tokens)
        for n in range(min_n, max_n + 1):
            for i in range(n_tokens - n + 1):
                window = tokens[i : i + n]
                # Filter out ngrams starting or ending with a stopword
                if window[0] in self.stopwords or window[-1] in self.stopwords:
                    continue
                # Don't include ngrams that are all numbers
                phrase = " ".join(window)
                if not all(w.isdigit() for w in window):
                    ngrams.append(phrase)
        return ngrams

    def extract_candidates_by_pos(
        self,
        text: str,
        ngram_range: Tuple[int, int] = (1, 3),
        allowed_pos_tags: Optional[Set[str]] = None
    ) -> List[str]:
        """
        Extract candidate phrases matching typical keyword patterns
        (sequences of zero or more adjectives followed by one or more nouns).
        """
        if allowed_pos_tags is None:
            # Nouns (NN, NNS, NNP, NNPS) and Adjectives (JJ, JJR, JJS)
            allowed_pos_tags = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"}

        tokens = self.tokenize_words(text)
        if not tokens:
            return []

        tagged = self.pos_tag(tokens)
        candidates: List[str] = []
        min_n, max_n = ngram_range

        current_phrase: List[str] = []
        for word, tag in tagged:
            word_lower = word.lower()
            if tag in allowed_pos_tags and word_lower not in self.stopwords:
                current_phrase.append(word_lower)
            else:
                if current_phrase:
                    for n in range(min_n, min(max_n + 1, len(current_phrase) + 1)):
                        for i in range(len(current_phrase) - n + 1):
                            cand = " ".join(current_phrase[i : i + n])
                            candidates.append(cand)
                    current_phrase = []

        if current_phrase:
            for n in range(min_n, min(max_n + 1, len(current_phrase) + 1)):
                for i in range(len(current_phrase) - n + 1):
                    cand = " ".join(current_phrase[i : i + n])
                    candidates.append(cand)

        return candidates

    def get_token_positions(self, text: str, phrase: str) -> List[int]:
        """Find character start positions of a phrase in the text (case-insensitive)."""
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        return [match.start() for match in pattern.finditer(text)]
