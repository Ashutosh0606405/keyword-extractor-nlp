"""
YAKE-inspired (Yet Another Keyword Extractor) statistical feature extractor.
Extracts keywords from single documents using statistical features: casing, position,
frequency, context relatedness, and sentence dispersion.
"""

from collections import Counter, defaultdict
import math
from typing import Dict, List, Optional, Set, Tuple

from keyword_extractor.base import BaseKeywordExtractor, KeywordResult
from keyword_extractor.preprocessor import TextPreprocessor


class YAKEExtractor(BaseKeywordExtractor):
    """
    Lightweight, fast statistical keyword extractor based on the YAKE algorithm.
    Evaluates individual token properties and candidate n-grams.
    """

    def __init__(
        self,
        preprocessor: Optional[TextPreprocessor] = None,
        window_size: int = 2
    ):
        super().__init__(name="YAKE")
        self.preprocessor = preprocessor or TextPreprocessor()
        self.window_size = window_size

    def extract_keywords(
        self,
        text: str,
        top_k: int = 10,
        ngram_range: Tuple[int, int] = (1, 2),
        **kwargs
    ) -> List[KeywordResult]:
        """
        Extract ranked keywords using statistical feature scoring.
        """
        clean_text = self.preprocessor.clean_text(text)
        if not clean_text:
            return []

        sentences = self.preprocessor.split_sentences(clean_text)
        if not sentences:
            sentences = [clean_text]

        # Tokenize per sentence preserving original case for casing feature
        sent_tokens: List[List[str]] = []
        for s in sentences:
            toks = self.preprocessor.tokenize_words(s)
            if toks:
                sent_tokens.append(toks)

        all_tokens = [tok for sent in sent_tokens for tok in sent]
        if not all_tokens:
            return []

        total_words = len(all_tokens)
        word_freq = Counter(t.lower() for t in all_tokens)
        freq_values = list(word_freq.values())
        mean_freq = sum(freq_values) / len(freq_values) if freq_values else 1.0
        std_freq = math.sqrt(
            sum((f - mean_freq) ** 2 for f in freq_values) / max(1, len(freq_values))
        ) or 1.0

        # Feature 1: Casing (capitalized, all-caps acronym vs lowercase)
        casing_scores: Dict[str, float] = {}
        for w in word_freq:
            # Check occurrences in original text
            raw_cases = [t for t in all_tokens if t.lower() == w]
            cap_count = sum(1 for t in raw_cases if t.istitle())
            upper_count = sum(1 for t in raw_cases if t.isupper() and len(t) > 1)
            total = len(raw_cases) or 1
            # Higher casing score means more prominent
            casing_scores[w] = max(cap_count, upper_count) / total

        # Feature 2: Position (first and median occurrence)
        positions: Dict[str, List[int]] = defaultdict(list)
        for idx, tok in enumerate(all_tokens):
            positions[tok.lower()].append(idx)

        position_scores: Dict[str, float] = {}
        for w, pos_list in positions.items():
            first_pos = pos_list[0]
            # Earlier words get smaller values (more important)
            position_scores[w] = math.log(math.log(3.0 + first_pos))

        # Feature 3: Dispersion (number of sentences containing token / total sentences)
        num_sentences = len(sent_tokens) or 1
        sent_dispersion: Dict[str, float] = {}
        for w in word_freq:
            count = sum(1 for sent in sent_tokens if any(t.lower() == w for t in sent))
            sent_dispersion[w] = count / float(num_sentences)

        # Feature 4: Relatedness to context (co-occurrences in sliding window)
        left_contexts: Dict[str, Set[str]] = defaultdict(set)
        right_contexts: Dict[str, Set[str]] = defaultdict(set)

        for sent in sent_tokens:
            for i, tok in enumerate(sent):
                w = tok.lower()
                for step in range(1, self.window_size + 1):
                    if i - step >= 0:
                        left_contexts[w].add(sent[i - step].lower())
                    if i + step < len(sent):
                        right_contexts[w].add(sent[i + step].lower())

        rel_scores: Dict[str, float] = {}
        for w in word_freq:
            wl = len(left_contexts[w])
            wr = len(right_contexts[w])
            rel_scores[w] = 1.0 + (wl + wr) * (word_freq[w] / float(total_words))

        # Compute combined word scores
        # Lower S(w) indicates a more distinctive keyword
        word_s: Dict[str, float] = {}
        for w in word_freq:
            if w in self.preprocessor.stopwords:
                word_s[w] = float("inf")
                continue

            w_case = casing_scores.get(w, 0.0) + 1.0
            w_pos = position_scores.get(w, 1.0)
            w_freq = word_freq[w] / (mean_freq + std_freq)
            w_disp = sent_dispersion.get(w, 0.1)
            w_rel = rel_scores.get(w, 1.0)

            denom = w_case + (w_freq / w_rel) + (w_disp / w_rel)
            if denom <= 0:
                denom = 1e-6
            s = (w_rel * w_pos) / denom
            word_s[w] = s

        # Generate candidate n-grams
        min_n, max_n = ngram_range
        candidates: Dict[str, float] = {}
        phrase_freqs: Dict[str, int] = defaultdict(int)

        for sent in sent_tokens:
            n_tokens = len(sent)
            for n in range(min_n, min(max_n + 1, n_tokens + 1)):
                for i in range(n_tokens - n + 1):
                    ngram_tokens = [sent[i + j].lower() for j in range(n)]

                    # Stopword rules: cannot start or end with a stopword
                    if ngram_tokens[0] in self.preprocessor.stopwords:
                        continue
                    if ngram_tokens[-1] in self.preprocessor.stopwords:
                        continue
                    if any(len(t) < self.preprocessor.min_word_length for t in ngram_tokens):
                        continue

                    phrase = " ".join(ngram_tokens)
                    phrase_freqs[phrase] += 1

                    # Compute n-gram score: product of word scores divided by (1 + log(frequency))
                    product_s = 1.0
                    valid = True
                    for tok in ngram_tokens:
                        score = word_s.get(tok, float("inf"))
                        if math.isinf(score):
                            valid = False
                            break
                        product_s *= score

                    if valid:
                        # Frequency boost
                        cand_score = product_s / (1.0 + math.log(phrase_freqs[phrase] + 1))
                        # Keep lowest score (best)
                        if phrase not in candidates or cand_score < candidates[phrase]:
                            candidates[phrase] = cand_score

        if not candidates:
            return []

        # In YAKE, lower is better. Sort ascending by raw score
        sorted_cands = sorted(candidates.items(), key=lambda x: x[1])

        # Invert scores for unified output: higher is better in [0.0, 1.0]
        min_score = sorted_cands[0][1]
        results: List[KeywordResult] = []

        for phrase, raw_s in sorted_cands:
            positions_list = self.preprocessor.get_token_positions(text, phrase)
            frequency = phrase_freqs.get(phrase, len(positions_list) or 1)

            # Invert: score = min_score / raw_s (so top candidate gets 1.0)
            norm_score = round(min_score / (raw_s + 1e-9), 4)
            norm_score = min(1.0, max(0.01, norm_score))

            results.append(
                KeywordResult(
                    keyword=phrase,
                    score=norm_score,
                    ngram_size=len(phrase.split()),
                    frequency=frequency,
                    positions=positions_list,
                    metadata={"method": "YAKE", "raw_yake_score": raw_s}
                )
            )

            if len(results) >= top_k:
                break

        return results
