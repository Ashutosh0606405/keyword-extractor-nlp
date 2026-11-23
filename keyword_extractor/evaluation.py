"""
Evaluation and benchmark metrics for Keyword Extraction.
Computes Precision@K, Recall@K, F1@K, MRR, MAP, and runs automated benchmarks.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from keyword_extractor.base import KeywordResult
from keyword_extractor.datasets import BenchmarkDocument, BENCHMARK_DATASET


@dataclass
class EvaluationScores:
    """Evaluation metrics for a single document or aggregated benchmark."""
    precision: float
    recall: float
    f1_score: float
    soft_precision: float
    soft_recall: float
    soft_f1: float
    mrr: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "precision@k": round(self.precision, 4),
            "recall@k": round(self.recall, 4),
            "f1@k": round(self.f1_score, 4),
            "soft_precision@k": round(self.soft_precision, 4),
            "soft_recall@k": round(self.soft_recall, 4),
            "soft_f1@k": round(self.soft_f1, 4),
            "mrr": round(self.mrr, 4)
        }


def _normalize_phrase(phrase: str) -> str:
    """Normalize phrase for matching."""
    return " ".join(phrase.lower().strip().split())


def _is_soft_match(pred: str, gold: str) -> bool:
    """Check if prediction and gold keyword share significant word overlap or substring."""
    p_norm = _normalize_phrase(pred)
    g_norm = _normalize_phrase(gold)
    if p_norm == g_norm or p_norm in g_norm or g_norm in p_norm:
        return True
    # Word set overlap
    p_words = set(p_norm.split())
    g_words = set(g_norm.split())
    overlap = p_words.intersection(g_words)
    if overlap and (len(overlap) / max(len(p_words), len(g_words))) >= 0.5:
        return True
    return False


def evaluate_predictions(
    predicted_keywords: List[KeywordResult],
    gold_keywords: List[str],
    top_k: int = 10
) -> EvaluationScores:
    """
    Evaluate predicted keywords against gold-standard keywords.
    """
    preds = [_normalize_phrase(r.keyword) for r in predicted_keywords[:top_k]]
    golds = [_normalize_phrase(g) for g in gold_keywords]

    if not preds or not golds:
        return EvaluationScores(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # 1. Exact matches
    exact_matches = sum(1 for p in preds if p in golds)
    precision = exact_matches / float(len(preds))
    recall = exact_matches / float(len(golds))
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 2. Soft matches
    soft_matches = 0
    matched_golds: Set[str] = set()
    for p in preds:
        for g in golds:
            if g not in matched_golds and _is_soft_match(p, g):
                soft_matches += 1
                matched_golds.add(g)
                break

    soft_prec = soft_matches / float(len(preds))
    soft_rec = soft_matches / float(len(golds))
    soft_f1 = (2 * soft_prec * soft_rec) / (soft_prec + soft_rec) if (soft_prec + soft_rec) > 0 else 0.0

    # 3. MRR (Mean Reciprocal Rank of first relevant keyword)
    mrr = 0.0
    for rank_0, p in enumerate(preds):
        if any(_is_soft_match(p, g) for g in golds):
            mrr = 1.0 / (rank_0 + 1)
            break

    return EvaluationScores(
        precision=precision,
        recall=recall,
        f1_score=f1,
        soft_precision=soft_prec,
        soft_recall=soft_rec,
        soft_f1=soft_f1,
        mrr=mrr
    )


def run_benchmark(
    pipeline,
    methods: Optional[List[str]] = None,
    dataset: Optional[List[BenchmarkDocument]] = None,
    top_k: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Run an end-to-end benchmark across multiple documents and methods.
    Returns averaged metrics for each method.
    """
    eval_dataset = dataset or BENCHMARK_DATASET
    eval_methods = methods or ["ensemble", "tfidf", "rake", "positionrank", "yake", "semantic"]
    benchmark_results: Dict[str, Dict[str, float]] = {}

    for method in eval_methods:
        doc_scores: List[EvaluationScores] = []
        for doc in eval_dataset:
            preds = pipeline.extract(doc.text, top_k=top_k, method=method)
            score = evaluate_predictions(preds, doc.gold_keywords, top_k=top_k)
            doc_scores.append(score)

        # Average across documents
        avg_scores = EvaluationScores(
            precision=float(np.mean([s.precision for s in doc_scores])),
            recall=float(np.mean([s.recall for s in doc_scores])),
            f1_score=float(np.mean([s.f1_score for s in doc_scores])),
            soft_precision=float(np.mean([s.soft_precision for s in doc_scores])),
            soft_recall=float(np.mean([s.soft_recall for s in doc_scores])),
            soft_f1=float(np.mean([s.soft_f1 for s in doc_scores])),
            mrr=float(np.mean([s.mrr for s in doc_scores]))
        )
        benchmark_results[method] = avg_scores.to_dict()

    return benchmark_results
