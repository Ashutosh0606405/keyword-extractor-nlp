#!/usr/bin/env python3
"""
Interactive Demonstration Script for KeySense NLP Keyword Extractor.
Run: python3 demo.py
"""

import sys
import os

# Add package directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from keyword_extractor import KeywordExtractorPipeline, run_benchmark


SAMPLE_TEXT = """
Artificial intelligence and natural language processing have undergone a paradigm shift with
large language models and deep neural architectures. Transformer models utilize self-attention
mechanisms to compute representations of input sequences without recurrent neural connections.
Modern foundation models excel at complex semantic understanding, zero-shot question answering,
code generation, and automated text summarization across diverse enterprise applications.
However, training large deep learning models requires substantial computational resources, high-memory
GPUs, and curated datasets to mitigate hallucinations and alignment challenges.
"""


def main():
    print("=" * 75)
    print("      KeySense NLP: Production-Grade Keyword & Keyphrase Extractor     ")
    print("=" * 75)

    pipeline = KeywordExtractorPipeline()

    print("\n[1] Running Ensemble Hybrid Extractor (Consensus of 5 Algorithms)...")
    results = pipeline.extract(SAMPLE_TEXT, top_k=8, ngram_range=(1, 2))

    print(f"\n{'Rank':<5} | {'Keyword':<32} | {'Score':<8} | {'N-Gram':<6} | {'Frequency':<9} | {'Votes'}")
    print("-" * 75)
    for rank, r in enumerate(results, 1):
        votes = len(r.metadata.get("supporting_methods", []))
        vote_str = f"{votes} algorithms" if votes else "1"
        print(f"{rank:<5} | {r.keyword:<32} | {r.score:<8.4f} | {r.ngram_size:<6} | {r.frequency:<9} | {vote_str}")
    print("-" * 75)

    print("\n[2] Algorithm Comparison (Top 4 Keywords Across Each Paradigm):")
    comparison = pipeline.compare(SAMPLE_TEXT, top_k=4)
    for method_name, kws in comparison.items():
        kw_list = ", ".join([f"'{k.keyword}' ({k.score:.2f})" for k in kws])
        print(f"  • {method_name:<15}: {kw_list}")

    print("\n[3] Automated Benchmark Evaluation against Reference Gold-Standard:")
    bench_results = run_benchmark(pipeline, top_k=10)
    print(f"\n{'Algorithm':<16} | {'Precision@K':<12} | {'Recall@K':<10} | {'F1@K':<8} | {'Soft-F1':<8} | {'MRR':<6}")
    print("-" * 75)
    for method, metrics in bench_results.items():
        print(f"{method:<16} | {metrics['precision@k']:<12.4f} | {metrics['recall@k']:<10.4f} | {metrics['f1@k']:<8.4f} | {metrics['soft_f1@k']:<8.4f} | {metrics['mrr']:<6.4f}")
    print("-" * 75)

    print("\n[4] Interfaces Ready:")
    print("  • Web UI (Streamlit): python3 -m streamlit run web/app.py")
    print("  • REST API & UI:      python3 -m keyword_extractor.server --port 8000")
    print("  • CLI Tool:           python3 -m keyword_extractor.cli extract \"Your text here\"")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
