"""
Command-Line Interface (CLI) for KeySense NLP.
Provides subcommands for single string extraction, file processing, side-by-side comparison,
benchmark evaluation, and starting the API/Web server.
"""

import argparse
import json
import sys
from typing import List

from keyword_extractor.pipeline import KeywordExtractorPipeline
from keyword_extractor.evaluation import run_benchmark
from keyword_extractor.server import start_server


def print_table(results: List, title: str = "Extracted Keywords"):
    """Format and print results as an ASCII table."""
    if not results:
        print("\nNo keywords extracted.\n")
        return

    print(f"\n=== {title} ===")
    header = f"{'Rank':<5} | {'Keyword':<32} | {'Score':<8} | {'N-Gram':<6} | {'Freq':<5} | {'Method'}"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)

    for rank, res in enumerate(results, 1):
        method = res.metadata.get("method", "N/A")
        if "supporting_methods" in res.metadata:
            method = f"Consensus ({len(res.metadata['supporting_methods'])} votes)"
        print(f"{rank:<5} | {res.keyword:<32} | {res.score:<8.4f} | {res.ngram_size:<6} | {res.frequency:<5} | {method}")

    print(separator + "\n")


def main():
    parser = argparse.ArgumentParser(
        prog="keyword-extractor",
        description="KeySense NLP: Production-grade Keyword & Keyphrase Extractor"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: extract
    p_extract = subparsers.add_parser("extract", help="Extract keywords from a text string")
    p_extract.add_argument("text", type=str, help="Input text string or '-' to read from stdin")
    p_extract.add_argument("--method", "-m", default="ensemble", choices=list(KeywordExtractorPipeline.AVAILABLE_METHODS))
    p_extract.add_argument("--top-k", "-k", type=int, default=10, help="Number of keywords to extract")
    p_extract.add_argument("--min-n", type=int, default=1, help="Min n-gram length")
    p_extract.add_argument("--max-n", type=int, default=2, help="Max n-gram length")
    p_extract.add_argument("--diversity", "-d", type=float, default=0.5, help="MMR diversity parameter (0.0 to 1.0)")
    p_extract.add_argument("--format", choices=["table", "json"], default="table", help="Output format")

    # Command: file
    p_file = subparsers.add_parser("file", help="Extract keywords from a text file")
    p_file.add_argument("filepath", type=str, help="Path to input text file")
    p_file.add_argument("--method", "-m", default="ensemble", choices=list(KeywordExtractorPipeline.AVAILABLE_METHODS))
    p_file.add_argument("--top-k", "-k", type=int, default=10)
    p_file.add_argument("--min-n", type=int, default=1)
    p_file.add_argument("--max-n", type=int, default=2)
    p_file.add_argument("--output", "-o", type=str, help="Path to save output JSON")
    p_file.add_argument("--format", choices=["table", "json"], default="table")

    # Command: compare
    p_compare = subparsers.add_parser("compare", help="Compare all algorithms side-by-side on a text")
    p_compare.add_argument("text", type=str, help="Input text string")
    p_compare.add_argument("--top-k", "-k", type=int, default=5)

    # Command: benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run benchmark against gold-standard evaluation datasets")
    p_bench.add_argument("--top-k", "-k", type=int, default=10)

    # Command: serve
    p_serve = subparsers.add_parser("serve", help="Start the HTTP REST API and interactive Web UI server")
    p_serve.add_argument("--host", default="0.0.0.0", help="Host interface")
    p_serve.add_argument("--port", "-p", type=int, default=8000, help="Port to listen on")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    pipeline = KeywordExtractorPipeline()

    if args.command == "extract":
        text = args.text
        if text == "-":
            text = sys.stdin.read()

        results = pipeline.extract(
            text=text,
            top_k=args.top_k,
            ngram_range=(args.min_n, args.max_n),
            method=args.method,
            diversity=args.diversity
        )

        if args.format == "json":
            print(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            print_table(results, title=f"Keywords Extracted with '{args.method.upper()}'")

    elif args.command == "file":
        with open(args.filepath, "r", encoding="utf-8") as f:
            text = f.read()

        results = pipeline.extract(
            text=text,
            top_k=args.top_k,
            ngram_range=(args.min_n, args.max_n),
            method=args.method
        )

        if args.output:
            with open(args.output, "w", encoding="utf-8") as out_f:
                json.dump([r.to_dict() for r in results], out_f, indent=2)
            print(f"Results written to: {args.output}")
        else:
            if args.format == "json":
                print(json.dumps([r.to_dict() for r in results], indent=2))
            else:
                print_table(results, title=f"File: {args.filepath} ({args.method.upper()})")

    elif args.command == "compare":
        comp = pipeline.compare(args.text, top_k=args.top_k)
        for method, res in comp.items():
            print_table(res, title=f"Algorithm: {method}")

    elif args.command == "benchmark":
        print("\nRunning comprehensive benchmark against reference datasets...\n")
        bench = run_benchmark(pipeline, top_k=args.top_k)
        header = f"{'Algorithm':<16} | {'Precision@K':<12} | {'Recall@K':<10} | {'F1@K':<8} | {'Soft-F1':<8} | {'MRR':<6}"
        sep = "-" * len(header)
        print(sep)
        print(header)
        print(sep)
        for method, metrics in bench.items():
            print(f"{method:<16} | {metrics['precision@k']:<12.4f} | {metrics['recall@k']:<10.4f} | {metrics['f1@k']:<8.4f} | {metrics['soft_f1@k']:<8.4f} | {metrics['mrr']:<6.4f}")
        print(sep + "\n")

    elif args.command == "serve":
        start_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
