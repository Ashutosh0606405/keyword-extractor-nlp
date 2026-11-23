"""
Benchmark datasets and reference texts with ground-truth keywords for evaluation.
"""

from dataclasses import dataclass
from typing import List, Set


@dataclass
class BenchmarkDocument:
    """A benchmark document containing raw text and human-annotated gold keywords."""
    id: str
    title: str
    domain: str
    text: str
    gold_keywords: List[str]


BENCHMARK_DATASET: List[BenchmarkDocument] = [
    BenchmarkDocument(
        id="cs_attention_01",
        title="Attention Is All You Need Abstract",
        domain="Artificial Intelligence",
        text="""
        The dominant sequence transduction models are based on complex recurrent or convolutional neural networks
        in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through
        an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on
        attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation
        tasks show these models to be superior in quality while being more parallelizable and requiring significantly less
        time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over
        the existing best results, including ensembles, by over 2 BLEU.
        """,
        gold_keywords=[
            "transformer",
            "attention mechanism",
            "neural networks",
            "machine translation",
            "sequence transduction",
            "encoder-decoder",
            "convolutions",
            "recurrence",
            "parallelizable",
            "bleu"
        ]
    ),
    BenchmarkDocument(
        id="nlp_bert_02",
        title="BERT Pre-training Abstract",
        domain="Natural Language Processing",
        text="""
        We introduce a new language representation model called BERT, which stands for Bidirectional Encoder
        Representations from Transformers. Unlike recent language representation models, BERT is designed to
        pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and
        right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional
        output layer to create state-of-the-art models for a wide range of tasks, such as question answering and
        language inference, without substantial task-specific architecture modifications.
        """,
        gold_keywords=[
            "bert",
            "language representation",
            "transformers",
            "bidirectional representations",
            "pre-train",
            "fine-tuned",
            "question answering",
            "language inference"
        ]
    ),
    BenchmarkDocument(
        id="med_crispr_03",
        title="CRISPR Gene Editing Overview",
        domain="Biotechnology & Medicine",
        text="""
        CRISPR-Cas9 is a revolutionary genome editing technology adapted from a bacterial defense system.
        The system utilizes a single guide RNA molecule that directs the Cas9 endonuclease enzyme to introduce
        double-strand breaks at specific targeted DNA sequences. Subsequent cellular repair mechanisms, including
        non-homologous end joining and homology-directed repair, enable targeted gene knockouts, precise sequence insertions,
        or transcriptional modifications. CRISPR applications have expanded rapidly across genetic disease therapeutics,
        crop biotechnology, and functional genomics screening.
        """,
        gold_keywords=[
            "crispr-cas9",
            "genome editing",
            "guide rna",
            "endonuclease",
            "double-strand breaks",
            "dna sequences",
            "gene knockouts",
            "functional genomics",
            "biotechnology"
        ]
    )
]
