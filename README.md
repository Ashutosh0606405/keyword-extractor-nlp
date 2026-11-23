# 🔍 KeySense NLP - Production-Grade Keyword & Keyphrase Extractor

KeySense NLP is an enterprise-grade, multi-paradigm NLP library, REST API, CLI, and interactive Web Application for extracting highly accurate keywords and keyphrases from unstructured text.

---

## 🌟 Key Highlights

- **Multi-Paradigm Engine**: Implements statistical, graph-based, and semantic keyword extraction algorithms.
- **Ensemble Hybrid Fusion**: Combines multi-model outputs via **Reciprocal Rank Fusion (RRF)** to produce consensus keywords with high precision and recall.
- **Zero Heavy Setup Required**: 100% offline-ready with pure Python and standard scientific packages (`scikit-learn`, `numpy`, `networkx`, `nltk`).
- **Interactive UI & Visualizer**: Includes both a **Streamlit Web Application** with live in-text keyword highlighting and a lightweight **Single-Page HTML Web Application** served directly by the built-in HTTP server.
- **Full REST API**: Ready-to-deploy endpoints for single document, batch extraction, algorithm comparisons, and benchmarks.
- **Rich CLI**: Extract from strings, stream from stdin, process files, and generate formatted tables or JSON output.
- **Rigorous Evaluation Suite**: Built-in benchmark suite measuring Precision@K, Recall@K, F1@K, Soft-F1, and Mean Reciprocal Rank (MRR) against human-annotated ground truth.

---

## 📐 Architecture & Supported Algorithms

| Algorithm | Paradigm | Description | Best Suited For |
| :--- | :--- | :--- | :--- |
| **`Ensemble`** *(Default)* | **Hybrid Fusion** | Merges statistical, graph, and semantic extractors with Reciprocal Rank Fusion (RRF) | General-purpose, highest overall accuracy |
| **`PositionRank`** | **Graph Centrality** | PageRank biased by positional recency ($1 / \text{pos}$) | Research abstracts, articles with strong titles |
| **`TextRank`** | **Graph Centrality** | Co-occurrence network centrality with POS noun/adjective filtering | News, essays, interconnected documents |
| **`YAKE`** | **Statistical** | Features: casing, position, frequency, dispersion, context | Single documents, technical & domain jargon |
| **`RAKE`** | **Co-occurrence** | Stopword-delimited candidate degree-to-frequency ratios | Multi-word phrases and domain keywords |
| **`SemanticMMR`** | **Vector Similarity** | Embedding cosine similarity + Maximal Marginal Relevance (MMR) | High semantic relevance with diverse coverage |
| **`TF-IDF`** | **Statistical** | Sublinear term frequency with local pseudo-corpus or true corpus IDF | Large collections and fast baseline extraction |

---

## 🚀 Quickstart

### 1. Installation

From the project root:

```bash
cd /Users/ashutoshsoni/.gemini/antigravity/scratch/keyword-extractor-nlp
pip install -e .
```

Or install dependencies directly:
```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Showcase

```bash
python3 demo.py
```

### 3. Launch the Interactive Web Application

**Option A: Streamlit UI (Rich Data Science View)**
```bash
python3 -m streamlit run web/app.py
```

**Option B: Native HTTP REST API & Web UI (Port 8000)**
```bash
python3 -m keyword_extractor.server --port 8000
```
Then open `http://localhost:8000` in your web browser.

---

## 🐍 Python Library Usage

### Basic Extraction (Ensemble Mode)

```python
from keyword_extractor import KeywordExtractorPipeline

pipeline = KeywordExtractorPipeline()

text = """
Attention mechanisms and Transformer architectures have revolutionized
artificial intelligence and modern natural language processing.
"""

results = pipeline.extract(text, top_k=5)

for r in results:
    print(f"Keyword: {r.keyword:<25} | Score: {r.score:.4f} | Freq: {r.frequency}")
```

### Selecting an Algorithm & N-Gram Length

```python
# Extract up to trigram phrases using PositionRank
results = pipeline.extract(
    text,
    method="positionrank",
    top_k=5,
    ngram_range=(1, 3)
)
```

### Semantic Extraction with MMR Diversity Control

```python
# Lower diversity (0.2) = higher focus on primary theme
# Higher diversity (0.8) = broader topical coverage across sub-themes
results = pipeline.extract(
    text,
    method="semantic",
    top_k=6,
    diversity=0.6
)
```

### Side-by-Side Algorithm Comparison

```python
comparison = pipeline.compare(text, top_k=5)
for method_name, kws in comparison.items():
    print(f"[{method_name}]")
    print([k.keyword for k in kws])
```

### Batch Extraction to Pandas DataFrame

```python
documents = [
    "Deep neural networks identify hierarchical image features.",
    "Quantum processors utilize superposition and entanglement for computation."
]

batch_results = pipeline.extract_batch(documents, top_k=3)
df = pipeline.to_dataframe(batch_results[0])
print(df)
```

---

## 💻 Command-Line Interface (CLI)

The CLI supports extracting from raw strings, piping through stdin, processing text files, running comparisons, benchmarks, and starting the server.

```bash
# 1. Quick extraction
python3 -m keyword_extractor.cli extract "Natural language understanding and machine learning" --method ensemble --top-k 5

# 2. Output as structured JSON
python3 -m keyword_extractor.cli extract "Kubernetes automates container orchestration" --format json

# 3. Process an entire file
python3 -m keyword_extractor.cli file sample.txt --method yake --output results.json

# 4. Compare all models on a document
python3 -m keyword_extractor.cli compare "CRISPR-Cas9 enables targeted genome editing" --top-k 4

# 5. Run the evaluation benchmark suite
python3 -m keyword_extractor.cli benchmark

# 6. Start the API & Web server
python3 -m keyword_extractor.cli serve --port 8000
```

---

## 🌐 REST API Endpoints

When running `python3 -m keyword_extractor.server --port 8000`:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive Web Application |
| `GET` | `/api/health` | Healthcheck (`{"status": "ok"}`) |
| `GET` | `/api/methods` | Lists available extraction algorithms |
| `POST` | `/api/extract` | Extracts keywords for a single document |
| `POST` | `/api/compare` | Compares all algorithms on the given text |
| `POST` | `/api/batch` | Extracts keywords for a list of documents |
| `GET` | `/api/benchmark` | Runs automated evaluation on reference benchmarks |

### Example Request (`POST /api/extract`):

```json
{
  "text": "Deep reinforcement learning agents master complex decision-making tasks.",
  "method": "ensemble",
  "top_k": 5,
  "ngram_range": [1, 2],
  "diversity": 0.5
}
```

### Example Response:

```json
{
  "method": "ensemble",
  "count": 5,
  "keywords": [
    {
      "keyword": "decision-making tasks",
      "score": 1.0,
      "ngram_size": 2,
      "frequency": 1,
      "positions": [47],
      "metadata": {
        "method": "EnsembleHybrid",
        "vote_count": 4,
        "supporting_methods": ["TF-IDF", "RAKE", "PositionRank", "YAKE"]
      }
    }
  ]
}
```

---

## 🧪 Testing

Run the automated test suite:

```bash
python3 -m unittest discover tests
```

All 24 unit and integration tests cover:
- Text cleaning, POS tagging, and n-gram candidate generation
- Individual extractors (TF-IDF, RAKE, TextRank, PositionRank, YAKE, SemanticMMR)
- Ensemble Reciprocal Rank Fusion
- Pipeline batch and formatting
- Benchmark evaluations and soft matching
- REST API endpoint simulation and response verification

---

## 📁 Project Structure

```
keyword-extractor-nlp/
├── README.md                           # Documentation and guides
├── pyproject.toml                      # Packaging configuration
├── requirements.txt                    # Python dependencies
├── demo.py                             # Interactive runnable demo
├── keyword_extractor/                  # Core package
│   ├── __init__.py                     # Package export
│   ├── base.py                         # BaseKeywordExtractor & KeywordResult dataclass
│   ├── preprocessor.py                 # Text normalization, POS tagging & candidate generation
│   ├── algorithms/                     # Keyword extraction algorithms
│   │   ├── __init__.py
│   │   ├── tfidf.py                    # TF-IDF extractor
│   │   ├── rake.py                     # Rapid Automatic Keyword Extraction
│   │   ├── textrank.py                 # TextRank & PositionRank (PageRank)
│   │   ├── yake.py                     # YAKE statistical feature extractor
│   │   ├── semantic.py                 # KeyBERT-style Semantic MMR extractor
│   │   └── ensemble.py                 # Reciprocal Rank Fusion (RRF) Ensemble
│   ├── pipeline.py                     # Unified KeywordExtractorPipeline facade
│   ├── evaluation.py                   # Precision@K, Recall@K, F1@K, MRR metrics
│   ├── datasets.py                     # Gold-standard benchmark evaluation sets
│   ├── server.py                       # REST API & Web UI server
│   └── cli.py                          # Command-Line Interface tool
├── web/                                # Web interfaces
│   ├── app.py                          # Streamlit interactive application
│   └── static/
│       └── index.html                  # Lightweight client-side UI
└── tests/                              # Unit & integration tests
    ├── __init__.py
    ├── test_preprocessor.py
    ├── test_extractors.py
    ├── test_pipeline.py
    ├── test_evaluation.py
    └── test_server.py
```
