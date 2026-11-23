"""
Streamlit Web Application for KeySense NLP Keyword Extractor.
Run with: python3 -m streamlit run web/app.py
"""

import json
import streamlit as st
import pandas as pd

from keyword_extractor.pipeline import KeywordExtractorPipeline
from keyword_extractor.evaluation import run_benchmark, BENCHMARK_DATASET

st.set_page_config(
    page_title="KeySense NLP - Keyword & Keyphrase Extractor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.0rem;
        margin-bottom: 1.5rem;
    }
    .highlight-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.25rem;
        line-height: 1.8;
        font-size: 1.0rem;
    }
    mark.nlp-kw {
        background-color: rgba(56, 189, 248, 0.25);
        color: #e0f2fe;
        border-bottom: 2px solid #38bdf8;
        padding: 2px 5px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Sample Documents
SAMPLE_DOCS = {
    "AI - Attention Is All You Need": """The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU.""",

    "Biotech - CRISPR Gene Editing": """CRISPR-Cas9 is a revolutionary genome editing technology adapted from a bacterial defense system. The system utilizes a single guide RNA molecule that directs the Cas9 endonuclease enzyme to introduce double-strand breaks at specific targeted DNA sequences. Subsequent cellular repair mechanisms, including non-homologous end joining and homology-directed repair, enable targeted gene knockouts, precise sequence insertions, or transcriptional modifications. CRISPR applications have expanded rapidly across genetic disease therapeutics, crop biotechnology, and functional genomics screening.""",

    "Cloud - Kubernetes & Distributed Systems": """Cloud computing architectures provide on-demand access to scalable computing resources, storage infrastructure, and managed databases over the internet with pay-as-you-go pricing. Container orchestration platforms such as Kubernetes automate deployment, scaling, and operational management of microservices across distributed multi-cloud clusters. Serverless functions abstract underlying server provisioning, executing application code strictly in response to event triggers and automatically scaling from zero to thousands of concurrent requests."""
}

# Sidebar Controls
st.sidebar.title("⚙️ Configuration")

selected_method = st.sidebar.selectbox(
    "Algorithm Paradigm",
    options=["ensemble", "positionrank", "textrank", "yake", "rake", "semantic", "tfidf"],
    index=0,
    help="Select the extraction algorithm. 'ensemble' combines all methods using Reciprocal Rank Fusion."
)

top_k = st.sidebar.slider("Top Keywords (K)", min_value=3, max_value=25, value=10)

ngram_option = st.sidebar.selectbox(
    "N-Gram Phrase Length",
    options=["Unigrams & Bigrams (1, 2)", "1 to 3 words (1, 3)", "Unigrams only (1, 1)", "Phrases only (2, 3)"],
    index=0
)
ngram_map = {
    "Unigrams & Bigrams (1, 2)": (1, 2),
    "1 to 3 words (1, 3)": (1, 3),
    "Unigrams only (1, 1)": (1, 1),
    "Phrases only (2, 3)": (2, 3)
}
ngram_range = ngram_map[ngram_option]

diversity = st.sidebar.slider(
    "MMR Diversity Factor",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Higher values favor diverse keywords; lower values maximize pure document relevance."
)

custom_sw_input = st.sidebar.text_input(
    "Custom Stopwords (comma separated)",
    value="",
    placeholder="e.g. paper, study, results"
)
custom_stopwords = [w.strip() for w in custom_sw_input.split(",") if w.strip()]

# Header
st.markdown('<div class="main-title">🔍 KeySense NLP</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">State-of-the-Art Multi-Paradigm Keyword & Keyphrase Extractor</div>', unsafe_allow_html=True)

# Main Input Section
col1, col2 = st.columns([3, 1])
with col2:
    selected_sample = st.selectbox("Load Sample Text", options=list(SAMPLE_DOCS.keys()))
    uploaded_file = st.file_uploader("Or Upload File (.txt, .md)", type=["txt", "md"])

if uploaded_file is not None:
    input_text = uploaded_file.read().decode("utf-8")
else:
    input_text = SAMPLE_DOCS[selected_sample]

text_input = st.text_area("Input Document Text", value=input_text, height=200)

# Pipeline instantiation cached
@st.cache_resource
def get_pipeline():
    return KeywordExtractorPipeline()

pipeline = get_pipeline()
if custom_stopwords:
    pipeline.preprocessor.stopwords.update(custom_stopwords)

# Execution
tabs = st.tabs(["📊 Extracted Keywords", "🖍️ In-Text Highlighter", "⚖️ Algorithm Comparison", "📈 Benchmark & Metrics"])

with tabs[0]:
    if st.button("🚀 Extract Keywords", type="primary", use_container_width=True):
        with st.spinner("Extracting keywords..."):
            results = pipeline.extract(
                text=text_input,
                top_k=top_k,
                ngram_range=ngram_range,
                method=selected_method,
                diversity=diversity
            )

        if results:
            df = pipeline.to_dataframe(results)
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Importance Score",
                        help="Normalized score between 0.0 and 1.0",
                        format="%.3f",
                        min_value=0.0,
                        max_value=1.0,
                    ),
                    "Frequency": st.column_config.NumberColumn("Occurrences"),
                }
            )

            # Download CSV
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Keywords as CSV",
                data=csv_data,
                file_name="extracted_keywords.csv",
                mime="text/csv"
            )
        else:
            st.warning("No keywords found. Try adjusting stopword or n-gram settings.")

with tabs[1]:
    st.subheader("Interactive Document Keyword Highlighting")
    results = pipeline.extract(
        text=text_input,
        top_k=top_k,
        ngram_range=ngram_range,
        method=selected_method,
        diversity=diversity
    )
    if results:
        highlighted = text_input
        # Sort keywords by length descending so sub-words don't clobber full phrases
        kw_sorted = sorted([r.keyword for r in results], key=len, reverse=True)
        import re
        for kw in kw_sorted:
            pattern = re.compile(rf"\b({re.escape(kw)})\b", re.IGNORECASE)
            highlighted = pattern.sub(r'<mark class="nlp-kw">\1</mark>', highlighted)

        st.markdown(f'<div class="highlight-card">{highlighted}</div>', unsafe_allow_html=True)
    else:
        st.info("Extract keywords to view highlights.")

with tabs[2]:
    st.subheader("Side-by-Side Algorithm Comparison")
    if st.button("Run Multi-Algorithm Comparison"):
        with st.spinner("Comparing algorithms..."):
            comp_results = pipeline.compare(text=text_input, top_k=top_k, ngram_range=ngram_range)

            # Build comparison table
            max_rows = top_k
            table_dict = {}
            for method_name, kws in comp_results.items():
                kw_strings = [f"{r.keyword} ({r.score:.2f})" for r in kws]
                while len(kw_strings) < max_rows:
                    kw_strings.append("-")
                table_dict[method_name] = kw_strings[:max_rows]

            comp_df = pd.DataFrame(table_dict)
            st.dataframe(comp_df, use_container_width=True)

with tabs[3]:
    st.subheader("Benchmark Against Gold-Standard Datasets")
    st.write("Evaluates exact match, soft semantic match, and Mean Reciprocal Rank (MRR) against human-annotated ground truth.")
    if st.button("Run Benchmark Evaluation"):
        with st.spinner("Running benchmark suite..."):
            bench = run_benchmark(pipeline, top_k=top_k)
            bench_df = pd.DataFrame(bench).T
            st.dataframe(bench_df, use_container_width=True)

            st.bar_chart(bench_df[["f1@k", "soft_f1@k", "mrr"]])
