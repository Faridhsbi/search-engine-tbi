#  Search Engine "From Scratch"

A full-featured information retrieval search engine built from scratch in Python, implementing core IR concepts including indexing, compression, scoring, evaluation, and advanced retrieval techniques.

---

## Feature Highlights

### Core Features

| No. | Feature | Description |
|---|---------|-------------|
| 1 | **Elias Gamma Compression** | Bit-level compression algorithm alongside existing VBE, achieving superior compression ratios for small integers |
| 2 | **BM25 Scoring** | Okapi BM25 ranking (k1=1.2, b=0.75) with pre-computed avg doc length for accurate length normalization |
| 3 | **DCG, NDCG, AP Metrics** | Three additional evaluation metrics beyond RBP: Discounted Cumulative Gain, Normalized DCG, and Average Precision |
| 4 | **WAND Top-K Retrieval** | Weighted AND algorithm that prunes documents whose upper-bound BM25 scores can't enter the top-K heap, avoiding unnecessary scoring |

### Other Features

| No. | Feature | Description |
|---|---------|-------------|
| 1 | **SPIMI Indexing** | Single-Pass In-Memory Indexing, alternative to BSBI that builds in-memory inverted index per block without requiring a global term-to-termID mapping during the initial pass |
| 2 | **Trie Dictionary** | Prefix tree data structure for the term dictionary, enabling prefix search (autocomplete) and wildcard queries with `*` and `?` patterns |
| 3 | **LSI with FAISS** | Latent Semantic Indexing via Truncated SVD, with optional FAISS vector indexing for fast cosine similarity search in the reduced-dimensional semantic space |
| 4 | **Adaptive Retrieval (RM3)** | Pseudo-Relevance Feedback using RM3, first-pass BM25 retrieval identifies "pseudo-relevant" documents, from which expansion terms are extracted and interpolated with the original query |
| 5 | **Text Preprocessing** | Custom Porter Stemmer (implemented from scratch), regex tokenization, and English stopword removal, applied consistently to both documents and queries |
| 6 | **Interactive CLI** | Menu-driven command-line interface with support for all retrieval methods, Trie-based autocomplete, evaluation, and document viewing |

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd TP2

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# Step 1: Build the index (BSBI with VBE compression)
python bsbi.py

# Step 2: Run search demo
python search.py

# Step 3: Run full evaluation (RBP, DCG, NDCG, AP)
python evaluation.py

# Step 4: Launch interactive CLI
python interactive_cli.py
```

---

## Detailed Usage

### 1. Indexing

#### BSBI Indexing (default)
```bash
python bsbi.py
```

#### SPIMI Indexing (alternative)
```bash
python spimi.py
```

### 2. Search

```bash
# Demo mode --> shows all retrieval methods
python search.py

# Interactive mode --> choose methods, search interactively
python interactive_cli.py
```

### 3. Evaluation

```bash
# Evaluates TF-IDF, BM25, and BM25+WAND using RBP, DCG, NDCG, AP
python evaluation.py
```

### 4. Compression Test

```bash
# Test all three compression methods (Standard, VBE, Elias Gamma)
python compression.py
```

### 5. Build LSI Model

```bash
# Build LSI model from the existing index (requires numpy)
python lsi.py
```

### 6. Trie Test

```bash
# Test Trie data structure
python trie.py
```

---

## Evaluation Metrics

The search engine is evaluated on 30 queries from `queries.txt` with relevance judgments from `qrels.txt`.

| Metric | Formula | Range | Description |
|--------|---------|-------|-------------|
| **RBP** | `(1-p) * Σ rel_i * p^(i-1)` | [0, 1] | Rank-Biased Precision, user model with geometric persistence |
| **DCG** | `Σ rel_i / log2(i+1)` | [0, ∞) | Discounted Cumulative Gain, graded relevance with position discount |
| **NDCG** | `DCG / IDCG` | [0, 1] | Normalized DCG, DCG divided by ideal DCG |
| **AP** | `(1/R) * Σ P(k) * rel(k)` | [0, 1] | Average Precision, mean precision at each relevant doc |

### Evaluation Comparison Results

Based on a test of 30 queries using the CISI collection:

| Metric | TF-IDF | BM25 | BM25+WAND |
|--------|--------|------|-----------|
| **RBP** | 0.6454 | **0.6691** | **0.6691** |
| **DCG** | 5.7773 | **5.8878** | **5.8878** |
| **NDCG** | 0.8208 | **0.8302** | **0.8302** |
| **AP** | 0.5595 | **0.5788** | **0.5788** |

*Note: As expected, BM25 outperforms TF-IDF across all metrics. WAND optimization produces identical results to standard BM25, verifying its correctness while providing faster top-K retrieval.*

---

##  Technical Details

### Compression Methods

| Method | Type | Approach |
|--------|------|----------|
| **StandardPostings** | Byte-level | Raw 4-byte unsigned integers |
| **VBEPostings** | Byte-level | Gap-based Variable-Byte Encoding |
| **EliasGammaPostings** | **Bit-level** | Gap-based Elias Gamma Encoding, encodes integer n using floor(log2(n)) zeros + binary representation |

### Scoring Functions

**TF-IDF:**
```
w(t,D) = (1 + log tf(t,D)) * log(N / df(t))
```

**BM25:**
```
BM25(D,Q) = Σ IDF(t) * [tf(t,D) * (k1+1)] / [tf(t,D) + k1 * (1 - b + b * |D|/avgdl)]
IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
```

### WAND Algorithm

The WAND (Weighted AND) optimization skips documents that cannot enter the top-K results:
1. For each query term, compute an upper-bound BM25 score using max TF
2. For each candidate document, check if the sum of upper bounds exceeds the current top-K threshold
3. Only fully score documents that pass this check
4. This reduces the number of BM25 computations significantly

### LSI Pipeline

1. **Build TF-IDF matrix** A (terms × docs) from inverted index
2. **Truncated SVD**: A ≈ U_k Σ_k V_k^T
3. **Document vectors**: rows of V_k Σ_k (semantic embeddings)
4. **Query projection**: q_lsi = q^T U_k Σ_k^{-1}
5. **FAISS search**: cosine similarity via IndexFlatIP

### RM3 Pseudo-Relevance Feedback

1. First-pass BM25 retrieval → top-K "pseudo-relevant" documents
2. Extract term distribution from pseudo-relevant docs: P(t|R)
3. Interpolate: P_RM3(t) = α · P_orig(t) + (1-α) · P_RM1(t)
4. Re-rank with expanded weighted query

---

## Dependencies

| Package | Purpose | Required? |
|---------|---------|-----------|
| `tqdm` | Progress bars during indexing | Yes |
| `numpy` | SVD computation for LSI | For LSI only |
| `faiss-cpu` | Fast vector similarity search | For LSI only (optional, falls back to brute-force) |

All core features (indexing, retrieval, evaluation) work with **only `tqdm`**. The Porter Stemmer and stopword list are implemented from scratch, no NLP library needed.

