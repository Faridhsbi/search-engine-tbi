"""
search.py

Demonstrates all retrieval methods available in the search engine.
Run this script after indexing (python bsbi.py) to see results.
"""

from bsbi import BSBIIndex
from compression import VBEPostings, EliasGammaPostings
import time
import os

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir='collection',
                          postings_encoding=VBEPostings,
                          output_dir='index')

queries = ["alkylated with radioactive iodoacetate",
           "psychodrama for disturbed children",
           "lipid metabolism in toxemia and normal pregnancy"]

print("=" * 70)
print("  SEARCH ENGINE DEMO - Multiple Retrieval Methods")
print("=" * 70)

for query in queries:
    print(f"\n{'-' * 70}")
    print(f"  Query: {query}")
    print(f"{'-' * 70}")

    # TF-IDF
    print(f"\n  [TF-IDF Results] (top 10):")
    start = time.time()
    results = BSBI_instance.retrieve_tfidf(query, k=10)
    elapsed = time.time() - start
    for i, (score, doc) in enumerate(results, 1):
        print(f"    {i:>2}. {doc:30} score={score:.3f}")
    print(f"    ({elapsed*1000:.1f} ms)")

    # BM25
    print(f"\n  [BM25 Results] (top 10):")
    start = time.time()
    results = BSBI_instance.retrieve_bm25(query, k=10)
    elapsed = time.time() - start
    for i, (score, doc) in enumerate(results, 1):
        print(f"    {i:>2}. {doc:30} score={score:.3f}")
    print(f"    ({elapsed*1000:.1f} ms)")

    # BM25 + WAND
    print(f"\n  [BM25 + WAND Top-K Results] (top 10):")
    start = time.time()
    results = BSBI_instance.retrieve_bm25_wand(query, k=10)
    elapsed = time.time() - start
    for i, (score, doc) in enumerate(results, 1):
        print(f"    {i:>2}. {doc:30} score={score:.3f}")
    print(f"    ({elapsed*1000:.1f} ms)")

print(f"\n{'=' * 70}")

# Adaptive Retrieval (RM3)
try:
    from adaptive import AdaptiveRetrieval
    adaptive = AdaptiveRetrieval(BSBI_instance, fb_docs=10, fb_terms=20, alpha=0.5)
    print(f"\n  [Adaptive (RM3) Results] for first query:")
    start = time.time()
    results = adaptive.retrieve_adaptive(queries[0], k=10)
    elapsed = time.time() - start
    for i, (score, doc) in enumerate(results, 1):
        print(f"    {i:>2}. {doc:30} score={score:.3f}")
    print(f"    ({elapsed*1000:.1f} ms)")
except Exception as e:
    print(f"  [Adaptive retrieval not available: {e}]")

# LSI Retrieval
try:
    from lsi import LSIIndex
    import pickle
    lsi_model_path = os.path.join('index', 'lsi_model.pkl')
    if os.path.exists(lsi_model_path):
        from util import IdMap
        with open(os.path.join('index', 'terms.dict'), 'rb') as f:
            term_id_map = pickle.load(f)
        with open(os.path.join('index', 'docs.dict'), 'rb') as f:
            doc_id_map = pickle.load(f)
        lsi = LSIIndex(n_components=100)
        lsi.load_model(lsi_model_path)
        print(f"\n  [LSI Results] for first query:")
        start = time.time()
        results = lsi.retrieve(queries[0], term_id_map, doc_id_map, k=10)
        elapsed = time.time() - start
        for i, (score, doc) in enumerate(results, 1):
            print(f"    {i:>2}. {doc:30} score={score:.4f}")
        print(f"    ({elapsed*1000:.1f} ms)")
    else:
        print("  [LSI model not found. Run `python lsi.py` to build it.]")
except Exception as e:
    print(f"  [LSI retrieval not available: {e}]")

print(f"\n{'=' * 70}")