"""
interactive_cli.py

Interactive command-line interface for the search engine.
Allows users to:
  - Choose retrieval method (TF-IDF, BM25, WAND, LSI, RM3 Adaptive)
  - Choose compression method (Standard, VBE, Elias Gamma)
  - Search queries interactively
  - View document snippets
  - Prefix search / autocomplete (using Trie)
  - Run evaluation
  - Re-index the collection

Usage:
    python interactive_cli.py
"""

import os
import sys
import time
import pickle
import re

from bsbi import BSBIIndex
from compression import StandardPostings, VBEPostings, EliasGammaPostings
from preprocessing import preprocess


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    print()
    print("+==================================================================+")
    print("|           SEARCH ENGINE -- Interactive CLI                       |")
    print("+==================================================================+")
    print()


def print_menu():
    print("+--------------------------------------------------+")
    print("|  COMMANDS:                                       |")
    print("|    [1] Search (enter query)                      |")
    print("|    [2] Change retrieval method                   |")
    print("|    [3] Change compression & re-index             |")
    print("|    [4] Prefix search / Autocomplete              |")
    print("|    [5] Run full evaluation                       |")
    print("|    [6] Build LSI model                           |")
    print("|    [7] Index with SPIMI                          |")
    print("|    [8] View document                             |")
    print("|    [q] Quit                                      |")
    print("+--------------------------------------------------+")


def choose_retrieval_method():
    print("\n  Available retrieval methods:")
    print("    [1] TF-IDF (log TF x IDF)")
    print("    [2] BM25 (Okapi BM25)")
    print("    [3] BM25 + WAND (Top-K optimized)")
    print("    [4] Adaptive (RM3 Pseudo-Relevance Feedback)")
    print("    [5] LSI (Latent Semantic Indexing)")
    choice = input("  Select [1-5]: ").strip()
    methods = {'1': 'tfidf', '2': 'bm25', '3': 'wand', '4': 'adaptive', '5': 'lsi'}
    return methods.get(choice, 'bm25')


def choose_compression():
    print("\n  Available compression methods:")
    print("    [1] Standard (no compression)")
    print("    [2] Variable-Byte Encoding (VBE)")
    print("    [3] Elias Gamma (bit-level)")
    choice = input("  Select [1-3]: ").strip()
    encodings = {
        '1': ('Standard', StandardPostings),
        '2': ('VBE', VBEPostings),
        '3': ('Elias Gamma', EliasGammaPostings),
    }
    return encodings.get(choice, ('VBE', VBEPostings))


def search_query(bsbi, query, method, k=10):
    """Execute a search with the given method."""
    start = time.time()

    if method == 'tfidf':
        results = bsbi.retrieve_tfidf(query, k=k)
        method_name = "TF-IDF"
    elif method == 'bm25':
        results = bsbi.retrieve_bm25(query, k=k)
        method_name = "BM25"
    elif method == 'wand':
        results = bsbi.retrieve_bm25_wand(query, k=k)
        method_name = "BM25 + WAND"
    elif method == 'adaptive':
        try:
            from adaptive import AdaptiveRetrieval
            adaptive = AdaptiveRetrieval(bsbi, fb_docs=10, fb_terms=20, alpha=0.5)
            results = adaptive.retrieve_adaptive(query, k=k)
            method_name = "Adaptive (RM3)"
        except Exception as e:
            print(f"  Error: {e}")
            return
    elif method == 'lsi':
        try:
            from lsi import LSIIndex
            lsi_model_path = os.path.join('index', 'lsi_model.pkl')
            if not os.path.exists(lsi_model_path):
                print("  LSI model not found. Please build it first (option 6).")
                return
            with open(os.path.join('index', 'terms.dict'), 'rb') as f:
                term_id_map = pickle.load(f)
            with open(os.path.join('index', 'docs.dict'), 'rb') as f:
                doc_id_map = pickle.load(f)
            lsi = LSIIndex(n_components=100)
            lsi.load_model(lsi_model_path)
            results = lsi.retrieve(query, term_id_map, doc_id_map, k=k)
            method_name = "LSI"
        except Exception as e:
            print(f"  Error: {e}")
            return
    else:
        results = bsbi.retrieve_bm25(query, k=k)
        method_name = "BM25"

    elapsed = time.time() - start

    print(f"\n  [{method_name} Results] ({len(results)} hits, {elapsed*1000:.1f} ms)")
    for i, (score, doc) in enumerate(results, 1):
        # Extract doc ID for display
        match = re.search(r'\/(\d+)\.txt', doc)
        doc_id = match.group(1) if match else "?"
        print(f"    {i:>3}. [Doc {doc_id:>4}]  score={score:>8.4f}  {doc}")
    if not results:
        print("    No results found.")
    print()


def view_document(data_dir='collection'):
    """View the contents of a document."""
    doc_id = input("  Enter document ID (e.g., 42): ").strip()
    try:
        doc_num = int(doc_id)
    except ValueError:
        print("  Invalid document ID.")
        return

    # Search for the document across blocks
    for block in sorted(next(os.walk(data_dir))[1]):
        doc_path = os.path.join(data_dir, block, f"{doc_num}.txt")
        if os.path.exists(doc_path):
            with open(doc_path, 'r', encoding='utf8', errors='surrogateescape') as f:
                content = f.read()
            print(f"\n  --- Document {doc_num} ({doc_path}) ---")
            for line in content.strip().split('\n'):
                print(f"  | {line.rstrip()}")
            print(f"  ---")
            return

    print(f"  Document {doc_id} not found.")


def prefix_search_interactive(bsbi):
    """Interactive prefix search using Trie."""
    try:
        from trie import Trie
        trie_path = os.path.join('index', 'trie.pkl')
        if os.path.exists(trie_path):
            trie = Trie.load(trie_path)
        else:
            # Build trie from term_id_map
            print("  Building Trie from term dictionary...")
            if len(bsbi.term_id_map) == 0:
                bsbi.load()
            trie = Trie()
            for term_str, term_id in bsbi.term_id_map.str_to_id.items():
                trie.insert(term_str, term_id)
            trie.save(trie_path)
            print(f"  Trie built with {len(trie)} terms, saved to {trie_path}")

        prefix = input("  Enter prefix: ").strip().lower()
        results = trie.prefix_search(prefix)
        print(f"\n  Prefix '{prefix}' -> {len(results)} matches:")
        for term, tid in results[:30]:
            print(f"    {term} (ID: {tid})")
        if len(results) > 30:
            print(f"    ... and {len(results) - 30} more")

    except Exception as e:
        print(f"  Error: {e}")


def run_evaluation(bsbi):
    """Run the full evaluation suite."""
    try:
        from evaluation import eval, load_qrels
        qrels = load_qrels()
        eval(qrels)
    except Exception as e:
        print(f"  Error: {e}")


def build_lsi_model():
    """Build an LSI model from the current index."""
    try:
        from lsi import LSIIndex
        with open(os.path.join('index', 'terms.dict'), 'rb') as f:
            term_id_map = pickle.load(f)
        with open(os.path.join('index', 'docs.dict'), 'rb') as f:
            doc_id_map = pickle.load(f)

        n_comp = input("  Number of LSI dimensions (default 100): ").strip()
        n_comp = int(n_comp) if n_comp else 100

        lsi = LSIIndex(n_components=n_comp)
        lsi.build_from_inverted_index(
            index_name='main_index',
            postings_encoding=VBEPostings,
            directory='index',
            term_id_map=term_id_map,
            doc_id_map=doc_id_map
        )
        lsi.save(os.path.join('index', 'lsi_model.pkl'))
    except Exception as e:
        print(f"  Error: {e}")


def run_spimi_indexing():
    """Run SPIMI indexing."""
    try:
        from spimi import SPIMIIndex
        enc_name, enc = choose_compression()
        spimi = SPIMIIndex(data_dir='collection',
                          postings_encoding=enc,
                          output_dir='index',
                          block_size=100)
        spimi.index()
        print("  SPIMI indexing complete!")
    except Exception as e:
        print(f"  Error: {e}")


def main():
    current_compression_name = "VBE"
    current_compression = VBEPostings
    current_method = "bm25"

    bsbi = BSBIIndex(data_dir='collection',
                     postings_encoding=current_compression,
                     output_dir='index')

    # Check if index exists
    index_path = os.path.join('index', 'main_index.index')
    if not os.path.exists(index_path):
        print("  No index found. Building index first...")
        bsbi.index()
        print("  Indexing complete!")

    while True:
        print_header()
        print(f"  Current method: {current_method.upper()} | Compression: {current_compression_name}")
        print_menu()

        choice = input("\n  > ").strip().lower()

        if choice == '1':
            query = input("  Enter query: ").strip()
            if query:
                k = input("  Number of results (default 10): ").strip()
                k = int(k) if k else 10
                search_query(bsbi, query, current_method, k=k)
            input("\n  Press Enter to continue...")

        elif choice == '2':
            current_method = choose_retrieval_method()
            print(f"  Method set to: {current_method.upper()}")
            input("  Press Enter to continue...")

        elif choice == '3':
            current_compression_name, current_compression = choose_compression()
            print(f"  Compression set to: {current_compression_name}")
            print("  Re-indexing...")
            bsbi = BSBIIndex(data_dir='collection',
                             postings_encoding=current_compression,
                             output_dir='index')
            bsbi.index()
            print("  Re-indexing complete!")
            input("  Press Enter to continue...")

        elif choice == '4':
            prefix_search_interactive(bsbi)
            input("\n  Press Enter to continue...")

        elif choice == '5':
            run_evaluation(bsbi)
            input("\n  Press Enter to continue...")

        elif choice == '6':
            build_lsi_model()
            input("\n  Press Enter to continue...")

        elif choice == '7':
            run_spimi_indexing()
            input("\n  Press Enter to continue...")

        elif choice == '8':
            view_document()
            input("\n  Press Enter to continue...")

        elif choice == 'q':
            print("  Thank You!")
            break

        else:
            print("  Invalid option. Try again.")
            input("  Press Enter to continue...")


if __name__ == "__main__":
    main()
