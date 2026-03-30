"""
lsi.py

Latent Semantic Indexing (LSI) / Latent Semantic Analysis (LSA)
with optional FAISS vector indexing for efficient similarity search.

LSI reduces the high-dimensional term-document space to a lower-dimensional
"latent semantic" space using Truncated SVD (Singular Value Decomposition).
This captures latent relationships between terms and documents, allowing
retrieval of semantically related documents even when they don't share
exact query terms.

Pipeline:
  1. Build a TF-IDF weighted term-document matrix from the inverted index
  2. Apply Truncated SVD:  A ≈ U_k * Σ_k * V_k^T
  3. Document vectors = rows of (Σ_k * V_k^T)
  4. Query vector = project query into LSI space via q_lsi = q^T * U_k * Σ_k^{-1}
  5. Search using cosine similarity (or FAISS for efficiency)

Usage:
    python lsi.py
"""

import os
import pickle
import math
import time

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("WARNING: numpy not installed. LSI will not be available.")

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from index import InvertedIndexReader
from util import IdMap
from compression import VBEPostings
from preprocessing import preprocess


class LSIIndex:
    """
    Latent Semantic Indexing engine.

    Builds a low-dimensional vector representation of documents
    and supports semantic retrieval via cosine similarity.

    Attributes
    ----------
    n_components : int
        Number of SVD dimensions (latent topics)
    doc_vectors : np.ndarray
        Document vectors in the LSI space (N_docs x n_components)
    U_k : np.ndarray
        Truncated left singular vectors (N_terms x n_components)
    sigma_inv : np.ndarray
        Inverse of singular values (for query projection)
    term_to_idx : dict
        Mapping from term_id to row index in the term-document matrix
    doc_to_idx : dict
        Mapping from doc_id to column index in the term-document matrix
    idx_to_doc : dict
        Reverse mapping from column index to doc_id
    faiss_index : faiss.Index or None
        FAISS index for fast nearest-neighbor search
    """

    def __init__(self, n_components=100):
        if not HAS_NUMPY:
            raise ImportError("numpy is required for LSI. Install with: pip install numpy")
        self.n_components = n_components
        self.doc_vectors = None
        self.U_k = None
        self.sigma_inv = None
        self.term_to_idx = {}
        self.doc_to_idx = {}
        self.idx_to_doc = {}
        self.faiss_index = None

    def build_from_inverted_index(self, index_name, postings_encoding,
                                   directory, term_id_map, doc_id_map):
        """
        Build the LSI model from an existing inverted index.

        Steps:
          1. Read the inverted index to build a TF-IDF term-document matrix
          2. Apply truncated SVD
          3. Compute document vectors
          4. Optionally build a FAISS index

        Parameters
        ----------
        index_name : str
        postings_encoding : class
        directory : str
        term_id_map : IdMap
        doc_id_map : IdMap
        """
        print("[LSI] Building term-document matrix from inverted index...")

        with InvertedIndexReader(index_name, postings_encoding,
                                 directory=directory) as reader:
            N = len(reader.doc_length)

            # Build mappings
            all_term_ids = list(reader.postings_dict.keys())
            all_doc_ids = list(reader.doc_length.keys())

            self.term_to_idx = {tid: i for i, tid in enumerate(all_term_ids)}
            self.doc_to_idx = {did: j for j, did in enumerate(all_doc_ids)}
            self.idx_to_doc = {j: did for did, j in self.doc_to_idx.items()}

            n_terms = len(all_term_ids)
            n_docs = len(all_doc_ids)

            print(f"[LSI] Matrix dimensions: {n_terms} terms x {n_docs} docs")

            # Build sparse TF-IDF matrix (dense for small collections)
            # For the CISI dataset (~1033 docs), this fits in memory
            A = np.zeros((n_terms, n_docs), dtype=np.float32)

            reader.reset()
            for term_id, postings, tf_list in reader:
                if term_id not in self.term_to_idx:
                    continue
                row = self.term_to_idx[term_id]
                df = len(postings)
                idf = math.log(N / df) if df > 0 else 0

                for i in range(len(postings)):
                    doc_id = postings[i]
                    tf = tf_list[i]
                    if doc_id in self.doc_to_idx:
                        col = self.doc_to_idx[doc_id]
                        # TF-IDF weight
                        w = (1 + math.log(tf)) * idf if tf > 0 else 0
                        A[row, col] = w

        # Truncated SVD
        k = min(self.n_components, min(n_terms, n_docs) - 1)
        self.n_components = k
        print(f"[LSI] Computing truncated SVD with k={k}...")

        start = time.time()
        U, sigma, Vt = np.linalg.svd(A, full_matrices=False)
        elapsed = time.time() - start
        print(f"[LSI] SVD completed in {elapsed:.2f}s")

        # Truncate to k components
        self.U_k = U[:, :k]                     # (n_terms, k)
        sigma_k = sigma[:k]                      # (k,)
        Vt_k = Vt[:k, :]                         # (k, n_docs)

        # Document vectors: columns of Σ_k * V_k^T → rows of V_k^T^T * Σ_k = V_k * Σ_k
        # Each document vector is (k,) dimensional
        self.doc_vectors = (Vt_k.T * sigma_k).astype(np.float32)  # (n_docs, k)

        # For query projection: q_lsi = q^T * U_k * Σ_k^{-1}
        self.sigma_inv = np.diag(1.0 / sigma_k).astype(np.float32)  # (k, k)

        # Normalize document vectors for cosine similarity
        norms = np.linalg.norm(self.doc_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.doc_vectors_normalized = (self.doc_vectors / norms).astype(np.float32)

        # Build FAISS index if available
        if HAS_FAISS:
            print("[LSI] Building FAISS index for fast retrieval...")
            self.faiss_index = faiss.IndexFlatIP(k)  # Inner Product = cosine on normalized vectors
            self.faiss_index.add(self.doc_vectors_normalized)
            print(f"[LSI] FAISS index built with {self.faiss_index.ntotal} vectors")
        else:
            self.faiss_index = None
            print("[LSI] FAISS not available, using brute-force cosine similarity")

        print(f"[LSI] Build complete. {n_docs} documents in {k}-dimensional space.")

    def project_query(self, query_text, term_id_map):
        """
        Project a query into the LSI space.

        q_lsi = q^T * U_k * Σ_k^{-1}

        Parameters
        ----------
        query_text : str
        term_id_map : IdMap

        Returns
        -------
        np.ndarray
            Query vector in LSI space (k,)
        """
        tokens = preprocess(query_text)
        q_vec = np.zeros(len(self.term_to_idx), dtype=np.float32)

        for token in tokens:
            if token in term_id_map.str_to_id:
                term_id = term_id_map[token]
                if term_id in self.term_to_idx:
                    idx = self.term_to_idx[term_id]
                    q_vec[idx] += 1.0  # TF in query

        # Project: q_lsi = q^T * U_k * Σ_k^{-1}
        q_lsi = q_vec @ self.U_k @ self.sigma_inv  # (k,)
        return q_lsi

    def retrieve(self, query_text, term_id_map, doc_id_map, k=10):
        """
        Retrieve top-K documents using LSI cosine similarity.

        Parameters
        ----------
        query_text : str
        term_id_map : IdMap
        doc_id_map : IdMap
        k : int

        Returns
        -------
        List[Tuple[float, str]]
            List of (score, doc_name) tuples
        """
        q_lsi = self.project_query(query_text, term_id_map)

        # Normalize query vector
        q_norm = np.linalg.norm(q_lsi)
        if q_norm == 0:
            return []
        q_normalized = (q_lsi / q_norm).astype(np.float32).reshape(1, -1)

        if self.faiss_index is not None:
            # Use FAISS for fast search
            scores, indices = self.faiss_index.search(q_normalized, k)
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if idx < 0:
                    continue
                score = float(scores[0][i])
                doc_id = self.idx_to_doc[idx]
                doc_name = doc_id_map[doc_id]
                results.append((score, doc_name))
            return results
        else:
            # Brute-force cosine similarity
            similarities = self.doc_vectors_normalized @ q_normalized.T  # (n_docs, 1)
            similarities = similarities.flatten()

            # Get top-K indices
            top_indices = np.argsort(similarities)[::-1][:k]
            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                if score <= 0:
                    continue
                doc_id = self.idx_to_doc[idx]
                doc_name = doc_id_map[doc_id]
                results.append((score, doc_name))
            return results

    def save(self, filepath):
        """Save the LSI model to disk."""
        data = {
            'n_components': self.n_components,
            'doc_vectors': self.doc_vectors,
            'doc_vectors_normalized': self.doc_vectors_normalized,
            'U_k': self.U_k,
            'sigma_inv': self.sigma_inv,
            'term_to_idx': self.term_to_idx,
            'doc_to_idx': self.doc_to_idx,
            'idx_to_doc': self.idx_to_doc,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"[LSI] Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a previously saved LSI model."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.n_components = data['n_components']
        self.doc_vectors = data['doc_vectors']
        self.doc_vectors_normalized = data['doc_vectors_normalized']
        self.U_k = data['U_k']
        self.sigma_inv = data['sigma_inv']
        self.term_to_idx = data['term_to_idx']
        self.doc_to_idx = data['doc_to_idx']
        self.idx_to_doc = data['idx_to_doc']

        # Rebuild FAISS index
        if HAS_FAISS:
            self.faiss_index = faiss.IndexFlatIP(self.n_components)
            self.faiss_index.add(self.doc_vectors_normalized)
        else:
            self.faiss_index = None

        print(f"[LSI] Model loaded from {filepath} ({self.n_components} components)")


if __name__ == "__main__":
    if not HAS_NUMPY:
        print("numpy is required. Install with: pip install numpy")
        exit(1)

    # Load term and doc ID maps
    term_id_map = IdMap()
    doc_id_map = IdMap()
    with open(os.path.join('index', 'terms.dict'), 'rb') as f:
        term_id_map = pickle.load(f)
    with open(os.path.join('index', 'docs.dict'), 'rb') as f:
        doc_id_map = pickle.load(f)

    # Build LSI index
    lsi = LSIIndex(n_components=100)
    lsi.build_from_inverted_index(
        index_name='main_index',
        postings_encoding=VBEPostings,
        directory='index',
        term_id_map=term_id_map,
        doc_id_map=doc_id_map
    )

    # Save model
    lsi.save(os.path.join('index', 'lsi_model.pkl'))

    # Test retrieval
    queries = [
        "alkylated with radioactive iodoacetate",
        "psychodrama for disturbed children",
        "lipid metabolism in toxemia and normal pregnancy"
    ]

    for query in queries:
        print(f"\nQuery  : {query}")
        print("Results (LSI):")
        for (score, doc) in lsi.retrieve(query, term_id_map, doc_id_map, k=5):
            print(f"  {doc:30} {score:>.4f}")
