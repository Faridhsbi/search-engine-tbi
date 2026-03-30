"""
adaptive.py

Adaptive Retrieval using RM3 (Relevance Model 3) for 
Pseudo-Relevance Feedback (PRF).

RM3 improves retrieval by:
  1. First-pass retrieval with BM25 → get top-K "pseudo-relevant" documents
  2. Extract and weight terms from these documents (relevance model)
  3. Interpolate the original query with the relevance model terms
  4. Re-run retrieval with the expanded query

This implements the idea from:
  Lavrenko & Croft, "Relevance Based Language Models" (SIGIR 2001)
  Abdul-Jaleel et al., "UMass at TREC 2004" — RM3 variant

Usage:
    python adaptive.py
"""

import math
import os
import pickle
from collections import Counter

from index import InvertedIndexReader
from util import IdMap
from compression import VBEPostings
from preprocessing import preprocess


class AdaptiveRetrieval:
    """
    Adaptive retrieval engine using RM3 pseudo-relevance feedback.

    The process:
      1. Run initial BM25 retrieval
      2. Analyze top-K documents to extract expansion terms
      3. Build an expanded query using interpolation
      4. Re-run retrieval with the expanded query

    Attributes
    ----------
    bsbi_index : BSBIIndex or SPIMIIndex
        The underlying index for retrieval
    fb_docs : int
        Number of pseudo-relevant documents for feedback (default 10)
    fb_terms : int
        Number of expansion terms to add (default 20)
    alpha : float
        Interpolation weight for original query (default 0.5)
        expanded_query = alpha * Q_original + (1-alpha) * Q_rm
    """

    def __init__(self, bsbi_index, fb_docs=10, fb_terms=20, alpha=0.5):
        self.bsbi_index = bsbi_index
        self.fb_docs = fb_docs
        self.fb_terms = fb_terms
        self.alpha = alpha

    def estimate_relevance_model(self, query, index_reader):
        """
        Estimate the relevance model (RM1) from pseudo-relevant documents.

        For each term t in the top-K documents:
          P(t|R) ∝ Σ_{D ∈ top-K} P(t|D) * P(Q|D)

        where:
          P(t|D) = tf(t,D) / |D|   (maximum likelihood estimate)
          P(Q|D) = Π_{q ∈ Q} P(q|D)  (query likelihood)

        Parameters
        ----------
        query : str
            The original query
        index_reader : InvertedIndexReader
            Open index reader for fetching postings

        Returns
        -------
        dict
            Mapping: term_id -> relevance weight
        """
        # First-pass retrieval
        initial_results = self.bsbi_index.retrieve_bm25(query, k=self.fb_docs)
        if not initial_results:
            return {}

        # Get doc IDs of pseudo-relevant documents
        fb_doc_names = [doc_name for (_, doc_name) in initial_results]
        fb_doc_ids = set()
        for doc_name in fb_doc_names:
            for i in range(len(self.bsbi_index.doc_id_map.id_to_str)):
                if self.bsbi_index.doc_id_map[i] == doc_name:
                    fb_doc_ids.add(i)
                    break

        if not fb_doc_ids:
            return {}

        # Collect term frequencies from feedback documents
        # We need to iterate through the index to find terms in these docs
        term_scores = Counter()

        index_reader.reset()
        for term_id, postings, tf_list in index_reader:
            for i in range(len(postings)):
                doc_id = postings[i]
                if doc_id in fb_doc_ids:
                    tf = tf_list[i]
                    dl = index_reader.doc_length.get(doc_id, 1)
                    # P(t|D) = tf / dl
                    p_t_d = tf / dl if dl > 0 else 0
                    term_scores[term_id] += p_t_d

        # Normalize and select top fb_terms
        if not term_scores:
            return {}

        total = sum(term_scores.values())
        for t in term_scores:
            term_scores[t] /= total

        # Return top fb_terms
        top_terms = dict(term_scores.most_common(self.fb_terms))
        return top_terms

    def rm3_expand(self, query):
        """
        Expand a query using RM3 pseudo-relevance feedback.

        RM3 interpolates the original query model with the
        relevance model:
          P_RM3(t) = α * P_orig(t) + (1-α) * P_RM1(t)

        Parameters
        ----------
        query : str
            Original query string

        Returns
        -------
        dict
            Mapping: term_id -> weight (for weighted retrieval)
        """
        if len(self.bsbi_index.term_id_map) == 0:
            self.bsbi_index.load()

        # Original query term weights
        query_tokens = preprocess(query)
        orig_weights = Counter()
        total_query_terms = len(query_tokens)
        for token in query_tokens:
            if token in self.bsbi_index.term_id_map.str_to_id:
                term_id = self.bsbi_index.term_id_map[token]
                orig_weights[term_id] += 1.0 / total_query_terms if total_query_terms > 0 else 0

        # Estimate relevance model
        with InvertedIndexReader(self.bsbi_index.index_name,
                                 self.bsbi_index.postings_encoding,
                                 directory=self.bsbi_index.output_dir) as reader:
            rm_weights = self.estimate_relevance_model(query, reader)

        # Interpolate: RM3 = α * orig + (1-α) * RM1
        expanded = Counter()
        for term_id, w in orig_weights.items():
            expanded[term_id] += self.alpha * w
        for term_id, w in rm_weights.items():
            expanded[term_id] += (1 - self.alpha) * w

        return dict(expanded)

    def retrieve_adaptive(self, query, k=10, k1=1.2, b=0.75):
        """
        Full adaptive retrieval pipeline:
          1. Expand query using RM3
          2. Re-rank documents using expanded query weights with BM25

        Parameters
        ----------
        query : str
        k : int
        k1 : float
        b : float

        Returns
        -------
        List[Tuple[float, str]]
            List of (score, doc_name) tuples
        """
        if len(self.bsbi_index.term_id_map) == 0:
            self.bsbi_index.load()

        # Get expanded query weights
        expanded_weights = self.rm3_expand(query)
        if not expanded_weights:
            # Fall back to standard BM25
            return self.bsbi_index.retrieve_bm25(query, k=k, k1=k1, b=b)

        # Re-rank with expanded query using BM25 with query term weights
        with InvertedIndexReader(self.bsbi_index.index_name,
                                 self.bsbi_index.postings_encoding,
                                 directory=self.bsbi_index.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            avgdl = merged_index.avg_doc_length
            if avgdl == 0:
                avgdl = 1.0

            scores = {}
            for term_id, query_weight in expanded_weights.items():
                if term_id in merged_index.postings_dict:
                    df = merged_index.postings_dict[term_id][1]
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                    postings, tf_list = merged_index.get_postings_list(term_id)
                    for i in range(len(postings)):
                        doc_id = postings[i]
                        tf = tf_list[i]
                        dl = merged_index.doc_length.get(doc_id, 0)
                        tf_comp = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                        bm25_score = idf * tf_comp
                        # Weight by the RM3 query term weight
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        scores[doc_id] += query_weight * bm25_score

            docs = [(score, self.bsbi_index.doc_id_map[doc_id])
                    for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]


if __name__ == "__main__":
    from bsbi import BSBIIndex

    BSBI_instance = BSBIIndex(data_dir='collection',
                              postings_encoding=VBEPostings,
                              output_dir='index')

    adaptive = AdaptiveRetrieval(BSBI_instance, fb_docs=10, fb_terms=20, alpha=0.5)

    queries = [
        "alkylated with radioactive iodoacetate",
        "psychodrama for disturbed children",
        "lipid metabolism in toxemia and normal pregnancy"
    ]

    for query in queries:
        print(f"\nQuery  : {query}")

        # Standard BM25
        print("  BM25 Results:")
        for (score, doc) in BSBI_instance.retrieve_bm25(query, k=5):
            print(f"    {doc:30} {score:>.3f}")

        # Adaptive (RM3)
        print("  Adaptive (RM3) Results:")
        for (score, doc) in adaptive.retrieve_adaptive(query, k=5):
            print(f"    {doc:30} {score:>.3f}")
