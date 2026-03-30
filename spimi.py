"""
spimi.py

Single-Pass In-Memory Indexing (SPIMI) — an alternative to BSBI.

Key differences from BSBI:
  1. No global term-to-termID mapping during initial indexing.
     Instead, each block builds its own in-memory dictionary
     mapping term strings directly to postings lists.
  2. When a memory block is "full" (simulated by a document count
     threshold), it is sorted by term and written to disk.
  3. The merge step merges by term string, not term ID.
  4. After merging, a global term-to-termID mapping is built.

Usage:
    python spimi.py

This module supports the same retrieval methods as BSBIIndex
(TF-IDF, BM25, WAND) since the final index format is identical.
"""

import os
import pickle
import contextlib
import heapq
import math

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings, EliasGammaPostings
from preprocessing import preprocess
from tqdm import tqdm


class SPIMIIndex:
    """
    SPIMI (Single-Pass In-Memory Indexing) implementation.

    Unlike BSBI which maintains a global termID mapping and sorts
    <termID, docID> pairs, SPIMI directly builds an inverted index
    (dictionary -> postings list) for each block in memory.

    Attributes
    ----------
    data_dir : str
        Path to document collection
    output_dir : str
        Path to output index files
    postings_encoding : class
        Compression class (StandardPostings, VBEPostings, EliasGammaPostings)
    index_name : str
        Name of the final merged index
    block_size : int
        Maximum number of documents per block (simulates memory limit)
    """

    def __init__(self, data_dir, output_dir, postings_encoding,
                 index_name="main_index_spimi", block_size=100):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.postings_encoding = postings_encoding
        self.index_name = index_name
        self.block_size = block_size

        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.intermediate_indices = []

    def save(self):
        """Save term and doc ID maps."""
        with open(os.path.join(self.output_dir, 'terms_spimi.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs_spimi.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Load term and doc ID maps."""
        with open(os.path.join(self.output_dir, 'terms_spimi.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs_spimi.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def spimi_invert(self, docs):
        """
        SPIMI-Invert: Build an in-memory inverted index for a block of documents.

        Unlike BSBI, this method works with term strings directly,
        not termIDs. It builds a dictionary mapping each term to its
        postings list with TF counts.

        Parameters
        ----------
        docs : List[Tuple[str, str]]
            List of (doc_path, doc_text) tuples

        Returns
        -------
        dict
            Mapping: term_string -> {doc_id: tf_count}
        """
        inverted_index = {}

        for doc_path, doc_text in docs:
            doc_id = self.doc_id_map[doc_path]
            tokens = preprocess(doc_text)

            for token in tokens:
                if token not in inverted_index:
                    # SPIMI: allocate a new postings list (dictionary)
                    inverted_index[token] = {}
                if doc_id not in inverted_index[token]:
                    inverted_index[token][doc_id] = 0
                inverted_index[token][doc_id] += 1

        return inverted_index

    def write_block_to_disk(self, inverted_index, block_id):
        """
        Write a SPIMI block to disk.

        Sort the terms alphabetically, then for each term:
          - Get term ID from global term_id_map
          - Sort postings by doc ID
          - Write to InvertedIndexWriter

        Parameters
        ----------
        inverted_index : dict
            term_string -> {doc_id: tf}
        block_id : str
            Identifier for this intermediate index
        """
        with InvertedIndexWriter(block_id, self.postings_encoding,
                                 directory=self.output_dir) as writer:
            # SPIMI sorts terms alphabetically, then assigns IDs
            for term in sorted(inverted_index.keys()):
                term_id = self.term_id_map[term]
                doc_tf = inverted_index[term]
                sorted_docs = sorted(doc_tf.keys())
                postings = sorted_docs
                tf_list = [doc_tf[d] for d in sorted_docs]
                writer.append(term_id, postings, tf_list)

    def merge(self, indices, merged_index):
        """
        Multi-way merge of intermediate indices.
        Same as BSBI merge — merges by termID after all terms
        have been assigned IDs.
        """
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        try:
            curr, postings, tf_list = next(merged_iter)
        except StopIteration:
            return
        for t, postings_, tf_list_ in merged_iter:
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(
                    list(zip(postings, tf_list)),
                    list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def index(self):
        """
        Main SPIMI indexing pipeline:
          1. Scan documents in blocks of self.block_size
          2. For each block, call spimi_invert to build in-memory inverted index
          3. Write each block to disk
          4. Merge all blocks into the final index
        """
        print(f"[SPIMI] Starting indexing with block_size={self.block_size}")

        # Collect all documents
        all_docs = []
        for block_dir in sorted(next(os.walk(self.data_dir))[1]):
            dir_path = "./" + self.data_dir + "/" + block_dir
            for filename in sorted(next(os.walk(dir_path))[2]):
                doc_path = dir_path + "/" + filename
                all_docs.append(doc_path)

        # Process in blocks of self.block_size
        block_num = 0
        for i in tqdm(range(0, len(all_docs), self.block_size), desc="SPIMI Indexing"):
            block_docs = []
            for doc_path in all_docs[i:i + self.block_size]:
                with open(doc_path, 'r', encoding='utf8', errors='surrogateescape') as f:
                    doc_text = f.read()
                block_docs.append((doc_path, doc_text))

            # SPIMI-Invert: build in-memory inverted index for this block
            inverted_index = self.spimi_invert(block_docs)

            # Write block to disk
            block_id = f'spimi_intermediate_{block_num}'
            self.intermediate_indices.append(block_id)
            self.write_block_to_disk(inverted_index, block_id)
            block_num += 1

        # Save global term-to-ID and doc-to-ID mappings
        self.save()

        # Merge all intermediate indices
        print(f"[SPIMI] Merging {len(self.intermediate_indices)} blocks...")
        with InvertedIndexWriter(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [
                    stack.enter_context(
                        InvertedIndexReader(idx_id, self.postings_encoding,
                                           directory=self.output_dir))
                    for idx_id in self.intermediate_indices
                ]
                self.merge(indices, merged_index)
            merged_index.compute_avg_doc_length()

        print(f"[SPIMI] Indexing complete. Terms: {len(self.term_id_map)}, Docs: {len(self.doc_id_map)}")

    def retrieve_tfidf(self, query, k=10):
        """TF-IDF retrieval (same logic as BSBIIndex)."""
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        query_terms = preprocess(query)
        terms = []
        for word in query_terms:
            if word in self.term_id_map.str_to_id:
                terms.append(self.term_id_map[word])

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    N = len(merged_index.doc_length)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        if tf > 0:
                            scores[doc_id] += math.log(N / df) * (1 + math.log(tf))

            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """BM25 retrieval (same logic as BSBIIndex)."""
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        query_terms = preprocess(query)
        terms = []
        for word in query_terms:
            if word in self.term_id_map.str_to_id:
                terms.append(self.term_id_map[word])

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            avgdl = merged_index.avg_doc_length
            if avgdl == 0:
                avgdl = 1.0

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        dl = merged_index.doc_length.get(doc_id, 0)
                        tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        scores[doc_id] += idf * tf_component

            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]


if __name__ == "__main__":
    spimi = SPIMIIndex(data_dir='collection',
                       postings_encoding=VBEPostings,
                       output_dir='index',
                       block_size=100)
    spimi.index()

    # Quick test retrieval
    queries = ["alkylated with radioactive iodoacetate",
               "psychodrama for disturbed children",
               "lipid metabolism in toxemia and normal pregnancy"]

    for query in queries:
        print(f"\nQuery  : {query}")
        print("Results (BM25):")
        for (score, doc) in spimi.retrieve_bm25(query, k=5):
            print(f"  {doc:30} {score:>.3f}")
