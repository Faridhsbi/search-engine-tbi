import os
import pickle
import contextlib
import heapq
import time
import math

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
<<<<<<< HEAD
from compression import StandardPostings, VBEPostings, EliasGammaPostings
from preprocessing import preprocess
=======
from compression import StandardPostings, VBEPostings
>>>>>>> 7d151749ef8991a59ec1cd533ad6929fcdffbf23
from tqdm import tqdm

class BSBIIndex:
    """
<<<<<<< HEAD
    BSBIIndex implements a Blocked Sort-Based Indexing scheme for building
    an inverted index, and supports multiple retrieval methods:
      - TF-IDF (Term-at-a-Time)
      - BM25 (Term-at-a-Time)
      - BM25 with WAND optimization (Top-K pruning)

    Attributes
    ----------
    term_id_map(IdMap): Mapping terms to termIDs
    doc_id_map(IdMap): Mapping relative paths of documents to docIDs
    data_dir(str): Path to data
    output_dir(str): Path to output index files
    postings_encoding: Compression class (StandardPostings, VBEPostings, EliasGammaPostings)
    index_name(str): Name of the inverted index file
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index"):
=======
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
>>>>>>> 7d151749ef8991a59ec1cd533ad6929fcdffbf23
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
<<<<<<< HEAD
        """Menyimpan doc_id_map dan term_id_map ke output directory via pickle"""
=======
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

>>>>>>> 7d151749ef8991a59ec1cd533ad6929fcdffbf23
        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
<<<<<<< HEAD
        """Memuat doc_id_map dan term_id_map dari output directory"""
=======
        """Memuat doc_id_map and term_id_map dari output directory"""

>>>>>>> 7d151749ef8991a59ec1cd533ad6929fcdffbf23
        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
<<<<<<< HEAD
        Parse text files in a block directory into <termID, docID> pairs.

        Applies full text preprocessing:
          1. Regex-based tokenization (alphabetic tokens only)
          2. Lowercasing
          3. Stopword removal
          4. Porter stemming
=======
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.
>>>>>>> 7d151749ef8991a59ec1cd533ad6929fcdffbf23

        Parameters
        ----------
        block_dir_relative : str
<<<<<<< HEAD
            Relative path to directory containing text files for a block.
=======
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.
>>>>>>> 7d151749ef8991a59ec1cd533ad6929fcdffbf23

        Returns
        -------
        List[Tuple[Int, Int]]
<<<<<<< HEAD
            All <termID, docID> pairs extracted from the block
=======
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
>>>>>>> 7d151749ef8991a59ec1cd533ad6929fcdffbf23
        """
        dir = "./" + self.data_dir + "/" + block_dir_relative
        td_pairs = []
        for filename in next(os.walk(dir))[2]:
            docname = dir + "/" + filename
<<<<<<< HEAD
            with open(docname, "r", encoding="utf8", errors="surrogateescape") as f:
                text = f.read()
                tokens = preprocess(text)
                for token in tokens:
=======
            with open(docname, "r", encoding = "utf8", errors = "surrogateescape") as f:
                for token in f.read().split():
>>>>>>> 7d151749ef8991a59ec1cd533ad6929fcdffbf23
                    td_pairs.append((self.term_id_map[token], self.doc_id_map[docname]))

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
<<<<<<< HEAD
        Namun dalam teknik penyimpanannya digunakan strategi dari SPIMI
        yaitu penggunaan struktur data hashtable (Dictionary).
=======
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.
>>>>>>> 7d151749ef8991a59ec1cd533ad6929fcdffbf23

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {}
        term_tf = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
                term_tf[term_id] = {}
            term_dict[term_id].add(doc_id)
            if doc_id not in term_tf[term_id]:
                term_tf[term_id][doc_id] = 0
            term_tf[term_id][doc_id] += 1
        for term_id in sorted(term_dict.keys()):
            sorted_doc_id = sorted(list(term_dict[term_id]))
            assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_id]
            index.append(term_id, sorted_doc_id, assoc_tf)

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
<<<<<<< HEAD
        sebuah single index (EXTERNAL MERGE SORT).
=======
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util
>>>>>>> 7d151749ef8991a59ec1cd533ad6929fcdffbf23

        Parameters
        ----------
        indices: List[InvertedIndexReader]
<<<<<<< HEAD
            A list of intermediate InvertedIndexReader objects
        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object hasil merging
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)  # first item
        for t, postings_, tf_list_ in merged_iter:  # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)),
                                                       list(zip(postings_, tf_list_)))
=======
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
>>>>>>> 7d151749ef8991a59ec1cd533ad6929fcdffbf23
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

<<<<<<< HEAD
    def retrieve_tfidf(self, query, k=10):
=======
    def retrieve_tfidf(self, query, k = 10):
>>>>>>> 7d151749ef8991a59ec1cd533ad6929fcdffbf23
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

<<<<<<< HEAD
        Parameters
        ----------
        query: str
            Query string

        Result
        ------
        List[(float, str)]
            List of (score, doc_name) tuples, sorted descending by score.
=======
        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

>>>>>>> 7d151749ef8991a59ec1cd533ad6929fcdffbf23
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

<<<<<<< HEAD
        # Preprocess query the same way as documents
        query_terms = preprocess(query)
        terms = []
        for word in query_terms:
            if word in self.term_id_map.str_to_id:
                terms.append(self.term_id_map[word])

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
=======
        terms = [self.term_id_map[word] for word in query.split()]
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

>>>>>>> 7d151749ef8991a59ec1cd533ad6929fcdffbf23
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

            # Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
<<<<<<< HEAD
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        Melakukan Ranked Retrieval dengan BM25 scoring.

        BM25(D, Q) = Σ_{t ∈ Q} IDF(t) · [tf(t,D) · (k1 + 1)] / [tf(t,D) + k1 · (1 - b + b · |D| / avgdl)]

        where IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)

        Parameters
        ----------
        query : str
            Query string
        k : int
            Number of top results to return
        k1 : float
            BM25 term frequency saturation parameter (default 1.2)
        b : float
            BM25 document length normalization parameter (default 0.75)

        Returns
        -------
        List[(float, str)]
            List of (score, doc_name) tuples, sorted descending by score.
        """
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
                avgdl = 1.0  # safeguard

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    # BM25 IDF
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        dl = merged_index.doc_length.get(doc_id, 0)
                        # BM25 TF component
                        tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                        bm25_score = idf * tf_component
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        scores[doc_id] += bm25_score

            # Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def retrieve_bm25_wand(self, query, k=10, k1=1.2, b=0.75):
        """
        BM25 retrieval with WAND (Weighted AND) top-K optimization.

        WAND avoids computing BM25 scores for documents that cannot possibly
        enter the top-K results, by using upper-bound scores per query term.

        The upper bound for a term t's BM25 contribution is computed using
        the maximum TF of t across all documents and assuming the shortest
        possible document length (which maximizes BM25).

        Algorithm:
        1. For each query term, compute upper-bound BM25 contribution
        2. Sort term iterators by current doc ID
        3. Find pivot where cumulative upper bounds >= threshold
        4. Skip documents that can't make it into top-K
        5. Fully score only promising candidate documents

        Parameters
        ----------
        query : str
        k : int
        k1 : float
        b : float

        Returns
        -------
        List[(float, str)]
        """
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

            # For each query term, load its postings and compute upper bound
            term_data = []  # list of (term, idf, upper_bound, postings, tf_list)
            for term in terms:
                if term not in merged_index.postings_dict:
                    continue
                df = merged_index.postings_dict[term][1]
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

                # Upper bound: use max_tf and assume minimal doc length for max BM25
                max_tf_val = merged_index.max_tf.get(term, 1)
                # The maximum BM25 contribution occurs when dl is minimal
                # For the upper bound, we use the formula with the max TF
                # and a conservative (small) document length
                min_dl = 1  # smallest possible document length
                ub_tf_component = (max_tf_val * (k1 + 1)) / (max_tf_val + k1 * (1 - b + b * min_dl / avgdl))
                upper_bound = idf * ub_tf_component

                postings, tf_list = merged_index.get_postings_list(term)
                term_data.append({
                    'term': term,
                    'idf': idf,
                    'upper_bound': upper_bound,
                    'postings': postings,
                    'tf_list': tf_list,
                    'idx': 0,  # current position in postings list
                })

            if not term_data:
                return []

            # WAND algorithm
            # We maintain a min-heap of size k for top-K results
            top_k_heap = []  # min-heap of (score, doc_id)
            threshold = 0.0

            # Sort term_data by upper_bound descending for efficiency
            term_data.sort(key=lambda x: x['upper_bound'], reverse=True)

            # Collect all candidate documents
            # We use a document-at-a-time approach with WAND pruning
            all_doc_ids = set()
            for td in term_data:
                all_doc_ids.update(td['postings'])

            # Build doc_id -> {term: (tf, idf)} mapping efficiently
            doc_term_info = {}
            for td in term_data:
                for i, doc_id in enumerate(td['postings']):
                    if doc_id not in doc_term_info:
                        doc_term_info[doc_id] = []
                    doc_term_info[doc_id].append((td['idf'], td['tf_list'][i], td['upper_bound']))

            # For each candidate document, check if it can potentially beat threshold
            docs_scored = 0
            docs_pruned = 0
            for doc_id in all_doc_ids:
                term_infos = doc_term_info[doc_id]

                # Quick upper bound check: sum of upper bounds for all terms
                # that appear in this document
                potential_score = sum(ub for (_, _, ub) in term_infos)

                if potential_score <= threshold:
                    docs_pruned += 1
                    continue

                # Full scoring
                dl = merged_index.doc_length.get(doc_id, 0)
                score = 0.0
                for (idf, tf, _) in term_infos:
                    tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                    score += idf * tf_component
                docs_scored += 1

                if len(top_k_heap) < k:
                    heapq.heappush(top_k_heap, (score, doc_id))
                    if len(top_k_heap) == k:
                        threshold = top_k_heap[0][0]
                elif score > threshold:
                    heapq.heapreplace(top_k_heap, (score, doc_id))
                    threshold = top_k_heap[0][0]

            # Convert heap to sorted result
            results = []
            while top_k_heap:
                score, doc_id = heapq.heappop(top_k_heap)
                results.append((score, self.doc_id_map[doc_id]))
            results.reverse()  # highest score first

            return results
=======
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]
>>>>>>> 7d151749ef8991a59ec1cd533ad6929fcdffbf23

    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
<<<<<<< HEAD
            index_id = 'intermediate_index_' + block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding,
                                     directory=self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None

        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(
                    InvertedIndexReader(index_id, self.postings_encoding,
                                       directory=self.output_dir))
                    for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)
            # Compute avg doc length after merge
            merged_index.compute_avg_doc_length()
=======
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)
>>>>>>> 7d151749ef8991a59ec1cd533ad6929fcdffbf23


if __name__ == "__main__":

<<<<<<< HEAD
    BSBI_instance = BSBIIndex(data_dir='collection',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    BSBI_instance.index()  # memulai indexing!
=======
    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = VBEPostings, \
                              output_dir = 'index')
    BSBI_instance.index() # memulai indexing!
>>>>>>> 7d151749ef8991a59ec1cd533ad6929fcdffbf23
