import re
import math
from bsbi import BSBIIndex
from compression import VBEPostings

# Metric 1: RBP (Rank-Biased Precision) p = 0.8

def rbp(ranking, p=0.8):
    """
    Menghitung search effectiveness metric score dengan
    Rank Biased Precision (RBP).

    Parameters
    ----------
    ranking: List[int]
        Vektor biner seperti [1, 0, 1, 1, 1, 0]
        gold standard relevansi dari dokumen di rank 1, 2, 3, dst.

    Returns
    -------
    float
        Score RBP
    """
    score = 0.
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] * (p ** (i - 1))
    return (1 - p) * score


# Metric 2: DCG (Discounted Cumulative Gain)

def dcg(ranking):
    """
    Menghitung Discounted Cumulative Gain (DCG).

    DCG@k = Σ_{i=1}^{k} rel_i / log2(i + 1)

    Parameters
    ----------
    ranking: List[int]
        Vektor biner relevansi [1, 0, 1, 1, ...]

    Returns
    -------
    float
        Score DCG
    """
    score = 0.0
    for i in range(len(ranking)):
        # i is 0-indexed, rank position is i+1
        score += ranking[i] / math.log2(i + 2)  # log2(rank + 1) = log2(i + 2)
    return score


## Metric 3: NDCG (Normalized Discounted Cumulative Gain)

def ndcg(ranking):
    """
    Menghitung Normalized Discounted Cumulative Gain (NDCG).

    NDCG@k = DCG@k / IDCG@k

    where IDCG is the DCG of the ideal ranking (sorted by relevance descending).

    Parameters
    ----------
    ranking: List[int]
        Vektor biner relevansi [1, 0, 1, 1, ...]

    Returns
    -------
    float
        Score NDCG (between 0 and 1)
    """
    actual_dcg = dcg(ranking)
    # Ideal ranking: sort by relevance descending
    ideal_ranking = sorted(ranking, reverse=True)
    ideal_dcg = dcg(ideal_ranking)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


######## >>>>> Metric 4: AP (Average Precision)

def ap(ranking):
    """
    Menghitung Average Precision (AP).

    AP = (1/R) * Σ_{k=1}^{n} Precision(k) * rel(k)

    where R is total number of relevant documents in the ranking,
    and Precision(k) is the precision at position k.

    Parameters
    ----------
    ranking: List[int]
        Vektor biner relevansi [1, 0, 1, 1, ...]

    Returns
    -------
    float
        Score AP (between 0 and 1)
    """
    R = sum(ranking)  # total relevant documents
    if R == 0:
        return 0.0

    score = 0.0
    relevant_count = 0
    for i in range(len(ranking)):
        if ranking[i] == 1:
            relevant_count += 1
            precision_at_k = relevant_count / (i + 1)
            score += precision_at_k
    return score / R


######## >>>>> memuat qrels

def load_qrels(qrel_file="qrels.txt", max_q_id=30, max_doc_id=1033):
    """
    Memuat query relevance judgment (qrels) dalam format
    dictionary of dictionary: qrels[query_id][document_id]

    contoh: qrels["Q3"][12] = 1 artinya Doc 12 relevan dengan Q3
            qrels["Q3"][10] = 0 artinya Doc 10 tidak relevan dengan Q3
    """
    qrels = {"Q" + str(i): {j: 0 for j in range(1, max_doc_id + 1)}
             for i in range(1, max_q_id + 1)}
    with open(qrel_file) as file:
        for line in file:
            parts = line.strip().split()
            qid = parts[0]
            did = int(parts[1])
            qrels[qid][did] = 1
    return qrels


######## >>>>> EVALUASI !

def eval_retrieval(qrels, retrieval_func, method_name="", query_file="queries.txt", k=1000):
    """
    Evaluate a retrieval function using all four metrics:
    RBP, DCG, NDCG, and AP.

    Parameters
    ----------
    qrels : dict
        Loaded qrels (from load_qrels)
    retrieval_func : callable
        Function that takes (query_str, k) and returns List[(score, doc_name)]
    method_name : str
        Name of the retrieval method (for display)
    query_file : str
        Path to queries file
    k : int
        Number of top results to retrieve

    Returns
    -------
    dict
        Dictionary of metric_name -> mean_score
    """
    with open(query_file) as file:
        rbp_scores = []
        dcg_scores = []
        ndcg_scores = []
        ap_scores = []

        for qline in file:
            parts = qline.strip().split()
            if not parts:
                continue
            qid = parts[0]
            query = " ".join(parts[1:])

            ranking = []
            for (score, doc) in retrieval_func(query, k=k):
                did = int(re.search(r'\/.*\/.*\/(.*?)\.txt', doc).group(1))
                ranking.append(qrels[qid][did])

            rbp_scores.append(rbp(ranking))
            dcg_scores.append(dcg(ranking))
            ndcg_scores.append(ndcg(ranking))
            ap_scores.append(ap(ranking))

    results = {
        'RBP': sum(rbp_scores) / len(rbp_scores) if rbp_scores else 0,
        'DCG': sum(dcg_scores) / len(dcg_scores) if dcg_scores else 0,
        'NDCG': sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0,
        'AP': sum(ap_scores) / len(ap_scores) if ap_scores else 0,
    }

    if method_name:
        print(f"\n{'='*60}")
        print(f"  Evaluation Results: {method_name}")
        print(f"{'='*60}")
        for metric, score in results.items():
            print(f"  {metric:>6s} = {score:.4f}")

    return results


def eval(qrels, query_file="queries.txt", k=1000):
    """
    Loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    Evaluates both TF-IDF and BM25 retrieval methods.
    """
    BSBI_instance = BSBIIndex(data_dir='collection',
                              postings_encoding=VBEPostings,
                              output_dir='index')

    # Evaluate TF-IDF
    tfidf_results = eval_retrieval(
        qrels,
        BSBI_instance.retrieve_tfidf,
        method_name="TF-IDF",
        query_file=query_file,
        k=k
    )

    # Evaluate BM25
    bm25_results = eval_retrieval(
        qrels,
        BSBI_instance.retrieve_bm25,
        method_name="BM25",
        query_file=query_file,
        k=k
    )

    # Evaluate BM25 with WAND
    wand_results = eval_retrieval(
        qrels,
        BSBI_instance.retrieve_bm25_wand,
        method_name="BM25 + WAND",
        query_file=query_file,
        k=k
    )

    # Print comparison table
    print("\n\n" + "=" * 70)
    print("  COMPARISON TABLE")
    print("=" * 70)
    print(f"  {'Metric':<10} {'TF-IDF':>12} {'BM25':>12} {'BM25+WAND':>12}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
    for metric in ['RBP', 'DCG', 'NDCG', 'AP']:
        print(f"  {metric:<10} {tfidf_results[metric]:>12.4f} {bm25_results[metric]:>12.4f} {wand_results[metric]:>12.4f}")
    print("=" * 70)


if __name__ == '__main__':
    qrels = load_qrels()

    assert qrels["Q1"][166] == 1, "qrels salah"
    assert qrels["Q1"][300] == 0, "qrels salah"

    # Test individual metrics
    test_ranking = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    print("Test ranking:", test_ranking)
    print(f"  RBP  = {rbp(test_ranking):.4f}")
    print(f"  DCG  = {dcg(test_ranking):.4f}")
    print(f"  NDCG = {ndcg(test_ranking):.4f}")
    print(f"  AP   = {ap(test_ranking):.4f}")

    # Full evaluation
    eval(qrels)