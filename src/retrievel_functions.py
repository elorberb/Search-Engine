from backend import *

# -------- tfidf retrievel functions --------


def calc_idf(index, token):
    """Calculate the IDF of a token in an index."""
    return np.log2(len(index.DL) / (index.df[token]))


def calc_tfidf(tf: float, idf: float, dl: float) -> float:
    """Calculate the TF-IDF of a token in a document."""
    return tf * idf / dl


def update_query_tfidf(idf: float, tokens: list, token: str, QL: float) -> float:
    """Update the TF-IDF score of a token in the query."""
    tf = tokens.count(token)
    return calc_tfidf(tf, idf, QL)


def get_query_and_docs_tfidf(tokens: list, index: InvertedIndex, QL: float, bucket_name: str) -> tuple:
    """Calculate the TF-IDF scores of tokens in a query and documents."""
    query = {}
    docs = defaultdict(list)
    for token in set(tokens):  # Only calculate the TF-IDF score for each unique token once
        postings = get_posting(index, token, bucket_name)
        idf = calc_idf(index, token)
        query[token] = update_query_tfidf(idf, tokens, token, QL)
        for doc_id, freq in postings:
            docs[token].append((doc_id, calc_tfidf(freq, idf, index.DL[doc_id])))
    return query, docs


def calc_cosine_similarity(query_tfidf: dict, docs_tfidf: dict, QL: float) -> dict:
    """Calculate the cosine similarity between a query and documents."""
    cosine_sim = {}
    for token, val in docs_tfidf.items():
        for doc_id, tfidf in val:
            if doc_id in cosine_sim:
                cosine_sim[doc_id] += tfidf * query_tfidf[token]
            else:
                cosine_sim[doc_id] = tfidf * query_tfidf[token]
    return {doc_id: numerator / DL_normed[doc_id] * QL for doc_id, numerator in cosine_sim.items()}


def get_docs_id_ordered_by_cosine_similarity(cosine_sim: Dict[int, float]) -> List[int]:
    """Returns sorted document IDs based on cosine similarity."""
    return sorted(cosine_sim, key=cosine_sim.get, reverse=True)