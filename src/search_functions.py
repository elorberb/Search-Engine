

# ------ Search Functions ----------

def calc_search_title(query: str):
    query = tokenize(query)
    docs_id = binary_search(query, title_index, TITLE_BUCKET)
    return docs_id[:100]


def calc_search_anchor(query: str):
    query = tokenize(query)
    docs_id = binary_search(query, anchor_index, ANCHOR_BUCKET)
    return docs_id[:100]


def calc_search_body(query: str):
    query = tokenize(query)
    QL = len(query)
    QL_norm = np.linalg.norm(QL)
    query_tfidf, docs_tfidf = get_query_and_docs_tfidf(query, text_index, QL_norm, TEXT_BUCKET)
    cosine_sim = calc_cosine_similarity(query_tfidf, docs_tfidf, QL_norm, text_index)
    docs_id = get_docs_id_ordered_by_cosine_similarity(cosine_sim)

    return docs_id[:100]


def calc_search_body_stem(query: str):

    query = tokenize(query, stem=True)
    QL = len(query)
    QL_norm = np.linalg.norm(QL)
    query_tfidf, docs_tfidf = get_query_and_docs_tfidf(query, text_stem_index, QL_norm, TEXT_STEM_BUCKET)
    cosine_sim = calc_cosine_similarity(query_tfidf, docs_tfidf, QL_norm, text_stem_index)
    docs_id = get_docs_id_ordered_by_cosine_similarity(cosine_sim)

    return docs_id[:100]


def calc_search_anchor_double(query: str):
    query = tokenize(query)
    docs_id = binary_search(query, anchor_double_index, ANCHOR_DOUBLE_BUCKET)
    return docs_id[:100]


def search_bm25_body(query: str):
    query = tokenize(query)
    return calc_bm25(query, text_index, TEXT_BUCKET, DL)


def search_bm25_body_stem(query: str):
    query = tokenize(query, stem=True)
    return calc_bm25(query, text_stem_index, TEXT_STEM_BUCKET, DL)


# ------- Page Views and Page Rank Functions
def get_page_view(docs_id: List[str]) -> List[int]:
    values = [page_view[doc_id] for doc_id in docs_id]
    return values


def get_page_rank(docs_id: List[str]) -> List[int]:
    values = [page_rank[doc_id] for doc_id in docs_id]
    return values


# ---- Experimented Search Functions ----

# search using the search function from all the search functions ordered by page rank values
def search_v1(query):
    scores_by_page_rank = []
    doc_ids_title = calc_search_title(query)
    doc_ids_body = calc_search_body(query)
    doc_ids_anchor = calc_search_anchor(query)
    mutual_docs_ids = set(doc_ids_title) | set(doc_ids_body) |set(doc_ids_anchor)
    mutual_docs_ids = list(mutual_docs_ids)
    for doc_id in mutual_docs_ids:
        try:
            scores_by_page_rank.append((doc_id, page_rank[doc_id]))
        except:
            scores_by_page_rank.append((doc_id, 0))
    sorted_scores_by_page_rank = sorted(scores_by_page_rank, key=lambda x: x[1], reverse=True)
    return map_tuple2doc_id(sorted_scores_by_page_rank)


# search using the search function from all the search functions ordered by page views values
def search_v2(query):
    scores_by_page_views = []
    doc_ids_title = calc_search_title(query)
    doc_ids_body = calc_search_body(query)
    doc_ids_anchor = calc_search_anchor(query)
    mutual_docs_ids = set(doc_ids_title) | set(doc_ids_body) |set(doc_ids_anchor)
    mutual_docs_ids = list(mutual_docs_ids)
    for doc_id in mutual_docs_ids:
        scores_by_page_views.append((doc_id, page_views[doc_id]))
    sorted_scores_by_page_views = sorted(scores_by_page_views, key=lambda x: x[1], reverse=True)
    return map_tuple2doc_id(sorted_scores_by_page_views)


# search only by anchor index with word2vec model to extend query tokens.
def search_v3(query):
    query = tokenize(query, expand=True)
    docs_id = binary_search(query, anchor_index, ANCHOR_BUCKET)
    return docs_id[:100]


# search only by title index with word2vec model to extend query tokens.
def search_v4(query):
    query = tokenize(query, expand=True)
    docs_id = binary_search(query, title_index, TITLE_BUCKET)
    return docs_id[:100]


# search only by title index as it recieves the best results
def search_v5(query):
    return calc_search_title(query)