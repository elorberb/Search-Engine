from inverted_index_gcp import *
from time import time as tm
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from nltk.util import ngrams
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
from itertools import combinations
import math
from itertools import chain
import time

from bucket_manipulation import *

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"amazing-badge-343010-879c0a90f001.json"

# Bucket names
TEXT_BUCKET = 'body_text_no_stem'
TEXT_STEM_BUCKET = 'body_stem'
TITLE_BUCKET = 'title_bucket6'
ANCHOR_BUCKET = 'anchor_bucket3'
ANCHOR_DOUBLE_BUCKET = 'double_anchor_index'
EXTRA_FILES_BUCKET = 'extra_files_search'

# ---- Initialize Indices ----
text_index = ReadBucketData(TEXT_BUCKET).get_inverted_index(source_idx='postings_gcp/inverted_index_text.pkl',
                                                            dest_file='downloaded/index.pkl')

text_stem_index = ReadBucketData(TEXT_STEM_BUCKET).get_inverted_index(source_idx='postings_gcp/index_body_bucket0.pkl',
                                                                      dest_file='downloaded/index.pkl')

title_index = ReadBucketData(TITLE_BUCKET).get_inverted_index(source_idx='postings_gcp/title_index_nostem.pkl',
                                                              dest_file='downloaded/index.pkl')

anchor_index = ReadBucketData(ANCHOR_BUCKET).get_inverted_index(source_idx='postings_gcp/index_anchor_bucket0.pkl',
                                                                dest_file='downloaded/index.pkl')

anchor_double_index = ReadBucketData(ANCHOR_DOUBLE_BUCKET).get_inverted_index(
    source_idx='postings_gcp/inverted_index_anchor_double.pkl',
    dest_file='downloaded/index.pkl')

anchor_double_index.df = dict(anchor_double_index.df)

# ---- Initialize Page Rank, Views, id2title and word2vec model ----
page_views = ReadBucketData(EXTRA_FILES_BUCKET).get_pkl_file("pages/pageviews.pkl", "downloaded/pageviews.pkl")
page_rank = pd.read_pickle(
    ReadBucketData(EXTRA_FILES_BUCKET).download_from_buck("pages/page_rank.pickle", "downloaded/page_rank.pickle"))

# ---- calculate and Save Amount of docs, Doc Length for eah doc and id2title mapping dict ----
id2title = pd.read_pickle(
    ReadBucketData(EXTRA_FILES_BUCKET).download_from_buck("pages/id2title.pickle", "downloaded/id2title.pickle"))
num_of_docs = len(text_index.DL)
DL = text_index.DL
DL_normed = text_index.DS

# ---- download word2vec over wikipedia model  -----
word2vec_file = ReadBucketData(EXTRA_FILES_BUCKET).download_from_buck("wiki-news-300d-1M.vec",
                                                                      "downloaded/wiki-news-300d-1M.vec")
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_file)


# -------- retrieval functions --------


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


# ------ Search Functions ----------

def calc_search(query: str):
    return search_v5(query)


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
    cosine_sim = calc_cosine_similarity(query_tfidf, docs_tfidf, QL_norm)
    docs_id = get_docs_id_ordered_by_cosine_similarity(cosine_sim)

    return docs_id[:100]


def calc_search_body_stem(query: str):
    query = tokenize(query, stem=True)
    QL = len(query)
    QL_norm = np.linalg.norm(QL)
    query_tfidf, docs_tfidf = get_query_and_docs_tfidf(query, text_stem_index, QL_norm, TEXT_STEM_BUCKET)
    cosine_sim = calc_cosine_similarity(query_tfidf, docs_tfidf, QL_norm)
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
    values = [page_views.get(doc_id, 0) for doc_id in docs_id]
    return values


def get_page_rank(docs_id: List[str]) -> List[float]:
    values = [page_rank.get(doc_id, 0) for doc_id in docs_id]
    return values


# ---- Experimented Search Functions ----

# search using the search function from all the search functions ordered by page rank values
def search_v1(query):
    scores_by_page_rank = []
    doc_ids_title = calc_search_title(query)
    doc_ids_body = calc_search_body(query)
    doc_ids_anchor = calc_search_anchor(query)
    mutual_docs_ids = set(doc_ids_title) | set(doc_ids_body) | set(doc_ids_anchor)
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
    mutual_docs_ids = set(doc_ids_title) | set(doc_ids_body) | set(doc_ids_anchor)
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


# ---------- core functions --------------
def tokenize(text: str, stem=False, bigram=False, expand=False) -> List[str]:
    stemmer = PorterStemmer()
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    corpus_stopwords = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became", 'make']
    stopwords_frozen = frozenset(stopwords.words('english'))
    all_stopwords = stopwords_frozen.union(corpus_stopwords)

    tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in all_stopwords]
    if expand:
        tokens = expand_query_with_synonyms(tokens, word2vec_model)
        tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in all_stopwords]
    if bigram:
        tokens = ngrams(tokens, 2)
    if stem:
        tokens = [stemmer.stem(term) for term in tokens if term not in all_stopwords]
    else:
        tokens = [term for term in tokens if term not in all_stopwords]

    return tokens


def get_posting(index: InvertedIndex, token: str, bucket_name: str):
    posting_list = []
    with closing(MultiFileReader()) as reader:
        locs = [(posting_loc[0], posting_loc[1]) for posting_loc in index.posting_locs[token]]
        b = reader.read(locs, index.df[token] * TUPLE_SIZE, bucket_name)
        for i in range(index.df[token]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
    return posting_list


def binary_search(query: str, index: InvertedIndex, bucket_name: str):
    doc_counts = {}
    for token in set(query):
        token_posting = get_posting(index, token, bucket_name)
        for doc_id, tf in token_posting:
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
    sorted_doc_counts = Counter(doc_counts)
    return [doc_id for doc_id, _ in sorted_doc_counts.most_common()]


def map_doc_id2title(docs_id: List[int]) -> List[Tuple[int, str]]:
    """ Maps a list of document IDs to their corresponding titles."""
    return [(doc_id, id2title[doc_id]) for doc_id in docs_id]


def map_tuple2doc_id(scores: List[Tuple]) -> List[int]:
    """
    Given a list of tuples, where each tuple contains a document ID and a score,
    returns a list of the document IDs.
    """
    return [t[0] for t in scores]


def expand_query_with_synonyms(query_tokens: List[str], word_embedding_model) -> List[str]:
    """
    Expands a query by replacing each token with its most similar words as per the provided word embedding model.
    :param query_tokens: A list of query tokens (strings) to be expanded.
    :param word_embedding_model: A pre-trained word embedding model such as gensim's Word2Vec.
    :return: A list of expanded query tokens.
    """
    expanded_tokens = []
    for token in query_tokens:
        if token in word_embedding_model:
            synonyms = word_embedding_model.most_similar(token, topn=2)
            for synonym in synonyms:
                expanded_tokens.append(synonym[0])
        expanded_tokens.append(token)
    return expanded_tokens


# ----- BM25 ----------

class BM25:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5
    b : float, default 0.75

    index: inverted index
    """

    def __init__(self, index, bucket_name, DL, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.DL = DL
        self.N = len(DL)
        self.AVGDL = sum(DL.values()) / self.N
        self.bucket_name = bucket_name

    def get_candidate_documents_and_scores(self, query):
        candidates = {}
        terms_postings = {}
        for term in np.unique(query):
            list_of_doc = get_posting(self.index, term, self.bucket_name)
            terms_postings[term] = list_of_doc
            normlized_tfidf = [(doc_id, (freq / self.DL[doc_id]) * math.log(len(self.DL) / self.index.df[term], 10)) for
                               doc_id, freq in list_of_doc]

            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf
        return candidates, terms_postings

    def calc_idf(self, list_of_tokens):
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def search(self, query, N=100):
        self.idf = self.calc_idf(query)
        # get relevant canidate docs
        candidate_docs_and_tfidf, terms_postings = self.get_candidate_documents_and_scores(query)
        candidate_docs = [key[0] for key in candidate_docs_and_tfidf.keys()]
        # for each doc add score for query and add to scores dict
        score = [(doc_id, self._score(query, doc_id, terms_postings)) for doc_id in np.unique(candidate_docs)]
        return sorted(score, key=lambda x: x[1], reverse=True)[:N]

    def _score(self, query, doc_id, terms_postings):
        score = 0.0
        doc_len = self.DL[doc_id]

        for term in query:
            if term in self.index.df.keys():
                term_frequencies = dict(terms_postings[term])
                if doc_id in term_frequencies.keys():
                    freq = term_frequencies[doc_id]
                    numerator = self.idf[term] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                    score += (numerator / denominator)
        return score


# --------- BM25 functions ---------

def convert_bm25_score_to_title_and_id(scores: List[Tuple[int, float]]) -> List[Tuple[str, int]]:
    """
    Given a list of tuples representing document scores, where each tuple contains
    a document ID and a BM25 score, returns a list of tuples where each tuple
    contains the title of the document and its ID.
    """
    doc_ids = map_tuple2doc_id(scores)
    return map_doc_id2title(doc_ids)


def calc_bm25(query: str, index, bucket_name, DL) -> List[Tuple[str, int]]:
    """
    Given a query string and a index of tokenized documents, returns a list of tuples where each tuple
    contains the title of the document and its ID, sorted by relevance to the query as determined by BM25 ranking.
    """
    bm25_index = BM25(index, bucket_name, DL)
    bm25_score = bm25_index.search(query, 100)
    return bm25_score


# ----- Metrics ------------
def recall_at_k(true_list, predicted_list, k=40):
    """
      Calculate the recall at k for a given list of true items and a list of predicted items.

      Parameters:
      - true_list (list): a list of true items.
      - predicted_list (list): a list of predicted items.
      - k (int, optional): the number of items to consider in the predicted list (default is 40).

      Returns:
      - float: the recall at k, i.e., the fraction of true items that appear in the top k predicted items.

      Note:
      The recall at k is a measure of the effectiveness of a recommendation system. It represents the
      proportion of the items that the system recommended (i.e., the top k predicted items) that the user
      actually consumed (i.e., the true items).
      """
    # Sort the predicted list in decreasing order of confidence scores
    sorted_pred = sorted(predicted_list)
    pred = sorted_pred[:k]

    # Convert the true list to a set for fast membership checking
    true_set = set(true_list)

    # Use set intersection to calculate the intersection of the two lists
    inter = true_set & set(pred)

    return round(len(inter) / len(true_list), 3)

def recall_at_k(y_true, y_pred, k=40):
    """
      Calculate the recall at k for a given list of true items and a list of predicted items.

      Parameters:
      - y_true (list): a list of true items.
      - y_pred (list): a list of predicted items.
      - k (int, optional): the number of items to consider in the predicted list (default is 40).

      Returns:
      - float: the recall at k, i.e., the fraction of true items that appear in the top k predicted items.
      """
    # Sort the predicted list in decreasing order of confidence scores
    sorted_pred = sorted(y_pred)
    k_pred = sorted_pred[:k]

    # Use set intersection to calculate the intersection of the two lists
    inter = set(y_true) & set(k_pred)

    return round(len(inter) / len(y_true), 3)

def precision_at_k(y_true, y_pred, k=40):
    """
    Calculates the precision at k for the given lists of true and predicted items.
    Precision at k is the fraction of recommended items in the top-k list that are relevant.

    Parameters:
    y_true (list): A list of true/relevant items.
    y_pred (list): A list of predicted items, sorted in order of decreasing relevance.
    k (int, optional): The number of items to consider in the top-k list. Default is 40.

    Returns:
    float: The precision at k, rounded to 3 decimal places.
    """

    # Get the top-k predicted items
    pred = y_pred[:k]

    # Use set intersection to calculate the intersection of the two lists
    inter = set(y_true) & set(pred)

    # Calculate and return the precision at k
    return round(len(inter) / k, 3)

def r_precision(y_true, y_pred):
    """
    Calculates the R-precision for the given lists of true and predicted items.
    R-precision is the fraction of relevant items in the top-R list that are actually relevant.
    R is the total number of relevant items.

    Parameters:
    y_true (list): A list of true/relevant items.
    y_pred (list): A list of predicted items, sorted in order of decreasing relevance.

    Returns:
    float: The R-precision, rounded to 3 decimal places.
    """

    # Get the top-R predicted items, where R is the number of true/relevant items
    R = len(y_true)
    y_pred = y_pred[:R]

    # Use set intersection to calculate the intersection of the two lists
    inter = set(y_true) & set(y_pred)
    # Calculate and return the R-precision
    return round(len(inter) / R, 3)

def reciprocal_rank_at_k(y_true, y_pred, k=40):
    """
    Calculates the reciprocal rank at k for the given lists of true and predicted items.

    Parameters:
    y_true (list): A list of true/relevant items.
    y_pred (list): A list of predicted items, sorted in order of decreasing relevance.
    k (int, optional): The number of items to consider in the top-k list. Default is 40.

    Returns:
    float: The reciprocal rank at k, or 0 if no relevant items are found in the top-k list.
    """

    # Find the rank of the first relevant item in the top-k list
    flag = False
    for i in range(min(k, len(y_pred))):
        if y_pred[i] in y_true:
            k = i + 1
            flag = True
            break

    # If no relevant items are found, return 0
    if not flag:
        return 0

    # Calculate and return the reciprocal rank at k
    return 1 / k

def f_score(y_true, y_pred, k=40):
    """
    Calculates the F-score for the given lists of true and predicted items.

    Parameters:
    y_true (list): A list of true/relevant items.
    y_pred (list): A list of predicted items, sorted in order of decreasing relevance.
    k (int, optional): The number of items to consider in the top-k list. Default is 40.

    Returns:
    float: The F-score, or 0 if either precision or recall is 0.
    """
    # Calculate precision and recall
    precision = precision_at_k(y_true, y_pred, k)
    recall = recall_at_k(y_true, y_pred, k)

    # If either precision or recall is 0, return 0
    if precision + recall == 0:
        return 0

    # Calculate and return the F-score
    return (2 * recall * precision) / (recall + precision)

def average_precision(y_true, y_prod, k=40):
    """
    Calculates the average precision for the given lists of true and predicted items.
    The average precision is the average of the precision scores at each relevant item.

    Parameters:
    y_true (list): A list of true/relevant items.
    y_prod (list): A list of predicted items, sorted in order of decreasing relevance.
    k (int, optional): The number of items to consider in the top-k list. Default is 40.

    Returns:
    float: The average precision, or 0 if no relevant items are found in the top-k list.
    """
    # Get the top-k predicted items and initialize variable
    pred = y_prod[:k]
    relevant = 0
    total = 0

    # Iterate over the top-k predicted items
    for i in range(min(k, len(pred))):
        # If the current item is relevant, increment the relevant item count and add the precision score
        if pred[i] in y_true:
            relevant += 1
            total += relevant / (i + 1)

    # If no relevant items are found, return 0
    if relevant == 0:
        return 0
    # Calculate and return the Average Precision
    return round(total / relevant, 3)

def evaluate_all_metrics(y_true, y_pred, k, print_scores=True):
    """
    Evaluates the given y_pred using various metrics.

    Parameters:
    y_true (list): A list of lists of ground truth documents for each query
    y_pred (list): A list of lists of predicted documents for each query
    k (int): The rank at which to compute the metrics
    print_scores (bool, optional): Whether to print the scores for each metric. Default is True.

    Returns:
    dict: A dictionary mapping from metric names to lists of scores for each query
    """
    metrices = {
        'recall@k': recall_at_k,
        'precision@k': precision_at_k,
        'f_score@k': f_score,
        'r-precision': r_precision,
        'MRR@k': reciprocal_rank_at_k,
        'MAP@k': average_precision,
    }

    scores = {name: [] for name in metrices}

    for name, metric in metrices.items():
        if name == 'r-precision':
            scores[name].append(metric(y_true, y_pred))
        else:
            scores[name].append(metric(y_true, y_pred, k=k))

    if print_scores:
        for name, values in scores.items():
            print(name, sum(values) / len(values))

    return scores

# ---- Evaluating Functions --------

# The Average precision function given for test Search funcs
def average_precision(true_list, predicted_list, k=40):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    precisions = []
    for i, doc_id in enumerate(predicted_list):
        if doc_id in true_set:
            prec = (len(precisions) + 1) / (i + 1)
            precisions.append(prec)
    if len(precisions) == 0:
        return 0.0
    return round(sum(precisions) / len(precisions), 3)

def test_search_funcs_average_precision(func, queries):
    qs_res = []
    for q, true_wids in queries.items():
        duration, ap = None, None
        true_wids = list(map(int, true_wids))
        t_start = tm()
        pred_wids = func(q)
        duration = tm() - t_start
        ap = average_precision(true_wids, pred_wids)
        qs_res.append((q, duration, ap))
    scores = [x[2] for x in qs_res]
    times = [x[1] for x in qs_res]
    mean_score = sum(scores) / len(scores)
    mean_time = sum(times) / len(times)
    print(f"mean score: {mean_score}")
    print(f"mean times: {mean_time}")
    print(f"total time: {sum(times)}")
    print(f'ansewr per each q: {qs_res}')
    return mean_score, mean_time

def mean_scores(dicts):
    mean_scores_dict = defaultdict(float)
    count_scores_dict = defaultdict(int)
    for d in dicts:
        query, time, scores_dict = d
        for key in scores_dict.keys():
            mean_scores_dict[key] += scores_dict[key][0]
            count_scores_dict[key] += 1
    for key in mean_scores_dict.keys():
        mean_scores_dict[key] /= count_scores_dict[key]
    return mean_scores_dict

def test_search_funcs_all_metrices(func, queries):
    qs_res = []
    for q, true_wids in queries.items():
        duration, ap = None, None
        true_wids = list(map(int, true_wids))
        t_start = tm()
        pred_wids = func(q)
        duration = tm() - t_start
        scores = evaluate_all_metrics(true_wids, pred_wids, k=40, print_scores=False)
        qs_res.append((q, duration, scores))
    mean_score = mean_scores(qs_res)
    print(f'Average scores over all queries: {mean_score}')
    return qs_res, mean_score

