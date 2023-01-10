import pandas as pd
import numpy as np
import json
import re
import nltk
from nltk.corpus import stopwords
from inverted_index_colab import *
import math
from typing import List, Dict, Tuple

pages_path = r'C:\Users\elorberb\PycharmProjects\BGU projects\Search-Engine\src_new\pages'
indices_path = r'C:\Users\elorberb\PycharmProjects\BGU projects\Search-Engine\src_new\indexes'
page_rank_path = r'C:\Users\elorberb\PycharmProjects\BGU projects\Search-Engine\src_new\pages\page_rank.pickle'

title_index = InvertedIndex.read_index(indices_path, 'index_title')
text_index = InvertedIndex.read_index(indices_path, 'index_text')
anchor_index = InvertedIndex.read_index(indices_path, 'index_anchor')
page_rank = pd.read_pickle(page_rank_path)
page_view = InvertedIndex.read_index(pages_path, 'pageviews')
id2title = InvertedIndex.read_index(pages_path, 'id2title')
id2title = {t[0]: t[1] for t in id2title}  # convert to dict

num_of_docs = len(text_index.DL)
DL = text_index.DL

POSTINGS_TITLE = r"C:\Users\elorberb\PycharmProjects\BGU projects\Search-Engine\src_new\title"
POSTINGS_TEXT = r"C:\Users\elorberb\PycharmProjects\BGU projects\Search-Engine\src_new\text"
POSTINGS_ANCHOR = r"C:\Users\elorberb\PycharmProjects\BGU projects\Search-Engine\src_new\anchor"

SRC_PATH = '../src_new/'


def tokenize(text: str) -> List[str]:
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filters stopwords.

    Parameters:
    -----------
    text: string , represting the text to tokenize.
    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    stopwords_frozen = frozenset(stopwords.words('english'))

    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in stopwords_frozen]
    return list_of_tokens


def get_doc_id_by_count(query: str, index: InvertedIndex, root_path: str) -> List[int]:
    """
    Retrieve the documents relevant to the given query by the count of occurrences of the query terms in the document.

    Parameters:
    - query (str): The query to search for.
    - index (InvertedIndex): An index object to search in.
    - root_path (str): The path to the root directory of the repository.

    Returns:
    - List[int]: A list of document IDs sorted in decreasing order of relevance to the query.
    """
    doc_counts = {}
    filtered_tokens = [token for token in query if token in index.df]
    for token in set(filtered_tokens):
        token_posting = get_posting(index, token, root_path)
        for doc_id, tf in token_posting:
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
    sorted_doc_counts = Counter(doc_counts)
    return [doc_id for doc_id, _ in sorted_doc_counts.most_common()]


def get_posting(index: InvertedIndex, token: str, root_path: str) -> List[Tuple[int, int]]:
    """
    Retrieve the posting list for a given term from the inverted index.

    Parameters:
    - index (InvertedIndex): The inverted index object.
    - token (str): The term to retrieve the posting list for.
    - root_path (str): The root path of the inverted index files.

    Returns:
    - list: A list of tuples, where each tuple represents a (doc_id, tf) pair for the given token.
    """
    posting_list = []
    with closing(MultiFileReader()) as reader:
        locs = [(root_path + posting_loc[0], posting_loc[1]) for posting_loc in index.posting_locs[token]]
        b = reader.read(locs, index.df[token] * TUPLE_SIZE)
        for i in range(index.df[token]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
    return posting_list


def calc_idf(index: dict, token: str) -> float:
    """Calculate the IDF of a token in an index."""
    return np.log10(num_of_docs / (index.df[token]))


def calc_tfidf(tf: float, idf: float, dl: float) -> float:
    """Calculate the TF-IDF of a token in a document."""
    return tf * idf / dl


def update_token_tfidf(idf: float, tokens: list, token: str, QL: float) -> float:
    """Update the TF-IDF score of a token in a query."""
    tf = tokens.count(token)
    return calc_tfidf(tf, idf, QL)


def get_query_and_docs_tfidf(tokens: list, index: dict, QL: float, root_path: str) -> tuple:
    """Calculate the TF-IDF scores of tokens in a query and documents."""
    query = {}
    docs = defaultdict(list)
    for token in set(tokens):  # Only calculate the TF-IDF score for each unique token once
        postings = get_posting(index, token, root_path)
        idf = calc_idf(index, token)
        query[token] = update_token_tfidf(idf, tokens, token, QL)
        for doc_id, freq in postings:
            docs[token].append((doc_id, calc_tfidf(freq, idf, DL[doc_id])))
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
    return {doc_id: numerator / DL[doc_id] * QL for doc_id, numerator in cosine_sim.items()}


def get_docs_id_ordered_by_cosine_similarity(cosine_sim: Dict[int, float]) -> List[int]:
    """Returns sorted document IDs based on cosine similarity."""
    return sorted(cosine_sim, key=cosine_sim.get, reverse=True)


def map_doc_id2title(docs_id: List[int]) -> List[Tuple[int, str]]:
    """ Maps a list of document IDs to their corresponding titles."""
    return [(doc_id, id2title[doc_id]) for doc_id in docs_id]


def calc_search_body(query: str, N: int = 100) -> List[Tuple[int, str]]:
    """
    Returns a list of document titles sorted in descending order based on cosine similarity score with the query.

    Parameters:
    - query (str): The search query.
    - index (InvertedIndex): An object representing the inverted index.
    - root_path (str): The root path of the document collection.

    Returns:
    - List[Tuple[int, str]]: A list of tuples, where each tuple contains a document ID and its corresponding title.
    """
    query_tokens = tokenize(query)
    filtered_tokens = [token for token in query_tokens if token in text_index.df]
    QL = len(filtered_tokens)
    query_tfidf, docs_tfidf = get_query_and_docs_tfidf(filtered_tokens, text_index, QL, SRC_PATH)
    cosine_sim = calc_cosine_similarity(query_tfidf, docs_tfidf, QL)
    docs_id = get_docs_id_ordered_by_cosine_similarity(cosine_sim)

    return map_doc_id2title(docs_id)[:N]


def calc_search_title(query: str, N: int = 100) -> List[Tuple[int, str]]:
    """
    Searches for documents with title index containing the given query.
    Calculation made by the count of occurrences of the query terms in the document.

    Parameters:
        query (str): The search query.
        N (int, optional): The maximum number of results to return. Defaults to 100.

    Returns:
        List[Tuple[int, str]]: A list of (document ID, title) tuples.
    """
    query = tokenize(query)
    docs_id = get_doc_id_by_count(query, title_index, SRC_PATH)
    return map_doc_id2title(docs_id)[:N]


def calc_search_anchor(query: str, N: int = 100) -> List[Tuple[int, str]]:
    """
    Searches for documents with title index containing the given query.
    Calculation made by the count of occurrences of the query terms in the document.

    Parameters:
        query (str): The search query.
        N (int, optional): The maximum number of results to return. Defaults to 100.

    Returns:
        List[Tuple[int, str]]: A list of (document ID, title) tuples.
    """
    query = tokenize(query)
    docs_id = get_doc_id_by_count(query, anchor_index, SRC_PATH)
    return map_doc_id2title(docs_id)[:N]

# ----- Pages Functions ------


def get_page_view(docs_id: List[str]) -> List[int]:
    """
    Return a list of page views for the specified document IDs.

    Parameters:
    - doc_ids: list of document IDs (type: List[str])

    Returns:
    - values: list of page views for the specified document IDs (type: List[int])
    """
    values = [page_view[doc_id] for doc_id in docs_id]
    return values


def get_page_rank(docs_id: List[str]) -> List[int]:
    """
    Return a list of page ranks for the specified document IDs.

    Parameters:
    - doc_ids: list of document IDs (type: List[str])

    Returns:
    - values: list of page ranks for the specified document IDs (type: List[int])
    """
    values = [page_rank[doc_id] for doc_id in docs_id]
    return values