import pandas as pd
import numpy as np
from collections import defaultdict

def preprocess(query):
    """
    Preprocesses the given query by performing the following steps:
    1. Extract words from the query using a regular expression
    2. Removes stopwords from the list of extracted words
    3. Returns the resulting list of words

    Parameters:
    query (str): The query to be preprocessed

    Returns:
    list: A list of words extracted from the query
    """

    # Set of English stopwords
    english_stopwords = frozenset(stopwords.words('english'))

    # Additional stopwords specific to the corpus
    corpus_stopwords = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became", "make", "good", "best", "worst"]

    # Union of all stopwords
    all_stopwords = english_stopwords.union(corpus_stopwords)

    # Regular expression for matching words
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

    # Extract words from query and remove stopwords
    tokens = [token.group() for token in RE_WORD.finditer(query.lower()) if token.group() not in all_stopwords]

    return tokens


def tf_idf(posting_list, DL, N):
    """
    Calculate the term frequency-inverse document frequency (TF-IDF) for a word.

    Parameters:
    posting_list -- a list of (doc_id, frequency) tuples for a word
    DL -- a dictionary of {doc_id: doc_length} for all documents
    N -- the total number of documents

    Returns:
    a dictionary of {doc_id: tfidf_score} for the given word
    """
    tfidf = defaultdict(int)
    idf = 1 + math.log(N / len(posting_list), 10)
    for doc_id, freq in posting_list:
        tf = freq / (1 + math.log(DL[doc_id], 10))
        tfidf[doc_id] += tf * idf
    return tfidf


def cosine_similarity(documents, query):
    """
    Calculate the cosine similarity between a list of documents and a query document.

    Parameters
    ----------
    documents: pandas.DataFrame
        A dataframe containing the list of documents tfidf scores.
    query: pandas.DataFrame
        A dataframe containing the query tfidf scores.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing the cosine similarity scores
        for each document in `documents`, sorted in descending order.
    """
    # Calculate numerator - dot product between documents and query
    numerator = documents.dot(query.T.to_numpy()).T

    # Calculate denominator part of formula
    query_norm = np.sqrt(query.dot(query.T.to_numpy()).iat[0, 0])
    docs_norm = np.sqrt(documents.T.dot(documents.to_numpy()).iat[0, 0])
    denominator = float(query_norm * docs_norm)

    # Calculate cosine similarity scores
    scores = numerator.div(denominator)

    # Sort results in descending order
    sorted_scores = scores.sort_values(by=scores.index[0], ascending=False, axis=1)

    # Remove documents with too small score
    epsilon = 0.01
    filtered_scores = sorted_scores.loc[:, sorted_scores.iloc[0] > epsilon]

    return filtered_scores
