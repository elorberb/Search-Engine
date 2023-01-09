import pickle
import gzip
import math
import json
import nltk
from nltk.corpus import stopwords
import re
import os
from src.inverted_index_colab import *
import pandas as pd
import numpy as np
from collections import defaultdict
from collections import Counter


class Backend:

    def __init__(self):
        """
           Put super indexes and relevent files in main memory to save time during queries
           """

        indices_path = r'C:\Users\elorberb\PycharmProjects\BGU projects\Search-Engine\src\indexes'

        page_rank_path = r'C:\Users\elorberb\PycharmProjects\BGU projects\Search-Engine\src\pages\page_rank.pickle'
        pages_path = r'C:\Users\elorberb\PycharmProjects\BGU projects\Search-Engine\src\pages'

        inverted = InvertedIndex()
        self.title_index = inverted.read_index(indices_path, 'index_title')
        self.text_index = inverted.read_index(indices_path, 'index_text')
        self.anchor_index = inverted.read_index(indices_path, 'index_anchor')
        self.page_rank = pd.read_pickle(page_rank_path)
        self.page_view = inverted.read_index(pages_path, 'pageviews')
        self.id2title = inverted.read_index(pages_path, 'id2title')

        self.N = len(self.text_index.DL)
        self.DL = self.text_index.DL

    @staticmethod
    def tokenize(text):
        """
        This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

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

    def generate_query_tfidf_vector(self, query, index):
        """
        Generate a vector representing the query. Each entry within this vector represents a tfidf score.
        The terms representing the query will be the unique terms in the index.

        We will use tfidf on the query as well.
        For calculation of IDF, use log with base 10.
        tf will be normalized based on the length of the query.

        Parameters:
        -----------
        query: list of tokens in the query
        index: inverted index loaded from the corresponding files.

        Returns:
        -----------
        vectorized query with tfidf scores
        """
        epsilon = .0000001
        total_vocab_size = len(index.term_total)
        Q = np.zeros(total_vocab_size)
        term_vector = list(index.term_total.keys())
        counter = Counter(query)
        for token in np.unique(query):
            if token in index.term_total.keys():  # avoid terms that do not appear in the index.
                tf = counter[token] / len(query)  # term frequency divded by the length of the query
                df = index.df[token]
                idf = math.log(self.N / (df + epsilon), 10)  # smoothing
                try:
                    ind = term_vector.index(token)
                    Q[ind] = tf * idf
                except:
                    pass
        return Q

    @staticmethod
    def get_posting_iter(index):
        """
        This function returning the iterator working with posting list.

        Parameters:
        ----------
        index: inverted index
        """
        words, pls = zip(*index.posting_lists_iter())
        return words, pls

    def get_candidate_documents_and_scores(self, query, index, words, pls):
        """
        Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
        and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
        Then it will populate the dictionary 'candidates.'
        For calculation of IDF, use log with base 10.
        tf will be normalized based on the length of the document.

        Parameters:
        -----------
        query: list of tokens in the query
        index: inverted index loaded from the corresponding files.
        words,pls: iterator for working with posting.

        Returns:
        -----------
        dictionary of candidates. In the following format:
        key: pair (doc_id,term)
        value: tfidf score.
        """
        candidates = {}
        for term in np.unique(query):
            if term in words:
                list_of_doc = pls[words.index(term)]
                normlized_tfidf = [(doc_id, (freq / self.DL[doc_id]) * math.log(len(self.DL) / index.df[term], 10)) for
                                   doc_id, freq in list_of_doc]

                for doc_id, tfidf in normlized_tfidf:
                    candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

        return candidates

    def generate_document_tfidf_matrix(self, query, index, words, pls):
        """
        Generate a DataFrame `D` of tfidf scores for a given query.
        Rows will be the documents candidates for a given query
        Columns will be the unique terms in the index.
        The value for a given document and term will be its tfidf score.

        Parameters:
        -----------
        query: list of tokens in the query

        index: inverted index loaded from the corresponding files.
        words,pls: iterator for working with posting.

        Returns:
        -----------
        DataFrame of tfidf scores.
        """
        total_vocab_size = len(index.term_total)
        candidates_scores = self.get_candidate_documents_and_scores(query, index, words, pls)
        unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
        D = np.zeros((len(unique_candidates), total_vocab_size))
        D = pd.DataFrame(D)

        D.index = unique_candidates
        D.columns = index.term_total.keys()

        for key in candidates_scores:
            tfidf = candidates_scores[key]
            doc_id, term = key
            D.loc[doc_id][term] = tfidf

        return D

    @staticmethod
    def cosine_similarity(D, Q):
        """
        Calculate the cosine similarity for each candidate document.
        Generate a dictionary of cosine similarity scores
        key: doc_id
        value: cosine similarity score

        Parameters:
        -----------
        D: DataFrame of tfidf scores.
        Q: vectorized query with tfidf scores

        Returns:
        -----------
        dictionary of cosine similarity score as follows:
        key: document id (e.g., doc_id)
        value: cosine similarty score.
        """
        scores = {}
        for i, doc in D.iterrows():
            scores[i] = np.dot(doc, Q) / (np.sqrt(np.dot(doc, doc)) * np.sqrt(np.dot(Q, Q)))
        return scores

    def search_body(self, query, N=3):
        """
        Generate a dictionary that gathers for every query its topN tfidf score.

        Parameters:
        -----------
        - query: list of tokens in the query
        N: Integer. How many documents to retrieve. This argument is passed to the topN function. By default N = 3, for the topN function.

        Returns:
        -----------
        return: a dictionary of queries and topN pairs as follows:
                                                            key: query_id
                                                            value: list of pairs in the following format:(doc_id, score).
        """
        words, pls = self.get_posting_iter(self.text_index)  # get words and posting lists
        doc_matrix = self.generate_document_tfidf_matrix(query, self.text_index, words, pls)  # tfidf of doc
        query_vector = self.generate_query_tfidf_vector(query, self.text_index)  # tfidf of query
        sim_dict = self.cosine_similarity(doc_matrix, query_vector)  # cosine similarity
        top_n = sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1],
                       reverse=True)[:N]
        return top_n  # return score for top N

    def count_words_in_index(self, query, kind='title'):
        """
        Calculate anchor terms for a given query.

        Parameters:
        - query: list of tokens in the query

        Returns:
        - sorted_lst_tuples: list of tuples (token, count) sorted by count in descending order
        """
        # Use a Counter to store the term_total counts
        if kind == 'anchor':
            term_total = Counter(self.anchor_index.term_total)
        else:
            term_total = Counter(self.title_index.term_total)
        # Create a list of tuples (token, count) for the tokens in the query
        lst_tuples = [(token, term_total[token]) for token in query if token in term_total]
        # Sort the list of tuples by count in descending order
        sorted_lst_tuples = sorted(lst_tuples, key=lambda x: x[1], reverse=True)

        return sorted_lst_tuples

    def search_title(self, query, N=3):
        """
        Search the title index for the given query and return the top N results.

        Parameters:
        - query: list of tokens in the query
        - N: number of results to return (default is 3)

        Returns:
        - top_results: list of tuples (token, count) for the top N results
        """
        return self.count_words_in_index(query, kind='title')[:N]

    def search_anchor(self, query, N=3):
        """
        Search the anchor index for the given query and return the top N results.

        Parameters:
        - query: list of tokens in the query
        - N: number of results to return (default is 3)

        Returns:
        - top_results: list of tuples (token, count) for the top N results
        """
        return self.count_words_in_index(query, kind='anchor')[:N]

    def get_page_view(self, doc_ids):
        """
        Return a list of page views for the specified document IDs.

        Parameters:
        - doc_ids: list of document IDs

        Returns:
        - values: list of page views for the specified document IDs
        """
        values = [self.page_view[doc_id] for doc_id in doc_ids]
        return values

    def get_page_rank(self, doc_ids):
        """
        Return a list of page ranks for the specified document IDs.

        Parameters:
        - doc_ids: list of document IDs

        Returns:
        - values: list of page ranks for the specified document IDs
        """
        values = [self.page_rank[doc_id] for doc_id in doc_ids]
        return values
