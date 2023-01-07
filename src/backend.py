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


class Backend:

    def __init__(self):
        """
           Put super indexes and relevent files in main memory to save time during queries
           """

        title_index_path = r'C:\Users\elorberb\PycharmProjects\BGU projects\Search-Engine\src\indexes\index_title.pkl'
        text_index_path = r'C:\Users\elorberb\PycharmProjects\BGU projects\Search-Engine\src\indexes\index_text.pkl'
        anchor_index_path = r'C:\Users\elorberb\PycharmProjects\BGU projects\Search-Engine\src\indexes\index_anchor.pkl'

        self.title_index = pickle.load(open(title_index_path, "rb"))
        self.text_index = pickle.load(open(text_index_path, "rb"))
        self.anchor_index = pickle.load(open(anchor_index_path, "rb"))

        self.N = len(self.text_index.DL)

    def tokenize(self, text):
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
        query: (str). They query to be processed.
        index: inverted index loaded from the corresponding files.

        Returns:
        -----------
        vectorized query with tfidf scores
        """
        query = self.tokenize(query)
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

    def get_posting_iter(self, index):
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
        query: (str). They query to be processed.
        index:        inverted index loaded from the corresponding files.

        words,pls: iterator for working with posting.

        Returns:
        -----------
        dictionary of candidates. In the following format:
                                                                   key: pair (doc_id,term)
                                                                   value: tfidf score.
        """
        query = self.tokenize(query)
        candidates = {}
        for term in np.unique(query):
            if term in words:
                list_of_doc = pls[words.index(term)]
                normlized_tfidf = [(doc_id, (freq / index.DL[doc_id]) * math.log(len(index.DL) / index.df[term], 10)) for
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
        query: (str). They query to be processed.

        index:           inverted index loaded from the corresponding files.


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

    def cosine_similarity(self, D, Q):
        """
        Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
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
        # YOUR CODE HERE
        scores = {}
        for i, doc in D.iterrows():
            scores[i] = np.dot(doc, Q) / (np.sqrt(np.dot(doc, doc)) * np.sqrt(np.dot(Q, Q)))
        return scores

    def get_top_n(self, sim_dict, N=3):
        """
        Sort and return the highest N documents according to the cosine similarity score.
        Generate a dictionary of cosine similarity scores

        Parameters:
        -----------
        sim_dict: a dictionary of similarity score as follows:
                                                                    key: document id (e.g., doc_id)
                                                                    value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

        N: Integer (how many documents to retrieve). By default N = 3

        Returns:
        -----------
        a ranked list of pairs (doc_id, score) in the length of N.
        """

        return sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]

    def get_topN_score_for_queries(self, query, index, N=3):
        """
        Generate a dictionary that gathers for every query its topN score.

        Parameters:
        -----------
        queries_to_search: a dictionary of queries as follows:
                                                            key: query_id
                                                            value: list of tokens.
        index:           inverted index loaded from the corresponding files.
        N: Integer. How many documents to retrieve. This argument is passed to the topN function. By default N = 3, for the topN function.

        Returns:
        -----------
        return: a dictionary of queries and topN pairs as follows:
                                                            key: query_id
                                                            value: list of pairs in the following format:(doc_id, score).
        """
        # YOUR CODE HERE

        words, pls = self.get_posting_iter(index)  # get words and posting lists
        doc_matrix = self.generate_document_tfidf_matrix(query, index, words, pls)  # tfidf of doc
        query_vector = self.generate_query_tfidf_vector(query, index)  # tfidf of query
        return self.get_top_n(self.cosine_similarity(doc_matrix, query_vector), N)  # return score for top N














