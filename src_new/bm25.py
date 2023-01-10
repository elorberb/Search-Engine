import math
from itertools import chain
import time
from backend_new import *


class BM25:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5
    b : float, default 0.75

    index: inverted index
    """

    def __init__(self, index, DL, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.DL = DL
        self.N = len(DL)
        self.AVGDL = sum(DL.values()) / self.N
        self.words, self.pls = zip(*self.index.posting_lists_iter())

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

    def calc_idf(self, list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
        key: term
        value: bm25 idf score
        """
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def search(self, query, N=100):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
        key: query_id
        value: a ranked list of pairs (doc_id, score) in the length of N.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        self.idf = self.calc_idf(query)
        # get relevant canidate docs
        candidate_docs_and_tfidf = self.get_candidate_documents_and_scores(query, self.index, self.words, self.pls)
        candidate_docs = [key[0] for key in candidate_docs_and_tfidf.keys()]
        # for each doc add score for query and add to scores dict
        score = [(doc_id, self._score(query, doc_id)) for doc_id in np.unique(candidate_docs)]
        return sorted(score, key=lambda x: x[1], reverse=True)[:N]

    def _score(self, query, doc_id):
        """
        This function calculate the bm25 score for given query and document.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        score = 0.0
        doc_len = self.DL[doc_id]

        for term in query:
            if term in self.index.term_total.keys():
                term_frequencies = dict(self.pls[self.words.index(term)])
                if doc_id in term_frequencies.keys():
                    freq = term_frequencies[doc_id]
                    numerator = self.idf[term] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                    score += (numerator / denominator)
        return score
