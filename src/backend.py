import pickle
import gzip
import pandas as pd
import numpy as np
import math
from google.cloud import storage
from pandas._config.config import reset_option
from evaluation import Evaluation
import json
from nltk.corpus import stopwords

nltk.download('stopwords')
from time import time as t
import re
import os
from inverted_index_gcp import InvertedIndex, MultiFileReader
import pandas as pd
import numpy as np
from collections import defaultdict

os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\ofi1\Pycharm_Projects\BGU_Projects\Search-Engine\src\data-retrieval-project-d9f8e61f8a29 (1).json"

BUCKET_POSTINGS_BODY = 'postings_body/postings_gcp'

BUCKET_POSTINGS_TITLE = "postings_title/postings_gcp"


class Backend:

    def __init__(self):
        """
           Put super indexes and relevent files in main memory to save time during queries
           """

        # self.body_index = pickle.load(open("data/bodyindex.pkl", "rb"))
        # self.body_stem_index = pickle.load(open("data/body_stem_index.pkl", "rb"))
        self.title_index = pickle.load(open("src/indexes/index_title.pkl", "rb"))
        # self.anchor_index = pickle.load(open("data/anchor_index.pkl", "rb"))
        # self.page_rank = pd.read_csv(gzip.open('data/page_rank.csv.gz', 'rb'))
        # self.page_view = pickle.load(open('data/page_view.pkl', 'rb'))  # October Page view has missing documents
        # self.page_view_12 = pickle.load(
        #     open('data/pageviews-202112.pkl', 'rb'))  # Use December 2021 page views to include missing docs
        # self.anchor_double_index = pickle.load(open('data/anchor_double_index.pkl', 'rb'))
        # self.bucket_name = 'bodyindex'
        # self.client = storage.Client()
        # self.bucket = self.client.get_bucket(self.bucket_name)

    def preprocess(self, query):
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

    def tf_idf(self, posting_list, DL, N):
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


    def cosine_similarity(self, documents, query):
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

    def bm25(self, query):
        pass

    def main_search(self, query):
        pass

    def get_body(self, query):

        """
        Params:
        ------
        query - list of tokens. i.e: ["hello", "world"]
        ======
        Iterates over each words and downloads its relevent postings list.
        Calculates the tfidf of each word in each document
        and calculates the tfidf of each word in query
        Also, we remove words that their tfidf in the query_tfidf is smaller than some epsilon from the highest tfidf.

        Output:
        Dataframe: docs as columns and row as score
        List of remaining words
        """

        query_doc_tfidf = {}
        query_idf = {}
        query_tf = defaultdict(int)
        query_tfidf = {}

        for w, posting_list in self.body_index.posting_lists_iter(BUCKET_POSTINGS_BODY,
                                                                  query):  # Iterate over each posting list
            query_doc_tfidf[w] = self.tf_idf(posting_list, self.body_index.DL)  # save tfidf for each doc
            query_idf[w] = 1 + math.log(self.N / len(posting_list), 10)  # save udf of each word in query

        if len(query_idf) == 0:
            return pd.DataFrame({}), []

        for w in query_idf:
            query_tf[w] += 1  # Calculate term frequency of words in query

        for w, freq in query_tf.items():
            score = freq * query_idf[w]
            query_tfidf[w] = [score]  # save tfidf of each word in query

        max_tfidf = max(set().union(*query_tfidf.values()))  # get the highest tfidf in query
        epsilon = 0.9  # filters words with too small tfidf

        df_query_tfidf = pd.DataFrame(query_tfidf)
        words = df_query_tfidf.columns

        df_query_tfidf = df_query_tfidf.loc[:,
                         df_query_tfidf.iloc[0] > (max_tfidf - epsilon)]  # filter irrelevent words
        words = list(filter(lambda i: i not in df_query_tfidf.columns, words))  # Extract the words that were removed

        df = pd.DataFrame(query_doc_tfidf)
        df.drop(words, axis=1, inplace=True)  # filter irrelevent words

        # df = pd.concat([df_query_tfidf, df]).T
        df = df.fillna(value=0)
        df_query_tfidf = df_query_tfidf.fillna(value=0)

        cosine_sim_body = self.cosine_similarity(df, df_query_tfidf)  # calculate cosine similarity
        return cosine_sim_body, df.columns

    def get_title(self, query):

        """
        Params:
        ------
        query: list of tokens. i.e.: ["hello", "world"]
        ======
        Sorting all relevent docs of query with the title index
        Output:
        ------
        List of doc_ids sorted by anchor relevence
        """

        res = self.get_kind(query, self.title_index, BUCKET_POSTINGS_TITLE)
        res = sorted(res.items(), key=lambda x: x[1], reverse=True)
        return [x for x, y in res]

    def get_anchor(self, query):
        pass

    def get_view(self, query):
        pass

    def get_page_rank(self):
        pass


if __name__ == '__main__':
    back = backend('bodyindex')
    inver = back.get_inverted_index('data/bodyindex.pkl', 'bodyindex.pkl')
    print(inver.get_posting_list('the'))
