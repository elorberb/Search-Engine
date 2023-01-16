from inverted_index_gcp import *
from time import time as tm
from metrics import *
from bucket_manipulation import *
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

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"amazing-badge-343010-879c0a90f001.json"

# Bucket names
TEXT_BUCKET = 'body_text_no_stem'
TEXT_STEM_BUCKET = 'body_stem'
TITLE_BUCKET = 'title_bucket4'
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
