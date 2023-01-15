

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