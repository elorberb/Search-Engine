from backend import *
from core_functions import *

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