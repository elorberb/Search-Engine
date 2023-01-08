import unittest
from backend import *


class MyTestCase(unittest.TestCase):

    def test_tokenize(self):
        back = Backend()
        actual = back.tokenize("hello world")
        expected = ['hello', 'world']
        self.assertEqual(actual, expected)

    def test_generate_query_tfidf_vector(self):
        back = Backend()
        query = "hello world"
        query_tfidf = back.generate_query_tfidf_vector(query, back.text_index)
        nonzero_indices = query_tfidf.nonzero()
        print(nonzero_indices)
        nonzero_values = query_tfidf[nonzero_indices]
        print(nonzero_values)
        self.assertEqual(True, True)  # add assertion here

    def test_get_candidate_documents_and_scores(self):
        back = Backend()
        query = "hello world"
        words, pls = back.get_posting_iter(back.text_index)
        candidates = back.get_candidate_documents_and_scores(query, back.text_index, words, pls)
        print(candidates)
        self.assertEqual(True, True)

    def test_generate_document_tfidf_matrix(self):
        back = Backend()
        query = "hello world"
        words, pls = back.get_posting_iter(back.text_index)
        df = back.generate_document_tfidf_matrix(query, back.text_index, words, pls)
        nonzero_columns = df.loc[:, df.any()]
        print(nonzero_columns)
        self.assertEqual(True, True)

    def test_cosine_similarity(self):
        back = Backend()
        query = "hello world"
        query_vector = back.generate_query_tfidf_vector(query, back.text_index)  # tfidf of query
        words, pls = back.get_posting_iter(back.text_index)  # get words and posting lists
        doc_matrix = back.generate_document_tfidf_matrix(query, back.text_index, words, pls)  # tfidf of doc
        cosine_sim = back.cosine_similarity(doc_matrix, query_vector)
        print(cosine_sim)
        self.assertEqual(True, True)

    def test_get_topN_score_for_queries(self):
        back = Backend()
        query = "similarity laws must obeyed when constructing aeroelastic models of heated high speed aircraft"
        # score_text = back.get_topN_score_for_queries(query, back.text_index, 10)
        # score_title = back.get_topN_score_for_queries(query, back.title_index, 10)
        score_anchor = back.get_topN_score_for_queries(query, back.anchor_index, 10)

        # print(f"Text index scores:\n{score_text}")
        # print(f"Text index scores:\n{score_title}")
        print(f"Text index scores:\n{score_anchor}")
        self.assertEqual(True, True)





if __name__ == '__main__':
    unittest.main()
