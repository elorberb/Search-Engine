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

    def test_search_body(self):
        back = Backend()
        query = "similarity laws must obeyed when constructing aeroelastic models of heated high speed aircraft"
        query = back.tokenize(query)
        score_text = back.search_body(query, 10)

        print(f"Text index scores:\n{score_text}")
        self.assertEqual(True, True)

    def test_get_anchor(self):
        back = Backend()
        query = "hello world"
        query = back.tokenize(query)
        sorted_lst_tuples = back.calc_anchor(query)

        print(sorted_lst_tuples)
        self.assertEqual(True, True)


    def test_map_docId_to_title(self):
        back = Backend()
        query = "hello world"
        query = back.tokenize(query)
        print(back.title_index.term_total)
        self.assertEqual(True, True)


    def test_page_rank(self):
        back = Backend()
        items = back.page_rank.items()
        items = list(items)

        # Print the first 5 key-value pairs
        for i in range(5):
            print(items[i])
        self.assertEqual(True, True)


    def test_page_view(self):
        back = Backend()
        items = back.page_view.items()
        items = list(items)

        # Print the first 5 key-value pairs
        for i in range(5):
            print(items[i])
        self.assertEqual(True, True)





if __name__ == '__main__':
    unittest.main()
