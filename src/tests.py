import unittest
from backend import *
from bm25 import *


class MyTestCase(unittest.TestCase):
    def test_something(self):
        print(DL)
        self.assertEqual(True, True)  # add assertion here

    def test_get_relevant_docs_id(self):
        query = """
        Verse 1:
Inverted index, oh so fine
It stores words, one at a time
A little bit of structure, goes a long way
Makes searching lightning fast, hooray!

Chorus:
Inverted index, it's the way to go
Find what you need, in a row
Five words, appear at least twice
Inverted index, it's nice!

Verse 2:
Words are mapped, to their page
So you can find, in a rage
What you're looking for, in a jiff
Inverted index, it's so swift!

Chorus:
Inverted index, it's the way to go
Find what you need, in a row
Five words, appear at least twice
Inverted index, it's nice!

Bridge:
Searching for something, deep in a book
Inverted index, helps you look
For what you need, without delay
Inverted index, makes it all okay!

Chorus:
Inverted index, it's the way to go
Find what you need, in a row
Five words, appear at least twice
Inverted index, it's nice!
        """
        query = tokenize(query)
        ids = get_doc_id_by_count(query, title_index, SRC_PATH)
        print(ids[:5])
        self.assertEqual(True, True)  # add assertion here

    def test_search_body(self):
        query = "similarity laws must obeyed when constructing aeroelastic models of heated high speed aircraft"
        vals = calc_search(query)
        print(vals[:5])
        self.assertEqual(True, True)  # add assertion here

    def test_search_title(self):
        query = "similarity laws must obeyed when constructing aeroelastic models of heated high speed aircraft"
        vals = calc_search_title(query)
        print(vals)
        self.assertEqual(True, True)  # add assertion here

    def test_search_anchor(self):
        query = 'hello world'
        vals = calc_search_anchor(query)
        print(vals)
        self.assertEqual(True, True)  # add assertion here

    def test_id2title(self):
        print(len(id2title.keys()))
        self.assertEqual(True, True)

    def test_bm25(self):
        query = "best marvel movie"
        query = tokenize(query)
        bm25_text = BM25(text_index, DL)
        bm25_title = BM25(title_index, DL)
        bm25_score = bm25_text.search(query)
        bm25_score_title = bm25_title.search(query)
        print(bm25_score)
        print(bm25_score_title)
        self.assertEqual(True, True)

    def test_final_bm25(self):
        query = "best marvel movie"
        scores = calc_bm25(query, text_index)
        print(scores)
        self.assertEqual(True, True)

    def test_eval_retrieve(self):
        evaluate_retrieve()
        self.assertEqual(True, True)

    def test_word_embed(self):
        model = KeyedVectors.load_word2vec_format("../crawl-300d-2M.vec")
        print(type(model))
        self.assertEqual(True, True)

    def test_search(self):
        query = "best marvel movie"
        scores = search(query)
        print(scores)
        self.assertEqual(True, True)



if __name__ == '__main__':
    unittest.main()
