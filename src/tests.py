import unittest
from backend import *
from bm25 import *
from time import time
from metrics import *
from search_functions import *
from retrievel_functions import *


class MyTestCase(unittest.TestCase):

    def test_search_body(self):
        query = "best marvel movie"
        t_start = time()
        vals = calc_search_body(query)
        duration = time() - t_start
        print(vals)
        print(f'duration: {duration}')
        self.assertEqual(True, True)  # add assertion here

    def test_search_title(self):
        query = "best marvel movie"
        t_start = time()
        vals = calc_search_title(query)
        duration = time() - t_start
        print(vals)
        print(f'duration: {duration}')
        self.assertEqual(True, True)  # add assertion here

    def test_search_anchor(self):
        query = 'best marvel movie'
        vals = calc_search_anchor(query)
        print(vals)
        self.assertEqual(True, True)  # add assertion here

    def test_calc_search_body(self):
        with open('../extra_files/queries_train.json', 'rt') as f:
            queries = json.load(f)
        qs_res = []
        for q, true_wids in queries.items():
            duration, ap = None, None
            true_wids = list(map(int, true_wids))
            t_start = time()
            pred_wids = calc_search_body(q, text_index)
            duration = time() - t_start
            ap = average_precision(true_wids, pred_wids)
            qs_res.append((q, duration, ap))
        second_values = [x[2] for x in qs_res]
        mean = sum(second_values) / len(second_values)
        print(mean)
        print(qs_res)

    def test_calc_search_body_stem(self):
        with open('../extra_files/queries_train.json', 'rt') as f:
            queries = json.load(f)
        qs_res = []
        for q, true_wids in queries.items():
            duration, ap = None, None
            true_wids = list(map(int, true_wids))
            t_start = time()
            pred_wids = calc_search_body_stem(q)
            duration = time() - t_start
            ap = average_precision(true_wids, pred_wids)
            qs_res.append((q, duration, ap))
        second_values = [x[2] for x in qs_res]
        mean = sum(second_values) / len(second_values)
        print(mean)
        print(qs_res)

    def test_bm25(self):
        query = "best marvel movie"
        t_start = time()
        res = calc_bm25(query, text_index, TEXT_BUCKET, DL)
        duration = time() - t_start
        print(res)
        print(duration)
        self.assertEqual(True, True)  # add assertion here

    def test_calc_search_anchor(self):
        with open('../extra_files/queries_train.json', 'rt') as f:
            queries = json.load(f)
        qs_res = []
        for q, true_wids in queries.items():
            duration, ap = None, None
            true_wids = list(map(int, true_wids))
            t_start = time()
            pred_wids = calc_search_anchor(q)
            duration = time() - t_start
            ap = average_precision(true_wids, pred_wids)
            qs_res.append((q, duration, ap))
        second_values = [x[2] for x in qs_res]
        mean = sum(second_values) / len(second_values)
        print(mean)
        print(qs_res)
        self.assertEqual(True, True)  # add assertion here

    def test_calc_search_title(self):
        with open('../extra_files/queries_train.json', 'rt') as f:
            queries = json.load(f)
        qs_res = []
        for q, true_wids in queries.items():
            duration, ap = None, None
            true_wids = list(map(int, true_wids))
            t_start = time()
            pred_wids = calc_search_title(q)
            duration = time() - t_start
            ap = average_precision(true_wids, pred_wids)
            qs_res.append((q, duration, ap))
        second_values = [x[2] for x in qs_res]
        mean = sum(second_values) / len(second_values)
        print(mean)
        print(qs_res)
        self.assertEqual(True, True)  # add assertion here

    def test_calc_search_anchor_double(self):
        with open('../extra_files/queries_train.json', 'rt') as f:
            queries = json.load(f)
        qs_res = []
        for q, true_wids in queries.items():
            duration, ap = None, None
            true_wids = list(map(int, true_wids))
            t_start = time()
            pred_wids = calc_search_anchor_double(q)
            duration = time() - t_start
            ap = average_precision(true_wids, pred_wids)
            qs_res.append((q, duration, ap))
        second_values = [x[2] for x in qs_res]
        mean = sum(second_values) / len(second_values)
        print(mean)
        print(qs_res)
        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
