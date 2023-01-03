import unittest
from src.metrics import *


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_sum_values(self):
        c = sum_values(1, 2)
        self.assertEqual(c, 3)


if __name__ == '__main__':
    unittest.main()
