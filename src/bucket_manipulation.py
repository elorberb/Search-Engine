from inverted_index_gcp import InvertedIndex, MultiFileReader
import struct
import time
import os
from google.cloud import storage
from contextlib import closing
import pickle as pkl


class ReadBucketData:

    def __init__(self, bucket_name):
        self.TUPLE_SIZE = 6
        self.TUPLE_SIZE_BODY = 8
        self.TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
        self.client = storage.Client()
        self.bucket = self.client.get_bucket(bucket_name)

    def download_from_buck(self, source, dest):
        blob = self.bucket.get_blob(source)
        blob.download_to_filename(dest)
        return dest

    def get_pkl_file(self, source, dest):
        if dest not in os.listdir("."):
            self.download_from_buck(source, dest)
        with open(dest, "rb") as f:
            return pkl.load(f)

    def get_inverted_index(self, source_idx, dest_file):
        self.download_from_buck(source_idx, dest_file)
        return InvertedIndex().read_index(".", dest_file.split(".")[0])
