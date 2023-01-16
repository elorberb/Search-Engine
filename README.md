# Search-Engine Project

This project aims to create a search engine that operates on Wikipedia data, leveraging techniques like cosine similarity and tf-idf, as well as utilizing pre-trained models like word2vec.
The project is divided into several modules, each one with a unique responsibility that contributes to the overall functionality. The modules are:
###  <u>backend.py</u>
code for downloading the indexes and other relevant files from our BUCKETS.
###  <u>core_functions.py</u>
general functions for use in the modules of the rest of the project, such as tokenizing, mapping to title, and more.
###  <u>retrieve_functions.py</u>
all the functions that are intended for calculating tf-idf for queries and documents and calculating cosine similarity in order to build the search body.
###  <u>inverted_index_gcp.py</u>
the code for building the skeleton of the inverted index object.
###  <u>search_functions.py</u>
all the search functions we built in order to retrieve documents according to the various indexes and methods we used.
###  <u>metrics.py</u>
a module that contains all the evaluation functions we built in order to test the search functions.
###  <u>bucket_manipulation.py</u>
a module whose purpose is to connect to Buckets and download the indexes and other files from them in order to perform the retrieval.
###  <u>bm25.py</u>
module for building the bm25 search function.
###  <u>search_frontend.py</u>
the module given to us in order to do a test against a local server.


In addition to the modules, we added a tests folder where the following files are located:

### <u> tests.py</u>
Unittest module where we did the tests for the functions we built.
### <u> testing_engine.ipynb</u>
notebook in which we tested our code in GCP.
