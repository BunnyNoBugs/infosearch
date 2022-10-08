`TfidfSearch` class in  `hw2.py` implements TF-IDF search.
It has the following functionality:

1. Create a search object: `search = TfidfSearch()`
2. Load the desired corpus using `search.load_corpus(PATH_TO_CORPUS)`
3. Perform the search using the function which ranks the docs in the corpus depending on the
   query: `search.rank_by_query(QUERY)`