import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
from typing import List

from nltk.corpus import stopwords

STOPWORDS = stopwords.words('russian')

import spacy

nlp = spacy.load('ru_core_news_lg', disable=['ner', 'parser'])


class TfidfSearch:
    def __init__(self, corpus):
        self._corpus = np.array(corpus)
        self._vectorizer = TfidfVectorizer().fit(self._corpus)
        self._index = self._index_text_collection(self._corpus)

    def _index_text_collection(self, collection):
        return self._vectorizer.transform(collection)

    def _compute_similarities(self, query):
        return linear_kernel(self._index, query)

    def rank_by_query(self, query: str) -> List[str]:
        query_indexed = self._index_text_collection([query])
        similarities = self._compute_similarities(query_indexed)[:, 0]
        return np.take_along_axis(self._corpus, np.argsort(-similarities), 0)


def preprocess_text(text):
    doc = nlp(text)
    lemmata = [token.lemma_ for token in doc if token.is_alpha if token.pos_ != 'PRON' if
               token.text not in STOPWORDS]
    lemmata = [lemma for lemma in lemmata if lemma not in STOPWORDS]

    return ' '.join(lemmata)


def main():
    corpus = []
    for root, dirs, files in os.walk('friends-data'):
        if '.ipynb_checkpoints' not in root:
            for name in files:
                filepath = os.path.join(root, name)
                with open(filepath, encoding='utf-8') as f:
                    text = f.read()
                    corpus.append(text)
    assert len(corpus) == 165  # check for corpus size

    start_time = time.time()
    corpus = [preprocess_text(text) for text in corpus]
    print(f'Corpus preprocessing took {time.time() - start_time} seconds')

    search = TfidfSearch(corpus)
    print(search.rank_by_query('лайза минелли')[:2])


if __name__ == '__main__':
    main()
