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
    def __init__(self):
        self._corpus = None
        self._vectorizer = TfidfVectorizer()

    def _index_text_collection(self, collection):
        return self._vectorizer.transform(collection)

    @staticmethod
    def _preprocess_text(text):
        doc = nlp(text)
        lemmas = [token.lemma_ for token in doc if token.is_alpha if token.pos_ != 'PRON' if
                  token.text not in STOPWORDS]
        lemmas = [lemma for lemma in lemmas if lemma not in STOPWORDS]

        return ' '.join(lemmas)

    def load_corpus(self, path):
        corpus = []
        for root, dirs, files in os.walk(path):
            if '.ipynb_checkpoints' not in root:
                for name in files:
                    filepath = os.path.join(root, name)
                    with open(filepath, encoding='utf-8') as f:
                        text = f.read()
                        corpus.append(text)
        self._corpus = corpus
        self._index_corpus()

    def _index_corpus(self):
        start_time = time.time()
        self._preprocessed_corpus = [self._preprocess_text(text) for text in self._corpus]
        print(f'Corpus preprocessing took {time.time() - start_time} seconds')  # 83 seconds when testing
        self._preprocessed_corpus = np.array(self._preprocessed_corpus)
        self._vectorizer.fit(self._preprocessed_corpus)
        self._index = self._index_text_collection(self._preprocessed_corpus)

    def _compute_similarities(self, query):
        # use linear_kernel as TfidfVectorizer returns l2-normalized vectors
        return linear_kernel(self._index, query)

    def rank_by_query(self, query: str) -> List[str]:
        query_preprocessed = self._preprocess_text(query)
        query_indexed = self._index_text_collection([query_preprocessed])
        similarities = self._compute_similarities(query_indexed)[:, 0]
        return np.take_along_axis(np.array(self._corpus), np.argsort(-similarities), 0)


def main():
    search = TfidfSearch()
    search.load_corpus('../friends-data')
    print(search.rank_by_query('Лайза Минелли')[:2])


if __name__ == '__main__':
    main()
