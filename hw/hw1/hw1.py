import time
import os
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from typing import List

from nltk.corpus import stopwords

STOPWORDS = stopwords.words('russian')

import spacy

nlp = spacy.load('ru_core_news_lg', disable=['ner', 'parser'])


class InvertedIndex:
    def __init__(self, corpus):
        self._corpus = corpus
        self._vectorizer = CountVectorizer()
        self.index = self._vectorizer.fit_transform(self._corpus)
        self._words = np.array(self._vectorizer.get_feature_names_out())
        self._total_count = np.asarray(self.index.sum(axis=0)).squeeze()

    def get_most_frequent_word(self):
        return self._words[self._total_count.argmax()]

    def get_random_least_frequent_words(self, n=1):
        least_frequent_idx = np.where(self._total_count == self._total_count.min())[0]
        return self._words[np.random.choice(least_frequent_idx, n)]

    def get_shared_words(self):
        return self._words[np.all(self.index.toarray(), axis=0)]

    def count_occurrences(self, words_variants: List[List[str]]):
        occurrences = {}
        for word in words_variants:
            variants_idx = []
            for variant in word:
                variants_idx.append(self._vectorizer.vocabulary_.get(variant))
            occurrences[tuple(word)] = self.index[:, variants_idx].sum()
        return occurrences


def preprocess_text(text):
    doc = nlp(text)
    lemmata = [token.lemma_ for token in doc if token.is_alpha if token.pos_ != 'PRON' if token.text not in STOPWORDS]
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
    print('Start preprocessing the corpus...')
    corpus = [preprocess_text(text) for text in corpus]
    print(f'Corpus preprocessing took {time.time() - start_time} seconds')  # 83 sec when testing

    index = InvertedIndex(corpus)

    print(f'The most frequent word in the corpus is: {index.get_most_frequent_word()}')
    print(f'Random least frequent words in the corpus: {", ".join(index.get_random_least_frequent_words(2))}')
    shared_words = index.get_shared_words()
    print(f'{len(shared_words)} words are shared by all documents in the corpus: {", ".join(shared_words)}')
    characters = [
        ['моника', 'мон'],
        ['рэйчел', 'рэйч'],
        ['чендлер', 'чен'],
        ['фиби', 'фибс'],
        ['росс'],
        ['джоуи', 'джои']
    ]
    print(f"Characters' names occurrences: {index.count_occurrences(characters)}. "
          f"Ross turns out to be the most popular.")


if __name__ == '__main__':
    main()
