import json
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from nltk.corpus import stopwords

STOPWORDS = stopwords.words('russian')


class BM25Search:
    def __init__(self):
        """
        Create a search object.
        """
        self._corpus = None
        self._vectorizer = CountVectorizer(stop_words=STOPWORDS)

    def load_corpus(self, path, max_samples=50000):
        """
        Load the corpus from a given path.

        :param path: path to the corpus
        :param max_samples: samples limit in the corpus
        """
        with open(path, encoding='utf-8') as f:
            corpus = [json.loads(x) for x in list(f)]
        self._corpus = []
        for sample in corpus:
            max_value = 0
            best_answer = None
            for answer in sample['answers']:
                answer_text = answer['text']
                answer_value = answer['author_rating']['value']
                if answer_value:
                    answer_value = int(answer_value)
                    if (answer_value > max_value) and answer_text:
                        max_value = answer_value
                        best_answer = answer_text
            if best_answer:
                self._corpus.append(best_answer)
            if len(self._corpus) == max_samples:
                self._corpus = np.array(self._corpus)
                break

        self._index_corpus()

    def _index_corpus(self, k=2.0, b=0.75):
        """
        Index the corpus with BM25 function.

        :param k: k constant for BM25 function
        :param b: b constant for BM25 function
        """
        tf = self._vectorizer.fit_transform(self._corpus)

        idf_vectorizer = TfidfVectorizer(use_idf=True, stop_words=STOPWORDS)
        idf_vectorizer.fit(self._corpus)
        idf = idf_vectorizer.idf_

        len_d = tf.sum(axis=1)
        avgdl = len_d.mean()
        # numerator
        A = tf.multiply(idf * (k + 1)).tocsr()
        # denominator
        tf_binary = csr_matrix((np.ones_like(tf.nonzero()[0]), tf.nonzero()))
        B_1 = tf_binary.multiply(k * (1 - b + b * len_d / avgdl)).tocsr()
        B = tf + B_1
        bm25 = csr_matrix((A.data / B.data, tf.nonzero()))
        self._index = bm25

    def rank_by_query(self, query: str, results_limit=10) -> np:
        """
        Rank the docs of the corpus according to the query.

        :param query: text of the query
        :param results_limit: number of docs to return
        :return: top N docs in the corpus
        """
        query_vectorized = self._vectorizer.transform([query]).T
        scores = (self._index * query_vectorized).toarray().ravel()
        sorted_scores_idx = scores.argsort(axis=0)[::-1]
        ranked_corpus = self._corpus[sorted_scores_idx]
        return ranked_corpus[:results_limit]


def main():
    search = BM25Search()
    search.load_corpus('questions_about_love.jsonl', max_samples=50000)
    print(search.rank_by_query('две одна'))


if __name__ == '__main__':
    main()
