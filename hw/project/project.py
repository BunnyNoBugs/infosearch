from typing import Literal
import time
import json
import torch
import numpy as np
import pickle
import requests
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

from hf_token import hf_token
from nltk.corpus import stopwords

STOPWORDS = stopwords.words('russian')


class _TfidfEngine:
    def __init__(self, corpus):
        self._corpus = corpus
        self._vectorizer = TfidfVectorizer(stop_words=STOPWORDS)
        self._index = None
        self._index_corpus()

    def _index_corpus(self):
        self._index = self._vectorizer.fit_transform(self._corpus)

    def rank_by_query(self, query: str):
        query_vectorized = self._vectorizer.transform([query]).T
        scores = (self._index * query_vectorized).toarray().ravel()
        sorted_scores_idx = scores.argsort(axis=0)[::-1]
        ranked_corpus = self._corpus[sorted_scores_idx]
        return ranked_corpus


class _BM25Engine:
    def __init__(self, corpus):
        self._corpus = corpus
        self._vectorizer = CountVectorizer(stop_words=STOPWORDS)
        self._index = None
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

    def rank_by_query(self, query: str):
        """
        Rank the docs of the corpus according to the query.

        :param query: text of the query
        :return: ranked docs of the corpus
        """
        query_vectorized = self._vectorizer.transform([query]).T
        scores = (self._index * query_vectorized).toarray().ravel()
        sorted_scores_idx = scores.argsort(axis=0)[::-1]
        ranked_corpus = self._corpus[sorted_scores_idx]
        return ranked_corpus


class _BertEngine:
    def __init__(self, corpus, embeddings_path):
        self._model_id = 'sberbank-ai/sbert_large_nlu_ru'
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._corpus = corpus
        self._embeddings = None
        self.load_embeddings(embeddings_path)

    def load_embeddings(self, embeddings_path):
        """
        Load precomputed embeddings.

        :param embeddings_path: path to embeddings
        """
        with open(embeddings_path, 'rb') as f:
            self._embeddings = pickle.load(f)

    def _get_query_embedding(self, query: str):
        def api_call(texts):
            response = requests.post(api_url, headers=headers,
                                     json={"inputs": texts, "options": {"wait_for_model": True}})
            return response.json()

        # Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(token_embeddings, attention_mask):
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self._model_id}"
        headers = {"Authorization": f"Bearer {hf_token}"}
        api_output = api_call([query])
        token_embeddings = torch.Tensor(api_output[0])
        encoded_query = self._tokenizer(query, padding=True, truncation=True, max_length=512, return_tensors='pt')
        query_embedding = mean_pooling(token_embeddings, encoded_query['attention_mask'])
        return query_embedding

    def rank_by_query(self, query: str):
        """
        Rank the docs of the corpus according to the query.

        :param query: text of the query
        :return: ranked docs of the corpus
        """
        query_embedding = self._get_query_embedding(query)
        scores = cosine_similarity(self._embeddings, query_embedding)
        sorted_scores_idx = scores.argsort(axis=0)[::-1]
        ranked_corpus = self._corpus[sorted_scores_idx]
        return ranked_corpus


class Search:
    """
    todo: add doc
    """

    def __init__(self, corpus_path, embeddings_path):
        self._corpus = None
        self._method = None
        self._engine = None
        self._load_corpus(corpus_path)
        self._tfidf_engine = _TfidfEngine(self._corpus)
        self._bm25_engine = _BM25Engine(self._corpus)
        self._bert_engine = _BertEngine(self._corpus, embeddings_path)
        self._SEARCH_ENGINES = {
            'tfidf': self._tfidf_engine,
            'bm25': self._bm25_engine,
            'bert': self._bert_engine
        }
        self.method = 'bm25'

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, search_method: Literal['tfidf', 'bm25', 'bert']):
        if search_method in self._SEARCH_ENGINES:
            self._method = search_method
            self._engine = self._SEARCH_ENGINES[self._method]
        else:
            print('Please choose from supported search methods.')

    def _load_corpus(self, path):
        """
        Load the corpus from a given path.

        :param path: path to the corpus
        """
        with open(path, encoding='utf-8') as f:
            corpus = json.load(f)
        self._corpus = np.array(list(corpus.values()))

    def rank_by_query(self, query: str, results_limit=10):
        start_time = time.time()
        search_results = self._engine.rank_by_query(query)
        print(f'Search runtime: {round(time.time() - start_time, 2)} sec')
        return search_results[:results_limit]


def main():
    search = Search(corpus_path='../hw4/data/corpus_50000.json',
                    embeddings_path='../hw4/data/answers_embeddings.pickle')
    search.method = 'bert'
    print(search.rank_by_query('С мужчиной не видимся'))


if __name__ == '__main__':
    main()
