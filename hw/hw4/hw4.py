import pickle
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, List
from functools import lru_cache
from tqdm import tqdm


class BertSearch:
    """
    Implements search based on BERT embeddings.
    """

    def __init__(self):
        """
        Create a search object.
        """
        self._questions = None
        self._answers = None
        self._questions_embeddings = None
        self._answers_embeddings = None

    def load_corpus(self, path):
        """
        Load the corpus from a given path.

        :param path: path to the corpus
        """
        with open(path, encoding='utf-8') as f:
            corpus = json.load(f)
        self._questions = np.array(list(corpus.keys()))
        self._answers = np.array(list(corpus.values()))

    def load_embeddings(self, questions_path, answers_path):
        """
        Load precomputed embeddings.

        :param questions_path: path to questions embeddings
        :param answers_path: path to answers embeddings
        """
        with open(questions_path, 'rb') as f:
            self._questions_embeddings = pickle.load(f)
        with open(answers_path, 'rb') as f:
            self._answers_embeddings = pickle.load(f)
        pass

    def rank_by_query(self, query: str, results_limit=5) -> Optional[List[str]]:
        """
        Rank the docs of the corpus according to the query.

        :param query: text of the query
        :param results_limit: number of docs to return
        :return: top N docs in the corpus
        """
        if query not in self._questions:
            print('Sorry, BERT search only accepts queries that are present in the questions.')
            return None
        else:
            question_idx = np.nonzero(self._questions == query)
            query_embedding = self._questions_embeddings[question_idx]
            scores = cosine_similarity(self._answers_embeddings, query_embedding)
            sorted_scores_idx = scores.argsort(axis=0)[::-1]
            ranked_answers = self._answers[sorted_scores_idx]
            return ranked_answers[:results_limit]

    @lru_cache
    def compute_quality_metric(self, top_k=5, n_test_samples=100):
        """
        Compute search quality metric: hit rate at top-k results.

        :param top_k: number of search results to compute metric with
        :param n_test_samples: number of randomly sampled test questions to compute metric with
        :return: hit rate at top-k results
        """
        print('Computing quality metric...')
        rng = np.random.default_rng()
        test_idxs = rng.choice(len(self._questions), n_test_samples, replace=False)
        hits = 0
        for idx in tqdm(test_idxs):
            question = self._questions[idx]
            true_answer = self._answers[idx]
            search_results = self.rank_by_query(question, results_limit=top_k)
            hits += int(true_answer in search_results)
        return hits / n_test_samples


def main():
    search = BertSearch()
    search.load_corpus('data/corpus_50000.json')
    search.load_embeddings('data/questions_embeddings.pickle', 'data/answers_embeddings.pickle')
    print(search.rank_by_query('С мужчиной не видимся'))
    top_k = 5
    print(f'Search quality: {search.compute_quality_metric(top_k=top_k, n_test_samples=1000)} hit rate at top-{top_k}')


if __name__ == '__main__':
    main()
