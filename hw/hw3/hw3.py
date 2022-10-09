import json


class BM25Search:
    def __init__(self):
        """Create a search object."""
        # Not sure what an __init__ docstring should look like!
        self._corpus = []
        self._vectorizer = None

    def load_corpus(self, path, max_samples=50000):
        """Load the corpus from a given path."""
        with open(path, encoding='utf-8') as f:
            corpus = [json.loads(x) for x in list(f)[:max_samples]]
        for sample in corpus:
            max_value = 0
            best_answer = None
            for answer in sample['answers']:
                value = answer['author_rating']['value']
                if value:  # todo: more elegant?
                    value = int(value)
                    if value > max_value:
                        max_value = value
                        best_answer = answer['text']
            self._corpus.append(best_answer)

        self._index_corpus()

    def _index_corpus(self):
        """Create an index of the loaded corpus."""
        pass


def main():
    search = BM25Search()
    search.load_corpus('questions_about_love.jsonl')


if __name__ == '__main__':
    main()
