import json


def prepare_corpus(path, save_path, max_samples=50000):
    """Prepare corpus of questions and answers from original corpus and save it.

    :param path: path to original corpus
    :param save_path: path to save prepared corpus
    :param max_samples: maximum samples in the prepared corpus
    """
    with open(path, encoding='utf-8') as f:
        corpus = [json.loads(x) for x in list(f)]
    prepared_corpus = {}
    for sample in corpus:
        question = sample['question']
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
            prepared_corpus[question] = best_answer
        if len(prepared_corpus) == max_samples:
            break

    with open(save_path, mode='w', encoding='utf-8') as f:
        json.dump(prepared_corpus, f, ensure_ascii=False)


def main():
    prepare_corpus('../hw3/questions_about_love.jsonl', 'corpus_50000.json')


if __name__ == '__main__':
    main()
