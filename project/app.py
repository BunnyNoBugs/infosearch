import streamlit as st
from search import Search

search = Search(corpus_path='../hw/hw4/data/corpus_50000.json',
                embeddings_path='../hw/hw4/data/answers_embeddings.pickle')
method2id = {
    'TF-IDF': 'tfidf',
    'BM25': 'bm25',
    'BERT': 'bert'
}

st.title('Search for love!')
st.image('images/bender.jpg')
query = st.text_input('Please type in your query...')
search_method = st.selectbox('Please choose search method:', method2id)

method_id = method2id[search_method]
search.method = method_id
if query:
    search_results, search_runtime = search.rank_by_query(query)
    results_formatted = [f'{idx + 1}. {result}' for result, idx in zip(search_results, range(len(search_results)))]
    st.text(f'Search runtime: {round(search_runtime, 2)} sec')
    st.text('\n'.join(results_formatted))
