"""
Information Retrieval and Text Analytics
Full project implementation using the 20 Newsgroups dataset

This file is written as a Jupyter-friendly Python script (use with VSCode/Colab/Jupyter).
Cell separators use '# %%' so you can run cell-by-cell in an interactive environment.

Features included:
- Data loading (20 Newsgroups)
- Preprocessing: tokenization, lowercasing, stopword removal, lemmatization, optional stemming
- Vectorization: Count (BoW) and TF-IDF
- Retrieval models: Vector Space Model (cosine similarity), Boolean retrieval (inverted index), BM25 (rank_bm25)
- Query processing pipeline
- Evaluation: Precision, Recall, Average Precision, MAP, Precision@k
- Visualizations: WordCloud, word frequency bar plot, similarity bar chart, LDA topic modelling
- Example queries and result analysis

Requirements (install if needed):
!pip install scikit-learn nltk gensim rank_bm25 wordcloud matplotlib pandas seaborn tqdm
Optional (for interactive LDA viz):
!pip install pyldavis

Run this file as a notebook for best experience.
"""

# %%
# 1. Imports and setup
import os
import math
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from rank_bm25 import BM25Okapi

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from tqdm import tqdm

import gensim
from gensim import corpora
from gensim.models import LdaModel

# Ensure NLTK data
nltk_packages = ["punkt", "wordnet", "omw-1.4", "stopwords"]
for pkg in nltk_packages:
    try:
        nltk.data.find(pkg)
    except Exception:
        nltk.download(pkg)

# %%
# 2. Load 20 Newsgroups dataset
categories = None  # or list of categories or None for all
newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers','footers','quotes'))

docs = newsgroups.data
targets = newsgroups.target
target_names = newsgroups.target_names

print(f"Loaded {len(docs)} documents across {len(target_names)} categories.")

# Create a DataFrame for convenience
df = pd.DataFrame({'text': docs, 'target': targets})

# %%
# 3. Preprocessing pipeline
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

import re

TOKEN_RE = re.compile(r"\b[a-zA-Z]{2,}\b")  # words with only letters, length>=2


def preprocess_text(text: str, do_lemmatize=True, remove_stopwords=True) -> List[str]:
    # Basic cleaning
    if not isinstance(text, str):
        text = ''
    text = text.lower()
    # Keep only words
    tokens = TOKEN_RE.findall(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words]
    if do_lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

# Test preprocess on first document
print('Sample tokens:', preprocess_text(df.loc[0,'text'])[:30])

# Preprocess entire corpus (this can take time)
print('Preprocessing documents...')
corpus_tokens = [preprocess_text(t) for t in tqdm(df['text'])]

# Also create preprocessed text strings for vectorizers
corpus_texts = [' '.join(tokens) for tokens in corpus_tokens]

# %%
# 4. Vectorization: BoW and TF-IDF
# Count Vectorizer (BoW)
count_vectorizer = CountVectorizer(max_df=0.85, min_df=3)
X_count = count_vectorizer.fit_transform(corpus_texts)
print('BoW shape:', X_count.shape)

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.85, min_df=3)
X_tfidf = tfidf_vectorizer.fit_transform(corpus_texts)
print('TF-IDF shape:', X_tfidf.shape)

# Save vocabulary for reference
bow_vocab = count_vectorizer.get_feature_names_out()
tfidf_vocab = tfidf_vectorizer.get_feature_names_out()

# %%
# 5. Build Inverted Index for Boolean Retrieval
# Inverted index maps token -> set(doc_ids)
inverted_index = defaultdict(set)
for doc_id, tokens in enumerate(corpus_tokens):
    for token in set(tokens):
        inverted_index[token].add(doc_id)

print('Inverted index size (unique tokens):', len(inverted_index))

# Boolean query parser (supports AND, OR, NOT, parentheses)
# For simplicity we'll implement a small recursive descent parser for infix boolean queries

import shlex

class BooleanQueryEngine:
    def __init__(self, inverted_index, total_docs):
        self.index = inverted_index
        self.N = total_docs

    def term_docs(self, term):
        return set(self.index.get(term, set()))

    def all_docs(self):
        return set(range(self.N))

    def parse(self, query: str) -> set:
        # Tokenize
        tokens = self.tokenize(query)
        self.tokens = tokens
        self.pos = 0
        result = self._parse_or()
        return result

    def tokenize(self, query):
        # Use shlex to respect quotes; then merge operators
        raw = shlex.split(query)
        toks = []
        for r in raw:
            # split parentheses and operators
            s = r.replace('(', ' ( ').replace(')', ' ) ')
            for part in s.split():
                toks.append(part.upper() if part.upper() in ['AND','OR','NOT','(',')'] else part.lower())
        return toks

    def _parse_or(self):
        left = self._parse_and()
        while self._peek() == 'OR':
            self._eat('OR')
            right = self._parse_and()
            left = left.union(right)
        return left

    def _parse_and(self):
        left = self._parse_not()
        while self._peek() == 'AND':
            self._eat('AND')
            right = self._parse_not()
            left = left.intersection(right)
        return left

    def _parse_not(self):
        if self._peek() == 'NOT':
            self._eat('NOT')
            x = self._parse_atom()
            return self.all_docs() - x
        else:
            return self._parse_atom()

    def _parse_atom(self):
        if self._peek() == '(':
            self._eat('(')
            x = self._parse_or()
            if self._peek() == ')':
                self._eat(')')
            return x
        else:
            term = self._eat_term()
            return self.term_docs(term)

    def _peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _eat(self, token):
        if self._peek() == token:
            self.pos += 1
            return token
        raise ValueError(f"Expected {token} got {self._peek()}")

    def _eat_term(self):
        t = self._peek()
        if t is None:
            raise ValueError('Unexpected end of query')
        self.pos += 1
        return t

boolean_engine = BooleanQueryEngine(inverted_index, len(corpus_texts))

# %%
# 6. BM25 setup
bm25 = BM25Okapi(corpus_tokens)

# %%
# 7. Query processing and retrieval functions

def preprocess_query(query: str) -> List[str]:
    return preprocess_text(query)


def retrieve_vsm(query: str, top_k=10, use_tfidf=True):
    tokens = preprocess_query(query)
    q_text = ' '.join(tokens)
    if use_tfidf:
        q_vec = tfidf_vectorizer.transform([q_text])
        sims = cosine_similarity(q_vec, X_tfidf).flatten()
    else:
        q_vec = count_vectorizer.transform([q_text])
        sims = cosine_similarity(q_vec, X_count).flatten()
    ranked_idx = np.argsort(-sims)[:top_k]
    return list(zip(ranked_idx, sims[ranked_idx]))


def retrieve_boolean(query: str, top_k=50):
    result_set = boolean_engine.parse(query)
    # For reproduction, we return top_k doc ids (no ranking)
    return list(result_set)[:top_k]


def retrieve_bm25(query: str, top_k=10):
    tokens = preprocess_query(query)
    scores = bm25.get_scores(tokens)
    ranked_idx = np.argsort(-scores)[:top_k]
    return list(zip(ranked_idx, scores[ranked_idx]))

# %%
# 8. Helper to display results

def show_results(results, model_name='Model'):
    print(f"Top results ({model_name}):")
    for idx, score in results:
        print('---')
        print(f"Doc ID: {idx} | Score: {score:.4f} | Category: {target_names[targets[idx]]}")
        snippet = df.loc[idx,'text']
        print(snippet[:400].replace('\n',' '))
        print('\n')

# %%
# 9. Example queries and retrieval
queries = [
    'space exploration NASA',
    'computer graphics and rendering',
    'medical patient treatment',
    'religion and philosophy',
    'encryption and cryptography'
]

for q in queries:
    print('\n' + '='*80)
    print('Query:', q)
    vsm_res = retrieve_vsm(q, top_k=5, use_tfidf=True)
    show_results(vsm_res, 'VSM-TFIDF')
    bm25_res = retrieve_bm25(q, top_k=5)
    show_results(bm25_res, 'BM25')
    # boolean example (simple)
    bool_query = ' '.join(preprocess_query(q)[:3])  # use first 3 terms joined with AND
    bool_query = ' AND '.join(preprocess_query(q)[:3])
    print('Boolean query:', bool_query)
    bres = retrieve_boolean(bool_query, top_k=5)
    print('Boolean hits:', bres[:5])

# %%
# 10. Evaluation utilities

# For evaluation we need relevance judgments. 20 Newsgroups doesn't have per-query relevance judgments,
# so we'll create simulated queries from documents: for evaluation purposes we'll pick documents as "query seeds"
# and define relevance as documents in the same target category. This is a common proxy when explicit qrels
# are not available.


def create_eval_queries_from_docs(num_queries=100, seed=42):
    np.random.seed(seed)
    doc_ids = np.random.choice(len(df), size=num_queries, replace=False)
    eval_queries = []
    for doc_id in doc_ids:
        text = df.loc[doc_id,'text']
        # create a short query: top words by TF-IDF from that document
        tokens = preprocess_text(text)
        if len(tokens) < 3:
            continue
        query = ' '.join(tokens[:6])
        relevant_set = set(df[df['target']==df.loc[doc_id,'target']].index.tolist())
        eval_queries.append({'query': query, 'relevant': relevant_set})
    return eval_queries


eval_queries = create_eval_queries_from_docs(200)
print('Prepared', len(eval_queries), 'evaluation queries')

# Metrics

def precision_at_k(retrieved: List[int], relevant: set, k: int) -> float:
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    return sum(1 for r in retrieved_k if r in relevant) / len(retrieved_k)


def recall_at_k(retrieved: List[int], relevant: set, k: int) -> float:
    retrieved_k = set(retrieved[:k])
    if not relevant:
        return 0.0
    return len(retrieved_k.intersection(relevant)) / len(relevant)


def average_precision(retrieved: List[int], relevant: set) -> float:
    if not relevant:
        return 0.0
    score = 0.0
    num_hits = 0.0
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            num_hits += 1.0
            score += num_hits / i
    return score / len(relevant)


def mean_average_precision(run_results: List[Tuple[List[int], set]]) -> float:
    aps = [average_precision(retrieved, relevant) for retrieved, relevant in run_results]
    return float(np.mean(aps))

# Evaluate function that runs a model on eval queries and computes MAP, P@10, Recall@10

def evaluate_model(eval_queries, model='vsm', top_k=10):
    results = []
    P_at_k = []
    R_at_k = []
    for q in tqdm(eval_queries):
        query = q['query']
        relevant = q['relevant']
        if model == 'vsm':
            res = retrieve_vsm(query, top_k=top_k, use_tfidf=True)
            retrieved = [doc for doc, score in res]
        elif model == 'bm25':
            res = retrieve_bm25(query, top_k=top_k)
            retrieved = [doc for doc, score in res]
        elif model == 'bow':
            res = retrieve_vsm(query, top_k=top_k, use_tfidf=False)
            retrieved = [doc for doc, score in res]
        else:
            raise ValueError('Unknown model')
        results.append((retrieved, relevant))
        P_at_k.append(precision_at_k(retrieved, relevant, top_k))
        R_at_k.append(recall_at_k(retrieved, relevant, top_k))
    MAP = mean_average_precision(results)
    return {'MAP': MAP, 'P@k': np.mean(P_at_k), 'R@k': np.mean(R_at_k)}

# Warning: evaluation can take time. We'll evaluate on a subset for speed.
eval_subset = eval_queries[:100]
print('Evaluating VSM...')
metrics_vsm = evaluate_model(eval_subset, model='vsm', top_k=10)
print('VSM metrics:', metrics_vsm)
print('Evaluating BM25...')
metrics_bm25 = evaluate_model(eval_subset, model='bm25', top_k=10)
print('BM25 metrics:', metrics_bm25)

# %%
# 11. Visualization

# WordCloud for a category

def wordcloud_for_category(cat_index, max_words=100):
    cat_docs = df[df['target']==cat_index]['text'].tolist()
    tokens = []
    for d in cat_docs:
        tokens.extend(preprocess_text(d))
    text = ' '.join(tokens)
    wc = WordCloud(width=800, height=400, background_color='white', max_words=max_words).generate(text)
    plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'WordCloud for category: {target_names[cat_index]}')
    plt.show()

# Example: show wordcloud for first 3 categories
for i in range(3):
    wordcloud_for_category(i)

# Frequency distribution of top words across the corpus
all_tokens = [token for tokens in corpus_tokens for token in tokens]
freq = Counter(all_tokens)
most_common = freq.most_common(30)
words, counts = zip(*most_common)
plt.figure(figsize=(12,6))
sns.barplot(x=list(counts), y=list(words))
plt.title('Top 30 tokens by frequency (preprocessed)')
plt.xlabel('Count')
plt.show()

# Similarity bar chart for a sample query
sample_q = queries[0]
vsm_res = retrieve_vsm(sample_q, top_k=10, use_tfidf=True)
ids, scores = zip(*vsm_res)
plt.figure(figsize=(10,6))
plt.barh(range(len(ids)), list(scores)[::-1])
plt.yticks(range(len(ids)), [f"{i} ({target_names[targets[i]]})" for i in ids][::-1])
plt.xlabel('Cosine similarity')
plt.title(f'Similarity scores for query: {sample_q}')
plt.show()

# %%
# 12. Topic modeling with LDA (Gensim)
# Use tokenized corpus and dictionary/corpus for gensim

dictionary = corpora.Dictionary(corpus_tokens)
dictionary.filter_extremes(no_below=10, no_above=0.5)
corpus_gensim = [dictionary.doc2bow(text) for text in corpus_tokens]

# Train LDA
num_topics = 10
lda = LdaModel(corpus=corpus_gensim, id2word=dictionary, num_topics=num_topics, passes=5, random_state=42)

# Print topics
for i in range(num_topics):
    print('Topic', i, lda.print_topic(i, topn=10))

# Optionally visualize with pyLDAvis (if installed)
try:
    import pyLDAvis.gensim_models as gensimvis
    import pyLDAvis
    vis = gensimvis.prepare(lda, corpus_gensim, dictionary)
    pyLDAvis.display(vis)
except Exception as e:
    print('pyLDAvis not available or failed to render:', e)

# %%
# 13. Results & Analysis (printed summary)
print('\nRESULTS SUMMARY')
print('VSM (TF-IDF) metrics on eval subset:', metrics_vsm)
print('BM25 metrics on eval subset:', metrics_bm25)
print('\nObservations:')
print('- BM25 often performs comparably or better than vanilla TF-IDF cosine similarity for short natural-language queries.')
print('- Boolean retrieval returns exact matches and is useful for structured queries, but lacks ranking.')
print('- Preprocessing choices (lemmatization, stopword removal, min_df/max_df) substantially affect results.')

# %%
# 14. Save artifacts (optional)
os.makedirs('artifacts', exist_ok=True)
# Save vectorizers
import pickle
with open('artifacts/count_vectorizer.pkl', 'wb') as f:
    pickle.dump(count_vectorizer, f)
with open('artifacts/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
with open('artifacts/bm25.pkl', 'wb') as f:
    pickle.dump(bm25, f)
with open('artifacts/lda_model.pkl', 'wb') as f:
    pickle.dump(lda, f)
print('Saved artifacts to ./artifacts')

# %%
# 15. Conclusion and next steps
"""
Conclusion:
- This notebook demonstrates a full IR pipeline using the 20 Newsgroups dataset.
- It includes preprocessing, three retrieval models, evaluation (using category-based proxy relevance),
  visualizations, and topic modeling.

Next steps / Improvements:
- Use human-labeled query relevance (qrels) for more accurate evaluation.
- Add semantic search using sentence embeddings (SBERT / SentenceTransformers).
- Build a simple web UI (Flask/FastAPI + React) for interactive search.
- Improve boolean query parsing to support phrase queries and proximity operators.
"""
print('Project complete. Modify cells to experiment further.')
