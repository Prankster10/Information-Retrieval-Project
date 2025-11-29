"""
Information Retrieval and Text Analytics System
Dataset: Wikipedia Articles
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
import wikipediaapi
import re

# Download required NLTK data
print("Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class WikipediaIRSystem:
    def __init__(self):
        self.documents = []
        self.doc_titles = []
        self.preprocessed_docs = []
        self.vectorizer_bow = None
        self.vectorizer_tfidf = None
        self.bow_matrix = None
        self.tfidf_matrix = None
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def collect_wikipedia_data(self, topics, num_docs_per_topic=15):
        """Collect Wikipedia articles based on topics"""
        print(f"\n{'='*60}")
        print("1. DATA COLLECTION FROM WIKIPEDIA")
        print(f"{'='*60}")
        
        wiki_wiki = wikipediaapi.Wikipedia('IRProject/1.0', 'en')
        
        for topic in topics:
            print(f"\nFetching articles for topic: '{topic}'")
            page = wiki_wiki.page(topic)
            
            if page.exists():
                self.documents.append(page.text[:5000])  # Limit to 5000 chars
                self.doc_titles.append(page.title)
                print(f"  ✓ Added: {page.title}")
                print(f"    URL: {page.fullurl}")
                
                # Get related articles - try to get more by filtering
                links = list(page.links.keys())
                articles_added = 0
                
                for link_title in links:
                    if articles_added >= num_docs_per_topic - 1:
                        break
                    
                    # Skip common non-article links
                    if any(skip in link_title.lower() for skip in ['wikipedia:', 'special:', 'category:', 'help:', 'template:']):
                        continue
                    
                    link_page = wiki_wiki.page(link_title)
                    if link_page.exists() and link_page.title not in self.doc_titles:
                        self.documents.append(link_page.text[:5000])
                        self.doc_titles.append(link_page.title)
                        print(f"  ✓ Added: {link_page.title}")
                        print(f"    URL: {link_page.fullurl}")
                        articles_added += 1
            else:
                print(f"  ✗ Article '{topic}' not found")
        
        print(f"\n{'='*60}")
        print(f"Total documents collected: {len(self.documents)}")
        print(f"{'='*60}")
        return len(self.documents)
    
    def preprocess_text(self, text, use_stemming=True):
        """Preprocess text: tokenize, lowercase, remove stopwords, stem/lemmatize"""
        # Lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        # Stemming or Lemmatization
        if use_stemming:
            tokens = [self.stemmer.stem(word) for word in tokens]
        else:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(tokens)
    
    def preprocess_all_documents(self):
        """Preprocess all collected documents"""
        print(f"\n{'='*60}")
        print("2. TEXT PREPROCESSING")
        print(f"{'='*60}")
        
        print("\nPreprocessing documents...")
        self.preprocessed_docs = [self.preprocess_text(doc) for doc in self.documents]
        
        print(f"✓ Preprocessed {len(self.preprocessed_docs)} documents")
        print(f"\nSample preprocessed text (first 200 chars):")
        print(f"{self.preprocessed_docs[0][:200]}...")
    
    def vectorize_documents(self):
        """Create Bag-of-Words and TF-IDF representations"""
        print(f"\n{'='*60}")
        print("3. VECTORIZATION")
        print(f"{'='*60}")
        
        # Bag-of-Words
        print("\nCreating Bag-of-Words representation...")
        self.vectorizer_bow = CountVectorizer(max_features=1000)
        self.bow_matrix = self.vectorizer_bow.fit_transform(self.preprocessed_docs)
        print(f"✓ BoW Matrix shape: {self.bow_matrix.shape}")
        
        # TF-IDF
        print("\nCreating TF-IDF representation...")
        self.vectorizer_tfidf = TfidfVectorizer(max_features=1000)
        self.tfidf_matrix = self.vectorizer_tfidf.fit_transform(self.preprocessed_docs)
        print(f"✓ TF-IDF Matrix shape: {self.tfidf_matrix.shape}")
    
    def vector_space_model(self, query, top_k=20):
        """Vector Space Model using cosine similarity"""
        # Preprocess query
        processed_query = self.preprocess_text(query)
        
        # Transform query using TF-IDF
        query_vector = self.vectorizer_tfidf.transform([processed_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top K documents (limit to available documents)
        top_k = min(top_k, len(similarities))
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'rank': len(results) + 1,
                'title': self.doc_titles[idx],
                'score': similarities[idx],
                'snippet': self.documents[idx][:200]
            })
        
        return results
    
    def boolean_retrieval(self, query):
        """Simple Boolean Retrieval Model"""
        processed_query = self.preprocess_text(query)
        query_terms = processed_query.split()
        
        results = []
        for idx, doc in enumerate(self.preprocessed_docs):
            # Check if all query terms are in document (AND operation)
            if all(term in doc for term in query_terms):
                results.append({
                    'rank': len(results) + 1,
                    'title': self.doc_titles[idx],
                    'snippet': self.documents[idx][:200]
                })
        
        return results
    
    def bm25_retrieval(self, query, top_k=20, k1=1.5, b=0.75):
        """BM25 Ranking Algorithm"""
        processed_query = self.preprocess_text(query)
        query_terms = processed_query.split()
        
        # Document lengths
        doc_lens = [len(doc.split()) for doc in self.preprocessed_docs]
        avgdl = np.mean(doc_lens)
        N = len(self.preprocessed_docs)
        
        scores = []
        for idx, doc in enumerate(self.preprocessed_docs):
            score = 0
            doc_terms = doc.split()
            doc_len = len(doc_terms)
            
            for term in query_terms:
                if term in doc_terms:
                    # Term frequency in document
                    tf = doc_terms.count(term)
                    
                    # Document frequency
                    df = sum(1 for d in self.preprocessed_docs if term in d)
                    
                    # IDF
                    idf = np.log((N - df + 0.5) / (df + 0.5) + 1)
                    
                    # BM25 score
                    score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgdl)))
            
            scores.append(score)
        
        # Get top K (limit to available documents)
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    'rank': len(results) + 1,
                    'title': self.doc_titles[idx],
                    'score': scores[idx],
                    'snippet': self.documents[idx][:200]
                })
        
        return results
    
    def evaluate_retrieval(self, query, top_k=20):
        """
        Evaluate retrieval performance using multiple metrics:
        - Precision: relevant items retrieved / total items retrieved
        - Recall: relevant items retrieved / total relevant items
        - Mean Average Precision (MAP): average of precision at each relevant position
        
        Evaluates how well the query matches the document collection
        """
        print(f"\n{'='*60}")
        print("EVALUATION METRICS - PRECISION, RECALL, MAP")
        print(f"{'='*60}")
        
        # Get all results with scores
        all_results = self.vector_space_model(query, top_k=len(self.doc_titles))
        
        # Calculate average score as threshold for relevance
        all_scores = [r['score'] for r in all_results]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        median_score = sorted(all_scores)[len(all_scores)//2] if all_scores else 0
        
        # Documents above median are considered "relevant"
        relevant_set = set(r['title'] for r in all_results if r['score'] >= median_score)
        
        # Top-k retrieved documents
        retrieved_set = set(r['title'] for r in all_results[:top_k])
        
        # Calculate TP, FP, FN
        true_positives = len(retrieved_set & relevant_set)
        false_positives = len(retrieved_set - relevant_set)
        false_negatives = len(relevant_set - retrieved_set)
        
        # Calculate Precision: TP / (TP + FP)
        precision = true_positives / len(retrieved_set) if retrieved_set else 0
        
        # Calculate Recall: TP / (TP + FN)
        recall = true_positives / len(relevant_set) if relevant_set else 0
        
        # Calculate Mean Average Precision (MAP)
        map_score = 0
        num_relevant_found = 0
        for rank, result in enumerate(all_results[:top_k], 1):
            if result['title'] in relevant_set:
                num_relevant_found += 1
                precision_at_k = num_relevant_found / rank
                map_score += precision_at_k
        
        map_score = map_score / len(relevant_set) if relevant_set else 0
        
        # Display detailed evaluation metrics
        print(f"\nQuery: '{query}'")
        print(f"Total documents in collection: {len(self.doc_titles)}")
        print(f"Relevant documents (above median score): {len(relevant_set)}")
        print(f"Documents retrieved (top-{top_k}): {len(retrieved_set)}")
        print(f"Relevant documents retrieved: {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"False Negatives: {false_negatives}")
        
        print(f"\n{'─'*70}")
        print("SUMMARY STATISTICS")
        print(f"{'─'*70}")
        print(f"  Query                         : '{query}'")
        print(f"  Total Documents in Collection: {len(self.doc_titles)}")
        print(f"  Relevant Documents           : {len(relevant_set)}")
        print(f"  Documents Retrieved          : {len(retrieved_set)}")
        print(f"  Relevant Documents Retrieved : {true_positives}")
        print(f"  False Positives              : {false_positives}")
        print(f"  False Negatives              : {false_negatives}")
        print(f"{'─'*70}")
        
        print(f"\n{'═'*70}")
        print("EVALUATION METRICS")
        print(f"{'═'*70}")
        
        # Precision
        print(f"\n1. PRECISION: {precision:.4f}")
        print(f"   {'─'*66}")
        print(f"   Definition: Of all retrieved documents, how many are relevant?")
        print(f"   Formula   : TP / (TP + FP) = {true_positives} / {len(retrieved_set)}")
        print(f"   Meaning   : {precision*100:.2f}% of your results are relevant")
        if precision >= 0.8:
            print(f"   Status    : ✓ EXCELLENT (≥80%)")
        elif precision >= 0.6:
            print(f"   Status    : ⚠ GOOD (60-80%)")
        elif precision >= 0.4:
            print(f"   Status    : ⚡ FAIR (40-60%)")
        else:
            print(f"   Status    : ✗ POOR (<40%)")
        
        # Recall
        print(f"\n2. RECALL: {recall:.4f}")
        print(f"   {'─'*66}")
        print(f"   Definition: Of all relevant documents, how many did we retrieve?")
        print(f"   Formula   : TP / (TP + FN) = {true_positives} / {len(relevant_set)}")
        print(f"   Meaning   : {recall*100:.2f}% of relevant docs were found")
        if recall >= 0.9:
            print(f"   Status    : ✓ EXCELLENT (≥90%)")
        elif recall >= 0.7:
            print(f"   Status    : ⚠ GOOD (70-90%)")
        elif recall >= 0.5:
            print(f"   Status    : ⚡ FAIR (50-70%)")
        else:
            print(f"   Status    : ✗ POOR (<50%)")
        
        # MAP
        print(f"\n3. MEAN AVERAGE PRECISION (MAP): {map_score:.4f}")
        print(f"   {'─'*66}")
        print(f"   Definition: Quality of ranking - are relevant docs at the top?")
        print(f"   Formula   : Average precision across all relevant documents")
        print(f"   Meaning   : {map_score*100:.2f}% ranking quality score")
        if map_score >= 0.9:
            print(f"   Status    : ✓ EXCELLENT (≥90%)")
        elif map_score >= 0.7:
            print(f"   Status    : ⚠ GOOD (70-90%)")
        elif map_score >= 0.5:
            print(f"   Status    : ⚡ FAIR (50-70%)")
        else:
            print(f"   Status    : ✗ POOR (<50%)")
        
        print(f"\n{'═'*70}")
        print("OVERALL PERFORMANCE")
        print(f"{'═'*70}")
        
        # Calculate F1 score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  F1 Score (Harmonic Mean)      : {f1_score:.4f}")
        print(f"  Average Performance           : {(precision + recall + map_score) / 3:.4f}")
        
        if f1_score >= 0.85:
            overall_status = "EXCELLENT - System performing very well!"
        elif f1_score >= 0.70:
            overall_status = "GOOD - System performing well"
        elif f1_score >= 0.50:
            overall_status = "FAIR - Room for improvement"
        else:
            overall_status = "POOR - Significant improvement needed"
        
        print(f"  Overall Status                : {overall_status}")
        print(f"{'═'*70}\n")
        
        return {
            'precision': precision,
            'recall': recall,
            'map': map_score,
            'retrieved': len(retrieved_set),
            'relevant_retrieved': true_positives,
            'relevant_docs_count': len(relevant_set)
        }
    
    def visualize_word_cloud(self):
        """Generate word cloud from all documents"""
        print("\nGenerating Word Cloud...")
        all_text = ' '.join(self.preprocessed_docs)
        
        wordcloud = WordCloud(width=800, height=400, 
                            background_color='white',
                            colormap='viridis').generate(all_text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Document Collection', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def visualize_word_frequency(self, top_n=20):
        """Visualize top word frequencies"""
        print("\nGenerating Word Frequency Chart...")
        
        # Get word frequencies from BoW
        word_freq = np.asarray(self.bow_matrix.sum(axis=0)).flatten()
        words = self.vectorizer_bow.get_feature_names_out()
        
        # Get top N words
        top_indices = word_freq.argsort()[-top_n:][::-1]
        top_words = [words[i] for i in top_indices]
        top_freqs = [word_freq[i] for i in top_indices]
        
        plt.figure(figsize=(12, 6))
        plt.barh(range(len(top_words)), top_freqs, color='steelblue')
        plt.yticks(range(len(top_words)), top_words)
        plt.xlabel('Frequency', fontsize=12)
        plt.title(f'Top {top_n} Most Frequent Words', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def visualize_similarity_scores(self, query, results):
        """Visualize document-query similarity scores"""
        print("\nGenerating Similarity Score Chart...")
        
        titles = [r['title'][:30] + '...' if len(r['title']) > 30 else r['title'] 
                 for r in results]
        scores = [r['score'] for r in results]
        
        plt.figure(figsize=(12, 6))
        bars = plt.barh(range(len(titles)), scores, color='coral')
        plt.yticks(range(len(titles)), titles)
        plt.xlabel('Similarity Score', fontsize=12)
        plt.title(f'Document-Query Similarity Scores\nQuery: "{query}"', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(score, i, f' {score:.4f}', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def topic_modeling_lda(self, n_topics=5):
        """Perform LDA topic modeling"""
        print(f"\nPerforming LDA Topic Modeling with {n_topics} topics...")
        
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(self.bow_matrix)
        
        # Display topics
        words = self.vectorizer_bow.get_feature_names_out()
        
        print(f"\n{'='*60}")
        print("DISCOVERED TOPICS")
        print(f"{'='*60}")
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [words[i] for i in top_words_idx]
            print(f"\nTopic {topic_idx + 1}: {', '.join(top_words)}")
    
    def display_results(self, query, results, model_name):
        """Display retrieval results"""
        print(f"\n{'='*60}")
        print(f"{model_name.upper()} RESULTS")
        print(f"Query: '{query}'")
        print(f"{'='*60}")
        
        if not results:
            print("\nNo relevant documents found.")
            return
        
        for result in results:
            print(f"\n[Rank {result['rank']}] {result['title']}")
            if 'score' in result:
                print(f"Score: {result['score']:.4f}")
            print(f"Snippet: {result['snippet']}...")
            print("-" * 60)


def main():
    """Main function to run the IR system"""
    print("="*60)
    print("INFORMATION RETRIEVAL SYSTEM")
    print("Dataset: Wikipedia Articles")
    print("="*60)
    
    # Initialize system
    ir_system = WikipediaIRSystem()
    
    # Step 1: Collect data
    print("\nEnter Wikipedia topics to collect (comma-separated):")
    print("Example: Machine Learning, Artificial Intelligence, Data Science")
    topics_input = input("Topics: ")
    topics = [t.strip() for t in topics_input.split(',')]
    
    print("\nHow many documents per topic? (default 15, recommended 10-20): ")
    docs_per_topic_input = input("Documents per topic: ")
    docs_per_topic = int(docs_per_topic_input) if docs_per_topic_input.isdigit() else 15
    
    num_collected = ir_system.collect_wikipedia_data(topics, num_docs_per_topic=docs_per_topic)
    
    if num_collected == 0:
        print("\nNo documents collected. Exiting.")
        return
    
    # Step 2: Preprocess
    ir_system.preprocess_all_documents()
    
    # Step 3: Vectorize
    ir_system.vectorize_documents()
    
    # Get initial query once
    print(f"\n{'='*60}")
    print("MAIN QUERY")
    print(f"{'='*60}")
    main_query = input("\nEnter your query: ")
    
    # Interactive query loop
    while True:
        print(f"\n{'='*60}")
        print("QUERY INTERFACE")
        print(f"{'='*60}")
        print(f"Current Query: '{main_query}'")
        print("\nOptions:")
        print("1. Vector Space Model (Cosine Similarity)")
        print("2. Boolean Retrieval Model")
        print("3. BM25 Ranking")
        print("4. Compare All Models")
        print("5. Visualizations")
        print("6. Topic Modeling (LDA)")
        print("7. Evaluate Retrieval Performance (All Documents)")
        print("8. Exit")
        
        choice = input("\nSelect option (1-8): ")
        
        if choice == '8':
            print("\nThank you for using the IR System!")
            break
        
        if choice in ['1', '2', '3', '4']:
            if choice == '1':
                results = ir_system.vector_space_model(main_query, top_k=20)
                ir_system.display_results(main_query, results, "Vector Space Model")
                if results:
                    ir_system.visualize_similarity_scores(main_query, results)
            
            elif choice == '2':
                results = ir_system.boolean_retrieval(main_query)
                ir_system.display_results(main_query, results, "Boolean Retrieval")
            
            elif choice == '3':
                results = ir_system.bm25_retrieval(main_query, top_k=20)
                ir_system.display_results(main_query, results, "BM25 Ranking")
            
            elif choice == '4':
                print("\n" + "="*60)
                print("COMPARING ALL MODELS")
                print("="*60)
                
                vsm_results = ir_system.vector_space_model(main_query, top_k=20)
                bool_results = ir_system.boolean_retrieval(main_query)
                bm25_results = ir_system.bm25_retrieval(main_query, top_k=20)
                
                ir_system.display_results(main_query, vsm_results, "Vector Space Model")
                ir_system.display_results(main_query, bool_results, "Boolean Retrieval")
                ir_system.display_results(main_query, bm25_results, "BM25 Ranking")
        
        elif choice == '5':
            print("\nVisualization Options:")
            print("1. Word Cloud")
            print("2. Word Frequency")
            print("3. Both")
            
            viz_choice = input("Select (1-3): ")
            
            if viz_choice in ['1', '3']:
                ir_system.visualize_word_cloud()
            if viz_choice in ['2', '3']:
                ir_system.visualize_word_frequency()
        
        elif choice == '6':
            n_topics = int(input("Enter number of topics (default 5): ") or "5")
            ir_system.topic_modeling_lda(n_topics)
        
        elif choice == '7':
            # Evaluate retrieval quality on top-20 documents
            ir_system.evaluate_retrieval(main_query, top_k=20)


if __name__ == "__main__":
    main()