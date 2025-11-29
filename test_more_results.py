"""
Quick test to show MORE documents with updated defaults
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import re

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class QuickIRTest:
    def __init__(self):
        # 30 documents - way more!
        self.documents = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn from data",
            "Deep learning uses neural networks with multiple layers to process data",
            "Natural language processing helps computers understand human language",
            "Computer vision enables machines to interpret visual information from images and videos",
            "Data science combines statistics, programming, and domain expertise to extract insights",
            "Artificial intelligence is transforming industries and creating new opportunities",
            "Neural networks are inspired by biological neurons in the brain",
            "Big data analytics helps organizations make data-driven decisions",
            "Supervised learning requires labeled training data to train models",
            "Unsupervised learning discovers patterns in unlabeled data",
            "Python is the most popular language for machine learning",
            "TensorFlow is a powerful framework for building deep learning models",
            "PyTorch provides dynamic computational graphs for neural networks",
            "Scikit-learn offers simple and efficient tools for data mining",
            "Pandas is essential for data manipulation and analysis",
            "NumPy provides numerical computing capabilities",
            "Clustering algorithms group similar data points together",
            "Classification models predict categories for new data",
            "Regression analysis finds relationships between variables",
            "Feature engineering improves model performance",
            "Cross-validation evaluates model generalization",
            "Hyperparameter tuning optimizes model performance",
            "Ensemble methods combine multiple models for better predictions",
            "Random forests use multiple decision trees",
            "Support Vector Machines find optimal decision boundaries",
            "Gradient boosting creates strong models iteratively",
            "Recurrent neural networks process sequential data",
            "Convolutional neural networks excel at image processing",
            "Transformers revolutionize natural language understanding",
            "Transfer learning reuses pre-trained models"
        ]
        
        self.doc_titles = [
            "Machine Learning Basics",
            "Deep Learning Guide",
            "NLP Introduction",
            "Computer Vision Fundamentals",
            "Data Science Overview",
            "AI Transformation",
            "Neural Networks",
            "Big Data Analytics",
            "Supervised Learning",
            "Unsupervised Learning",
            "Python for ML",
            "TensorFlow Tutorial",
            "PyTorch Guide",
            "Scikit-learn Basics",
            "Pandas Tutorial",
            "NumPy Fundamentals",
            "Clustering Methods",
            "Classification Models",
            "Regression Analysis",
            "Feature Engineering",
            "Cross-Validation",
            "Hyperparameter Tuning",
            "Ensemble Methods",
            "Random Forests",
            "SVM Classifier",
            "Gradient Boosting",
            "RNN Models",
            "CNN for Images",
            "Transformers Architecture",
            "Transfer Learning"
        ]
        
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=500)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
    
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        tokens = [self.stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)
    
    def retrieve_documents(self, query, top_k=20):
        processed_query = self.preprocess_text(query)
        query_vector = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        top_k = min(top_k, len(similarities))
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append({
                'rank': rank,
                'title': self.doc_titles[idx],
                'score': similarities[idx]
            })
        
        return results


# Test with MORE documents
print("="*70)
print("TESTING WITH 30 DOCUMENTS - NOW YOU GET WAAAAAY MORE RESULTS!")
print("="*70)

tester = QuickIRTest()

# Test 1: Get 20 results (new default)
print("\n" + "="*70)
print("QUERY: 'machine learning deep learning'")
print("="*70)
print("\nRetrieving TOP 20 RESULTS:")
print("-"*70)

results = tester.retrieve_documents("machine learning deep learning", top_k=20)

for r in results:
    print(f"[{r['rank']:2d}] {r['title']:<30} Score: {r['score']:.4f}")

# Test 2: Get even MORE - 30 results!
print("\n" + "="*70)
print("QUERY: 'neural networks data learning'")
print("="*70)
print("\nRetrieving ALL 30 DOCUMENTS:")
print("-"*70)

results_all = tester.retrieve_documents("neural networks data learning", top_k=30)

for r in results_all:
    print(f"[{r['rank']:2d}] {r['title']:<30} Score: {r['score']:.4f}")

print("\n" + "="*70)
print(f"TOTAL DOCUMENTS RETRIEVED: {len(results_all)}")
print("="*70)

# Statistics
print("\n📊 STATISTICS:")
print(f"  • Average Score: {np.mean([r['score'] for r in results_all]):.4f}")
print(f"  • Max Score: {max(r['score'] for r in results_all):.4f}")
print(f"  • Min Score: {min(r['score'] for r in results_all):.4f}")
print(f"  • Documents with Score > 0: {len([r for r in results_all if r['score'] > 0])}")
print("\n✨ No more limiting to just 5 documents!")
