import streamlit as st
from collections import defaultdict
import math
import os
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix, adjusted_rand_score, silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
from joblib import Parallel, delayed
import numpy as np
import chardet

st.title("Text Classification and Clustering")

# Load stopwords
stop_words_list = ['a', 'is', 'the', 'of', 'all', 'and', 'to', 'can', 'be', 'as', 'once', 'for', 'at', 'am', 'are', 'has', 'have', 'had', 'up', 'his', 'her', 'in', 'on', 'no', 'we', 'do']

# Load documents and labels
documents = {}
labels = {}
for i in range(1, 27):
    file_path = f"extracted_papers/{i}.txt"
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            documents[i] = file.read()
            if i in [1, 2, 3, 7]:
                labels[i] = "Explainable Artificial Intelligence"
            elif i in [8, 9, 11]:
                labels[i] = "Heart Failure"
            elif i in [12, 13, 14, 15, 16]:
                labels[i] = "Time Series Forecasting"
            elif i in [17, 18, 21]:
                labels[i] = "Transformer Model"
            elif i in [22, 23, 24, 25, 26]:
                labels[i] = "Feature Selection"

# Preprocess documents
def preprocess(doc_id, text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    filtered_tokens = [stemmer.stem(token.lower()) for token in tokens if token.isalnum() and token.lower() not in stop_words_list]
    return doc_id, filtered_tokens

processed_docs = dict(Parallel(n_jobs=-1)(delayed(preprocess)(doc_id, text) for doc_id, text in documents.items()))

# Build TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform([" ".join(tokens) for tokens in processed_docs.values()])

# Apply Truncated SVD for dimensionality reduction
svd = TruncatedSVD(n_components=300)
tfidf_matrix_reduced = svd.fit_transform(tfidf_matrix)

# Prepare training and testing data
y = [labels[i] for i in processed_docs.keys()]
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix_reduced, y, test_size=0.2, random_state=42)

# Build k-NN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Train k-Means clustering
kmeans = KMeans(n_clusters=5, random_state=0)
clusters = kmeans.fit_predict(tfidf_matrix_reduced)

# Define functions for evaluation metrics
def evaluate_classification(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1

def evaluate_clustering(y_true, clusters):
    cont_matrix = contingency_matrix(y_true, clusters)
    purity = np.sum(np.amax(cont_matrix, axis=0)) / np.sum(cont_matrix)
    silhouette = silhouette_score(tfidf_matrix_reduced, clusters)
    rand_index = adjusted_rand_score(y_true, clusters)
    return purity, silhouette, rand_index

# Streamlit GUI

st.write("Evaluation Metrics:")
accuracy, precision, recall, f1 = evaluate_classification(y_test, knn.predict(X_test))
purity, silhouette, rand_index = evaluate_clustering(y, clusters)
st.write(f"Classification - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
st.write(f"Clustering - Purity: {purity}, Silhouette Score: {silhouette}, Rand Index: {rand_index}")

st.subheader("Test Document")
st.write("Enter a new document to classify or upload a txt file:")

# Allow user to input text or upload file
text_input = st.text_area("Text Input")
file_upload = st.file_uploader("Upload a txt file", type="txt")

if text_input:
    new_doc = text_input
elif file_upload is not None:
    raw_data = file_upload.getvalue()
    encoding_result = chardet.detect(raw_data)
    file_encoding = encoding_result['encoding']
    new_doc = raw_data.decode(file_encoding)
else:
    new_doc = None

if new_doc is not None:
    st.write("Predicted Class:")
    new_processed_doc = preprocess(27, new_doc)
    new_tfidf_matrix = tfidf_vectorizer.transform([" ".join(new_processed_doc[1])])
    new_tfidf_matrix_reduced = svd.transform(new_tfidf_matrix)
    new_y_pred = knn.predict(new_tfidf_matrix_reduced)
    st.write(f"{new_y_pred[0]}")

