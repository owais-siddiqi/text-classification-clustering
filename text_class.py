import tkinter as tk
from tkinter import ttk
from collections import defaultdict
import math
import os
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

with open("stopwords.txt", "r") as file:
    stop_words_list = file.read().splitlines()
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

def preprocess(documents):
    stemmer = PorterStemmer()
    processed_docs = {}
    for doc_id, text in documents.items():
        tokens = word_tokenize(text)
        filtered_tokens = [stemmer.stem(token.lower()) for token in tokens if token.isalnum() and token.lower() not in stop_words_list]
        processed_docs[doc_id] = filtered_tokens
    return processed_docs

processed_docs = preprocess(documents)

def build_inverted_index(processed_docs):
    inverted_index = defaultdict(set)
    for doc_id, tokens in processed_docs.items():
        for token in tokens:
            inverted_index[token].add(doc_id)
    return inverted_index

inverted_index = build_inverted_index(processed_docs)

def calculate_tf_idf(processed_docs, inverted_index):
    tf_idf_scores = defaultdict(dict)
    N = len(processed_docs)
    for doc_id, tokens in processed_docs.items():
        tf_idf_vec = {}
        for token in set(tokens):
            tf = tokens.count(token) / len(tokens)  # Term Frequency (TF)
            idf = math.log(N / (len(inverted_index[token]) + 1))  # Inverse Document Frequency (IDF)
            tf_idf = tf * idf
            tf_idf_vec[token] = tf_idf

        norm = math.sqrt(sum(tf_idf_vec[token]**2 for token in tf_idf_vec))
        tf_idf_scores[doc_id] = {token: tf_idf / norm for token, tf_idf in tf_idf_vec.items()}

    return tf_idf_scores

tf_idf_scores = calculate_tf_idf(processed_docs, inverted_index)
X = []
all_tokens = set(token for scores in tf_idf_scores.values() for token in scores.keys())  # Get all unique tokens
for doc_id, scores in tf_idf_scores.items():
    row = []
    for token in all_tokens:
        row.append(scores.get(token, 0))  #0 if token not present in document
    X.append(row)
X = np.array(X)
# prep training and testing data
y = [labels[i] for i in tf_idf_scores.keys()]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build k-NN classifier with optimized k
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# model eval
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
