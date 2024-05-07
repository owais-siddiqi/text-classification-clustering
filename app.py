import streamlit as st
from collections import defaultdict
import math
import os
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import joblib

stop_words_list = ["a", "is", "the", "of", "all", "and", "to", "can", "be", "as", "once", "for", "at", "am", "are", "has", "have", "had", "up", "his", "her", "in", "on", "no", "we", "do"]

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
def preprocess(documents):
    stemmer = PorterStemmer()
    processed_docs = {}
    for doc_id, text in documents.items():
        tokens = word_tokenize(text)
        filtered_tokens = [stemmer.stem(token.lower()) for token in tokens if token.isalnum() and token.lower() not in stop_words_list]
        processed_docs[doc_id] = filtered_tokens, tokens  # return stemmed and unstemmed tokens
    return processed_docs

processed_docs = preprocess(documents)

# Build inverted index
def build_inverted_index(processed_docs):
    inverted_index = defaultdict(set)
    for doc_id, (filtered_tokens, _) in processed_docs.items():
        for token in filtered_tokens:
            inverted_index[token].add(doc_id)
    return inverted_index

inverted_index = build_inverted_index(processed_docs)

# tf-idf scores for training documents
def calculate_tf_idf_training(processed_docs, inverted_index):
    tf_idf_scores = {}
    N = len(processed_docs)
    for doc_id, (filtered_tokens, _) in processed_docs.items():
        tf_idf_vec = {}
        for token in set(filtered_tokens):
            tf = filtered_tokens.count(token) / len(filtered_tokens)  # Term Frequency (TF)
            idf = math.log(N / (len(inverted_index[token]) + 1))  # Inverse Document Frequency (IDF)
            tf_idf = tf * idf
            tf_idf_vec[token] = tf_idf

        norm = math.sqrt(sum(tf_idf_vec[token]**2 for token in tf_idf_vec))
        tf_idf_scores[doc_id] = {token: tf_idf / norm for token, tf_idf in tf_idf_vec.items()}

    return tf_idf_scores

tf_idf_training_scores = calculate_tf_idf_training(processed_docs, inverted_index)

# store tf-idf scores and labels for training set
X_train = []
y_train = []
all_tokens = set(token for scores in tf_idf_training_scores.values() for token in scores.keys())  # Get all unique tokens
for doc_id, scores in tf_idf_training_scores.items():
    row = []
    for token in all_tokens:
        row.append(scores.get(token, 0))  # 0 if token not present in document
    X_train.append(row)
    y_train.append(labels[doc_id])

X_train = np.array(X_train)
y_train = np.array(y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
joblib.dump(knn, "knn_model.joblib")


# can be later seperated so retraining can be avoided
knn = joblib.load("knn_model.joblib")

# predict label for input text
def predict_label(input_text):
    processed_input = preprocess({0: input_text})
    _, input_tokens = processed_input[0]  # Unpack the processed input
    tf_idf_input = calculate_tf_idf_input(input_tokens, inverted_index)
    X_input = []
    for scores in [tf_idf_input]:
        row = []
        for token in all_tokens:
            row.append(scores.get(token, 0))  # use 0 if token not present in document
        X_input.append(row)

    X_input = np.array(X_input)
    return knn.predict(X_input)[0]

# tf-idf scores for input text
def calculate_tf_idf_input(input_tokens, inverted_index):
    tf_idf_vec = {}
    N = len(processed_docs)
    for token in set(input_tokens):
        tf = input_tokens.count(token) / len(input_tokens)
        idf = math.log(N / (len(inverted_index[token]) + 1))
        tf_idf = tf * idf
        tf_idf_vec[token] = tf_idf

    norm = math.sqrt(sum(tf_idf_vec[token]**2 for token in tf_idf_vec))
    return {token: tf_idf / norm for token, tf_idf in tf_idf_vec.items()}

# Streamlit GUI
st.title("Text Classification with k-NN")
st.write("Enter text below to classify:")
input_text = st.text_area("Input Text", "")

if st.button("Classify"):
    if input_text.strip() != "":
        predicted_label = predict_label(input_text)
        st.write(f"Predicted Label: {predicted_label}")
