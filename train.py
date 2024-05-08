from collections import defaultdict
import math
import os
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import joblib
from scipy.sparse import lil_matrix, csr_matrix


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

# preprocess documents and calculate idf values
def preprocess_and_calculate_idf(documents):
    stemmer = PorterStemmer()
    processed_docs = {}
    token_document_count = defaultdict(int)
    N = len(documents)
    for doc_id, text in documents.items():
        tokens = word_tokenize(text)
        filtered_tokens = [stemmer.stem(token.lower()) for token in tokens if token.isalnum() and token.lower() not in stop_words_list]
        processed_docs[doc_id] = filtered_tokens
        for token in set(filtered_tokens):
            token_document_count[token] += 1

    idf_values = {token: math.log(N / (count + 1)) for token, count in token_document_count.items()}
    return processed_docs, idf_values

processed_docs, idf_values = preprocess_and_calculate_idf(documents)

# tf-idf for training documents
def calculate_tf_idf_training(processed_docs, idf_values):
    tf_idf_scores = {}
    for doc_id, tokens in processed_docs.items():
        tf_idf_vec = {}
        total_tokens = len(tokens)
        for token in set(tokens):
            tf = tokens.count(token) / total_tokens  # Term Frequency (TF)
            tf_idf = tf * idf_values.get(token, 0)  # Use 0 if token not in idf_values
            tf_idf_vec[token] = tf_idf
        tf_idf_scores[doc_id] = tf_idf_vec
    return tf_idf_scores

tf_idf_training_scores = calculate_tf_idf_training(processed_docs, idf_values)

# tf-idf scores to sparse matrix
def convert_to_sparse_matrix(tf_idf_scores):
    num_docs = len(tf_idf_scores)
    num_tokens = len(set(token for scores in tf_idf_scores.values() for token in scores.keys()))
    matrix = lil_matrix((num_docs, num_tokens), dtype=np.float64)
    doc_ids = sorted(tf_idf_scores.keys())
    token_to_index = {token: i for i, token in enumerate(sorted(set(token for scores in tf_idf_scores.values() for token in scores.keys())))}
    for i, doc_id in enumerate(doc_ids):
        scores = tf_idf_scores[doc_id]
        for token, score in scores.items():
            matrix[i, token_to_index[token]] = score
    return matrix.tocsr()

X_train_sparse = convert_to_sparse_matrix(tf_idf_training_scores)
y_train = np.array([labels[i] for i in sorted(processed_docs.keys())])
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_sparse, y_train)

num_tokens = len(set(token for scores in tf_idf_training_scores.values() for token in scores.keys()))
token_to_index = {token: i for i, token in enumerate(sorted(set(token for scores in tf_idf_training_scores.values() for token in scores.keys())))}
# save trained model and other files
joblib.dump(knn, "joblib_files/knn_model.joblib")
joblib.dump(tf_idf_training_scores, "joblib_files/tf_idf_training_scores.joblib")
joblib.dump(idf_values, "joblib_files/idf_values.joblib")
joblib.dump(num_tokens, "joblib_files/num_tokens.joblib")
joblib.dump(token_to_index, "joblib_files/token_to_index.joblib")
joblib.dump(y_train, "joblib_files/y_train.joblib")
joblib.dump(processed_docs, "joblib_files/processed_docs.joblib")