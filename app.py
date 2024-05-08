import streamlit as st
import joblib
import os
from scipy.sparse import lil_matrix, csr_matrix
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
import numpy as np

# loading trained model and other files
knn = joblib.load("joblib_files/knn_model.joblib")
tf_idf_training_scores = joblib.load("joblib_files/tf_idf_training_scores.joblib")
idf_values = joblib.load("joblib_files/idf_values.joblib")
num_tokens = joblib.load("joblib_files/num_tokens.joblib")
token_to_index = joblib.load("joblib_files/token_to_index.joblib")
y_train = joblib.load("joblib_files/y_train.joblib")
processed_docs = joblib.load("joblib_files/processed_docs.joblib")


# Define or load the 'labels' variable
labels = {
    1: "Explainable Artificial Intelligence",
    2: "Explainable Artificial Intelligence",
    3: "Explainable Artificial Intelligence",
    7: "Explainable Artificial Intelligence",
    8: "Heart Failure",
    9: "Heart Failure",
    11: "Heart Failure",
    12: "Time Series Forecasting",
    13: "Time Series Forecasting",
    14: "Time Series Forecasting",
    15: "Time Series Forecasting",
    16: "Time Series Forecasting",
    17: "Transformer Model",
    18: "Transformer Model",
    21: "Transformer Model",
    22: "Feature Selection",
    23: "Feature Selection",
    24: "Feature Selection",
    25: "Feature Selection",
    26: "Feature Selection"
}

documents = {}
for i in range(1, 27):
    file_path = f"extracted_papers/{i}.txt"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            documents[i] = file.read()

# convert tf-idf scores to sparse matrix
def convert_to_sparse_matrix(tf_idf_scores, num_tokens):
    num_docs = len(tf_idf_scores)
    matrix = lil_matrix((num_docs, num_tokens), dtype=np.float64)
    for i, tf_idf_vec in tf_idf_scores.items():
        if i >= num_docs:
            continue  # Skip if row index is out of bounds
        for token_index_str, score in tf_idf_vec.items():
            try:
                token_index = int(token_index_str) 
                if token_index < num_tokens:
                    matrix[i, token_index] = score
            except ValueError:
                pass
    return matrix.tocsr()

# preprocess + tf-idf for input text
def preprocess_and_calculate_tf_idf_input(input_text, idf_values):
    stemmer = PorterStemmer()
    tokens = word_tokenize(input_text)
    stop_words_list = ["a", "is", "the", "of", "all", "and", "to", "can", "be", "as", "once", "for", "at", "am", "are", "has", "have", "had", "up", "his", "her", "in", "on", "no", "we", "do"]
    filtered_tokens = [stemmer.stem(token.lower()) for token in tokens if token.isalnum() and token.lower() not in stop_words_list]
    tf_idf_vec = {}
    total_tokens = len(filtered_tokens)
    for token in set(filtered_tokens):
        tf = filtered_tokens.count(token) / total_tokens
        tf_idf = tf * idf_values.get(token, 0)
        token_index = token_to_index.get(token)
        if token_index is not None:
            tf_idf_vec[token_index] = tf_idf
    return tf_idf_vec

def predict_label(input_text):
    tf_idf_input = preprocess_and_calculate_tf_idf_input(input_text, idf_values)
    X_input = csr_matrix(([value for value in tf_idf_input.values()], ([0] * len(tf_idf_input), list(tf_idf_input.keys()))), shape=(1, num_tokens))
    return knn.predict(X_input)[0]

def cluster_documents(tf_idf_scores, k):
    X_sparse = convert_to_sparse_matrix(tf_idf_scores, num_tokens)
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_sparse)
    return clusters

def get_clustered_documents(clusters, tf_idf_scores):
    clustered_docs = {}
    for doc_id, cluster_id in enumerate(clusters):
        # Extracting the document name
        doc_name = list(tf_idf_scores.keys())[doc_id]  
        if cluster_id not in clustered_docs:
            clustered_docs[cluster_id] = []
        clustered_docs[cluster_id].append(doc_name)
    return clustered_docs

def calculate_purity(clustered_docs, true_labels):
    total_docs = sum(len(docs) for docs in clustered_docs.values())
    purity = 0
    for docs in clustered_docs.values():
        max_count = 0
        for label in set(true_labels):
            count = sum(1 for doc in docs if labels[doc] == label)
            if count > max_count:
                max_count = count
        purity += max_count
    return purity / total_docs

def calculate_random_index(clustered_docs, true_labels):
    true_labels_num = []
    clustered_labels_num = []
    for docs in clustered_docs.values():
        for doc in docs:
            clustered_labels_num.append(labels[doc])
    for label in true_labels:
        true_labels_num.append(label)
    true_labels_num = np.array(true_labels_num)
    clustered_labels_num = np.array(clustered_labels_num)
    return adjusted_rand_score(true_labels_num, clustered_labels_num)
X_train_sparse = convert_to_sparse_matrix(tf_idf_training_scores, num_tokens)


# Streamlit GUI
st.title("Text Classification and Clustering with k-NN")

# Classification Section
with st.expander("Text Classification"):
    # Calculate classification metrics
    predictions = [predict_label(documents[i]) for i in sorted(processed_docs.keys())]
    accuracy = accuracy_score(y_train, predictions)
    precision = precision_score(y_train, predictions, average='weighted', zero_division=1)
    recall = recall_score(y_train, predictions, average='weighted')
    f1 = f1_score(y_train, predictions, average='weighted')

    # Display classification metrics
    st.write("Classification Metrics:")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

    input_text = st.text_area("Input Text", "")

    if st.button("Classify"):
        if input_text.strip() != "":
            predicted_label = predict_label(input_text)
            st.write(f"Predicted Label: {predicted_label}")


with st.expander("Text Clustering"):
    k = st.slider("Number of Clusters (k)", min_value=2, max_value=10, value=5)
    if st.button("Cluster"):
        clusters = cluster_documents(tf_idf_training_scores, k)
        clustered_docs = get_clustered_documents(clusters, tf_idf_training_scores)
        
        cols = st.columns(len(clustered_docs))
        for i, (cluster_id, filenames_in_cluster) in enumerate(clustered_docs.items()):
            with cols[i]:
                st.write(f"Cluster {cluster_id}:")
                for filename in filenames_in_cluster:
                    st.write(f"- {filename}")
        
        # Calculate clustering metrics
        true_labels = [labels[i] for i in sorted(processed_docs.keys())]
        purity = calculate_purity(clustered_docs, true_labels)
        silhouette = silhouette_score(X_train_sparse, clusters)
        random_index = calculate_random_index(clustered_docs, true_labels)

        # Display clustering metrics
        st.write("Clustering Metrics:")
        st.write(f"Purity: {purity:.2f}")
        st.write(f"Silhouette Score: {silhouette:.2f}")
        st.write(f"Random Index: {random_index:.2f}")
