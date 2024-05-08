import streamlit as st
import joblib
from scipy.sparse import lil_matrix, csr_matrix
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# loading trained model and other fiels
knn = joblib.load("joblib_files/knn_model.joblib")
tf_idf_training_scores = joblib.load("joblib_files/tf_idf_training_scores.joblib")
idf_values = joblib.load("joblib_files/idf_values.joblib")
num_tokens = joblib.load("joblib_files/num_tokens.joblib")
token_to_index = joblib.load("joblib_files/token_to_index.joblib")
y_train = joblib.load("joblib_files/y_train.joblib")  # Load true labels

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

X_train_sparse = convert_to_sparse_matrix(tf_idf_training_scores, num_tokens)

# Streamlit GUI
st.title("Text Classification with k-NN")
st.write("Enter text below to classify:")
input_text = st.text_area("Input Text", "")

if st.button("Classify"):
    if input_text.strip() != "":
        predicted_label = predict_label(input_text)
        st.write(f"Predicted Label: {predicted_label}")

        # # Calculate evaluation metrics
        # precision = precision_score([predicted_label], [y_train[0]], average='macro')
        # recall = recall_score([predicted_label], [y_train[0]], average='macro')
        # f1 = f1_score([predicted_label], [y_train[0]], average='macro')
        # accuracy = accuracy_score([predicted_label], [y_train[0]])

        # # Display evaluation metrics
        # st.write(f"Precision: {precision}")
        # st.write(f"Recall: {recall}")
        # st.write(f"F1 Score: {f1}")
        # st.write(f"Accuracy: {accuracy}")
