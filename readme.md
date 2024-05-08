# Text Classification with k-Nearest Neighbors

This project demonstrates text classification using the k-Nearest Neighbors (k-NN) algorithm and deploys it as a Streamlit web application. The classification model is trained on a set of labeled text documents, and users can input text via the web interface to classify it into predefined categories.

## Files

### train.py

`train.py` is responsible for training the text classification model and saving the necessary files for inference.

- **Data Loading and Preprocessing**: Loads text documents, preprocesses them by tokenizing, removing stopwords, and calculates IDF values.
- **TF-IDF Calculation**: Computes TF-IDF scores for each document based on preprocessed tokens.
- **Sparse Matrix Conversion**: Converts TF-IDF scores to a sparse matrix format.
- **Model Training and Saving**: Trains a k-NN classifier using TF-IDF scores and saves the trained model and relevant information.

### app.py

`app.py` contains the Streamlit web application for text classification.

- **Loading Pretrained Model**: Loads the trained k-NN model and relevant information for inference.
- **Text Preprocessing**: Preprocesses input text for classification.
- **Prediction**: Uses the trained model to predict the label for input text.
- **Streamlit Web Application**: Provides a web interface for users to input text and classify it.

## Instructions

1. **Training the Model**: Run `train.py` to train the text classification model and save the necessary files.

2. **Running the Application**: Run `app.py` to start the Streamlit web application.

3. **Accessing the Web Interface**: Open the provided URL in your web browser to access the application. Enter text in the input field and click "Classify" to see the predicted label.

## Requirements

- Python 3.x
- Required packages: nltk, scikit-learn, numpy, scipy, joblib, streamlit

Install the required packages using pip:

