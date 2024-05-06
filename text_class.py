import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# Load documents and assign labels
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

# Convert labels to numpy array
y = []
for i in range(1, 27):
    if i in labels:
        y.append(labels[i])
    else:
        print(f"Label not found for document {i}. Skipping...")
        continue

    if i not in documents:
        print(f"Document {i} not found. Skipping...")
        y.pop()  # Remove the label added for the missing document

y = np.array(y)

# Ensure the number of documents matches the number of labels
assert len(y) == len(documents), "Number of labels and documents do not match."

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(list(documents.values()), y, test_size=0.2, random_state=42)


# Create a pipeline with TF-IDF vectorizer and k-NN classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('knn', KNeighborsClassifier())
])

# Define hyperparameters for grid search
param_grid = {
    'knn__n_neighbors': [1, 3, 5, 7, 9]
}

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Use the best model for prediction
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred, average='weighted')

print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
