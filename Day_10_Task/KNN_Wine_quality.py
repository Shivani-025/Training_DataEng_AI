'''
Question 2
Implement the k-Nearest Neighbors (k-NN) algorithm to classify the Wine Quality dataset into good and bad quality wines using different values of k. Evaluate the model's performance using accuracy.

Steps:

Load the Wine Quality dataset from the UCI Machine Learning Repository.
Preprocess the data: normalize the features and binarize the target variable into good (quality >= 7) and bad (quality < 7).
Split the data into training and testing sets.
Implement the k-NN algorithm
Experiment with different values of k (e.g., k=3, 5, 7) and evaluate the model's performance.
Report the accuracy for each value of k

'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import torch

# Load the Wine Quality dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine = pd.read_csv(url, sep=';')

# Preprocess the data: normalize the features and binarize the target variable
X = wine.drop('quality', axis=1)
y = np.where(wine['quality'] >= 7, 1, 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

# Implementing the k-NN Algorithm
def euclidean_distance(a, b):
    return torch.sqrt(torch.sum((a - b) ** 2, dim=1))


def knn_predict(X_train, y_train, X_test, k):
    y_pred = []
    for test_point in X_test:
        distances = euclidean_distance(X_train, test_point)
        _, indices = torch.topk(distances, k, largest=False)
        nearest_labels = y_train[indices]
        majority_label = torch.mode(nearest_labels).values.item()
        y_pred.append(majority_label)
    return torch.tensor(y_pred)


# Experiment with different values of k
k_values = [3, 5, 7]
for k in k_values:
    y_pred = knn_predict(X_train_tensor, y_train_tensor, X_test_tensor, k)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'k = {k}, Accuracy: {accuracy * 100:.2f}%')

    # Generate classification report
    report = classification_report(y_test, y_pred)
    print('10.Classification Report:')
    print(report)
