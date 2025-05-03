'''
Exercise 1: Implement PCA on a Dataset using PyTorch and Visualize the Results
Dataset: Wine Quality Dataset

Objective: Implement PCA on the Wine Quality dataset to reduce its dimensionality and visualize the results in 2D.

Steps:
Load the Wine Quality dataset:
Use pandas to load the Wine Quality dataset from the UCI Machine Learning Repository.

Preprocess the data:
Normalize the features using StandardScaler.

Implement PCA using PyTorch:
Compute the covariance matrix.
Compute eigenvalues and eigenvectors of the covariance matrix.
Project the data onto the first two principal components.

Visualize the Results:
Use matplotlib to plot the 2D projection of the data.


'''

import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib.pyplot as plt

# Step 1: Load the Wine Quality Dataset
# Load the dataset from UCI ML Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine_data = pd.read_csv(url, sep=';')

# Display the first few rows of the dataset
print(wine_data.head())

# Step 2: Preprocess the Data
# Separate features from the target variable
X = wine_data.drop('quality', axis=1)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Implement PCA using PyTorch
# Convert numpy array to PyTorch tensor
X_tensor = torch.tensor(X_scaled, dtype=torch.float)

# Compute the covariance matrix manually
mean = torch.mean(X_tensor, dim=0)
X_centered = X_tensor - mean
cov_matrix = torch.matmul(X_centered.t(), X_centered) / (X_centered.size(0) - 1)

# Compute eigenvalues and eigenvectors using torch.linalg.eigh
eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

# Sort eigenvalues and corresponding eigenvectors in descending order
eigenvalues = eigenvalues.flip(0)
eigenvectors = eigenvectors.flip(1)

# Project the data onto the first two principal components
principal_components = torch.matmul(X_centered, eigenvectors[:, :2])

# Step 4: Visualize the Results
# Convert principal components back to numpy array for plotting
principal_components_np = principal_components.detach().numpy()

# Plotting the 2D projection
plt.figure(figsize=(8, 6))
plt.scatter(principal_components_np[:, 0], principal_components_np[:, 1], c=wine_data['quality'], cmap='viridis', alpha=0.5)
plt.title('PCA on Wine Quality Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Quality')
plt.grid(True)
plt.show()