'''
Exercise 1: Implement k-Means 11.Clustering on a Dataset using PyTorch
Dataset: Iris Dataset

Objective: Implement k-Means clustering on the Iris dataset to group the data into clusters. Visualize the results and evaluate the clustering performance using a suitable metric such as silhouette score.

Steps:
Load the Iris dataset:

Use pandas to load the Iris dataset from the UCI Machine Learning Repository.

Preprocess the data:
Normalize the features using StandardScaler.

Implement k-Means 11.Clustering using PyTorch:
Initialize cluster centroids randomly.
Assign each data point to the nearest centroid.
Update centroids by computing the mean of the assigned points.
Repeat the process until convergence.

Visualize the Clusters:
Use matplotlib to plot the clusters.
If the data has more than 2 features, use PCA to reduce dimensionality before plotting.

Evaluate the 11.Clustering Performance:
Use silhouette score from sklearn.metrics.
'''

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import torch
import numpy as np
import matplotlib.pyplot as plt

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_df = pd.read_csv(url, header=None, names=column_names)

# Extract features and normalize them
features = iris_df.drop('class', axis=1)
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Implement k-Means 11.Clustering using PyTorch
def kmeans(X, k, num_iters=100):
    X = torch.tensor(X, dtype=torch.float)
    np.random.seed(42)
    centroids = X[torch.randint(0, X.shape[0], (k,))]

    for _ in range(num_iters):
        distances = torch.cdist(X, centroids)
        cluster_assignments = torch.argmin(distances, dim=1)
        new_centroids = torch.vstack([X[cluster_assignments == i].mean(dim=0) for i in range(k)])
        if torch.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, cluster_assignments

# Run k-Means clustering
k = 3
centroids, cluster_assignments = kmeans(normalized_features, k)

# Reduce dimensionality to 2D for visualization
pca = PCA(n_components=2)
pca_features = pca.fit_transform(normalized_features)

# Plot the clusters
plt.figure(figsize=(10, 6))
for i in range(k):
    cluster_points = pca_features[cluster_assignments == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')
plt.scatter(pca.transform(scaler.inverse_transform(centroids))[:, 0],
            pca.transform(scaler.inverse_transform(centroids))[:, 1],
            color='black', marker='x', s=100, label='Centroids')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('k-Means 11.Clustering of Iris Dataset')
plt.legend()
plt.show()

# Calculate silhouette score
score = silhouette_score(normalized_features, cluster_assignments)
print(f'Silhouette Score: {score}')