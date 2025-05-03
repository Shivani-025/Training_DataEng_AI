'''
Exercise 2: Apply Hierarchical 11.Clustering and Create a Dendrogram

Dataset: Wine Quality Dataset

Objective: Apply hierarchical clustering on the Wine Quality dataset to group the data into clusters. Create a dendrogram to visualize the hierarchical relationships.

Steps:

Load the Wine Quality dataset:
Use pandas to load the Wine Quality dataset from the UCI Machine Learning Repository.

Preprocess the data:
Normalize the features using StandardScaler.

Apply Hierarchical 11.Clustering:
Use scipy.cluster.hierarchy to perform hierarchical clustering.
Use linkage method to compute the hierarchical clustering.

Create and Visualize the Dendrogram:
Use dendrogram function to create the dendrogram.

Evaluate the 11.Clustering Performance:
Use silhouette score from sklearn.metrics.
'''
# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Step 1: Load the Wine Quality dataset
wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine_df = pd.read_csv(wine_url, sep=';')
print("First five rows of the Wine Quality dataset:")
print(wine_df.head())

# Step 2: Preprocess the data
# Normalize the features excluding 'quality'
scaler = StandardScaler()
X_wine = scaler.fit_transform(wine_df.drop('quality', axis=1))

# Convert to DataFrame for compatibility with clustering functions
X_wine_df = pd.DataFrame(X_wine, columns=wine_df.columns[:-1])

# Step 3: Apply Hierarchical 11.Clustering
# Perform hierarchical clustering using the 'ward' linkage method
Z = linkage( X_wine_df, method='ward')

# Step 4: Create and Visualize the Dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Hierarchical 11.Clustering Dendrogram (Wine Quality Dataset)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Step 5: Evaluate the 11.Clustering Performance
# Apply Agglomerative 11.Clustering to assign clusters for silhouette score calculation
agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
cluster_assignments_wine = agg_clustering.fit_predict(X_wine_df)

# Compute silhouette score
silhouette_avg_wine = silhouette_score(X_wine_df, cluster_assignments_wine)
print(f'Silhouette Score for Hierarchical 11.Clustering on Wine Quality Dataset: {silhouette_avg_wine}')