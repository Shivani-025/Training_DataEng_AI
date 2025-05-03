import numpy as np

# linear transformation matrix
A = np.array([
    [2, 0],
    [0, 3]
])

# vector
x = np.array([1, 2])

y = np.dot(A, x)

print("Matrix A:\n", A)
print("Vector x:", x)
print("Transformed vector y:", y)


#========================================================================

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate data
np.random.seed(0)
data = np.random.rand(5, 2)
print('\n\nPCA Data\n', data)

# Standardize the data
data_mean = np.mean(data, axis=0)
data_centered = data - data_mean

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_centered)

print("Principal Components:\n", principal_components)

# Plot the data and the principal components
plt.scatter(principal_components[:, 0], principal_components[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of the Data')
plt.show()

#======================================================

import numpy as np

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Compute the SVD
U, S, Vt = np.linalg.svd(A)

print('\n\nSVD')
print("Matrix A:\n", A)
print("Matrix U:\n", U)
print("Singular Values:", S)
print("Matrix V^T:\n", Vt)

# Reconstruct the matrix using SVD components
Sigma = np.zeros((A.shape[0], A.shape[1]))
Sigma[:A.shape[1], :A.shape[1]] = np.diag(S)
A_reconstructed = np.dot(U, np.dot(Sigma, Vt))

print("Reconstructed Matrix A:\n", A_reconstructed)

#==========================================================================