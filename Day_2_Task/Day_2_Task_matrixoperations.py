'''
Question 1: Matrix Operations with NumPy

Create two 3x3 matrices A and B with random integer values between 1 and 10.
Compute the following:
The sum of A and B.
The difference between A and B.
The element-wise product of A and B.
The matrix product of A and B.
The transpose of matrix A.
The determinant of matrix A.

'''

import numpy as np

# Create two 3x3 matrices A and B with random integer values between 1 and 10
A = np.random.randint(1, 10,size=(3, 3))
B = np.random.randint(1, 10,size=(3,3))

print("Matrix A:\n", A)
print("Matrix B:\n ", B)
# Compute the sum of A and B
sum_AB = np.add(A, B)
# Compute the difference between A and B
diff_AB = np.subtract(A, B)

# Compute the element-wise product of A and B
product_AB = np.multiply(A, B)

# Compute the matrix product of A and B
matrix_product_AB = np.dot(A, B)

# Compute the transpose of matrix A
transpose_A = np.transpose(A)

# Compute the determinant of matrix A
determinant_A = np.linalg.det(A)

print("Sum of A and B:\n", sum_AB)
print("\nDifference between A and B:\n", diff_AB)
print("\nElement-wise product of A and B:\n", product_AB)
print("\nMatrix product of A and B:\n", matrix_product_AB)
print("\nTranspose of matrix A:\n", transpose_A)
print("\nDeterminant of matrix A:", determinant_A)



