'''
Question 2: Solving Linear Equations with SciPy

Given the system of equations:
2x + 3y = 8
3x + 4y = 11

Represent the system of equations in matrix form AX = B.
Use scipy.linalg.solve to find the values of x and y.
'''

import numpy as np
from scipy.linalg import solve

# Represent the system of equations in matrix form AX = B
A = np.array([[2, 3], [3, 4]])
B = np.array([8, 11])

# Use scipy.linalg.solve to find the values of x and y
solution = solve(A, B)

print("Solution:")
print("x =", solution[0])
print("y =", solution[1])
