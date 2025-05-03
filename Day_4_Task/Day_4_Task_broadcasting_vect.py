'''
Exercise 4: Implement Broadcasting and Vectorized Operations

a. Broadcasting:
Create a 3x1 array with values from 1 to 3.
Create a 1x3 array with values from 4 to 6.
Add the two arrays using broadcasting.
Print the resulting array.

b. Vectorized operations:
Create two large arrays of size 1,000,000 with random values.
Compute the element-wise product of the two arrays.
Print the time taken for the computation using vectorized operations.
'''


import numpy as np

'''
a. Broadcasting:
Create a 3x1 array with values from 1 to 3.
Create a 1x3 array with values from 4 to 6.
Add the two arrays using broadcasting.
Print the resulting array.
'''

array1 = np.array([[1], [2], [3]])
array2 = np.array([[4, 5, 6]])
print("Array 1:\n")
print(array1)
print("Array 2:\n")
print(array2)
# Broadcasting array1 to match the shape of array2
print("Broadcasting array1 to match array2:\n", array1 + array2)



'''
b. Vectorized operations:
Create two large arrays of size 1,000,000 with random values.
Compute the element-wise product of the two arrays.
Print the time taken for the computation using vectorized operations.
'''
import time
large_array1 = np.random.rand(1000000)
large_array2 = np.random.rand(1000000)


start_time = time.time()
result_vectorized = large_array1 * large_array2
end_time = time.time()

print("\n\nVectorized Operations:")
print("Time taken for computation using vectorized operations:", end_time - start_time, "seconds")