'''
Exercise 5: Optimize Performance Using Vectorization and Numba

a. Vectorization:
Create a function to compute the element-wise square of an array using a for loop.
Create another function to perform the same computation using NumPy vectorization.
Compare the performance of the two functions using a large array of size 1,000,000.

b. Numba:
Use the @numba.jit decorator to optimize the function from step 1 that uses a for loop.
Compare the performance of the Numba-optimized function with the vectorized NumPy function.
'''


import numpy as np
import time
import numba

'''
a. Vectorization:
Create a function to compute the element-wise square of an array using a for loop.
Create another function to perform the same computation using NumPy vectorization.
Compare the performance of the two functions using a large array of size 1,000,000.
'''
def square_for_loop(arr):
    result = np.zeros_like(arr)
    for i in range(len(arr)):
        result[i] = arr[i] ** 2
    return result

def square_vectorized(arr):
    return np.square(arr)


large_array = np.random.rand(1000000)

start_time_for_loop = time.time()
result_for_loop = square_for_loop(large_array)
end_time_for_loop = time.time()

start_time_vectorized = time.time()
result_vectorized = square_vectorized(large_array)
end_time_vectorized = time.time()

print("Vectorization:")
print("Time taken for for loop method:", end_time_for_loop - start_time_for_loop, "seconds")
print("Time taken for vectorized method:", end_time_vectorized - start_time_vectorized, "seconds")
print()

'''
b. Numba:
Use the @numba.jit decorator to optimize the function from step 1 that uses a for loop.
Compare the performance of the Numba-optimized function with the vectorized NumPy function.
'''
@numba.jit
def square_numba(arr):
    result = np.zeros_like(arr)
    for i in range(len(arr)):
        result[i] = arr[i] ** 2
    return result

if __name__ == '__main__':

    start_time_numba = time.time()
    result_numba = square_numba(large_array)
    end_time_numba = time.time()

    print("Numba Optimization:")
    print("Time taken for Numba-optimized function:", end_time_numba - start_time_numba, "seconds")
