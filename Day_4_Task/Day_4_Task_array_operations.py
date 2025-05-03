'''Exercise 2: Perform Basic and Advanced Array Operations

a. Array arithmetic:
Create two 1-dimensional arrays of integers from 1 to 5 and 6 to 10.
Perform element-wise addition, subtraction, multiplication, and division and Print the results.

b. Indexing and slicing:
Create a 5x5 array with values from 1 to 25.
Extract the subarray consisting of the first two rows and columns.
Print the extracted subarray.

c. Boolean indexing:
Create a 1-dimensional array of integers from 10 to 19.
Extract elements greater than 15.
Print the resulting array.
'''

import numpy as np

'''
a. Array arithmetic:
Create two 1-dimensional arrays of integers from 1 to 5 and 6 to 10.
Perform element-wise addition, subtraction, multiplication, and division and Print the results.

'''
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([6, 7, 8, 9, 10])
# Element-wise addition
print("Element-wise addition:", array1 + array2)
# Element-wise subtraction
print("Element-wise subtraction:", array1 - array2)
# Element-wise multiplication
print("Element-wise multiplication:", array1 * array2)
# Element-wise division
print("Element-wise division:", array1 / array2)


'''
b. Indexing and slicing:
Create a 5x5 array with values from 1 to 25.
Extract the subarray consisting of the first two rows and columns.
Print the extracted subarray.
'''
arr = np.arange(1,26,)
array = arr.reshape(5,5)
print("\nA 5x5 array with values from 1 to 25:\n",array)
s_arr = array[0:2,0:2]
print("The extracted subarray :\n",s_arr)


'''
c. Boolean indexing:
Create a 1-dimensional array of integers from 10 to 19.
Extract elements greater than 15.
Print the resulting array.
'''

arr3 = np.arange(10, 20)
extracted = arr3[arr3 > 15]
print("Elements greater than 15:", extracted)