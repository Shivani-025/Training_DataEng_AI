'''
Exercise 1: Create different types of NumPy arrays and perform basic manipulations.

a. Create a 1-dimensional array:
Create a 1-dimensional array of integers from 0 to 9.
Print the array and its shape.

b. Create a 2-dimensional array:
Create a 2-dimensional array (3x3) with values from 1 to 9.
Print the array, its shape, and the sum of all elements.

c. Reshape the array:
Reshape the 1-dimensional array from step 1 into a 2x5 array.
Print the reshaped array and its shape.

'''

import numpy as np

'''
a. Create a 1-dimensional array:
Create a 1-dimensional array of integers from 0 to 9.
Print the array and its shape.
'''
l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
arr = np.array(l)
print("1-Dimensional Array: ", arr)
sh = np.shape(arr)
print("Shape of 1-Dimensional Array: ", sh)


'''
b. Create a 2-dimensional array:
Create a 2-dimensional array (3x3) with values from 1 to 9.
Print the array, its shape, and the sum of all elements.
'''
li = np.random.randint(0, 10, (3,3))
print("2-dimensional array (3x3) with values from 1 to 9:\n", li)
sh2 = np.shape(li)
print("Shape of 2-Dimensional array (3x3) : ", sh2)
s = np.sum(li)
print("The sum of all elements : ", s)



'''
c. Reshape the array:
Reshape the 1-dimensional array from step 1 into a 2x5 array.
Print the reshaped array and its shape.
'''

reshape_l = arr.reshape((2, 5))
print("Reshape the 1-dimensional array from step(a) into a 2x5 array :\n",reshape_l)