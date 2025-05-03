'''
Exercise 3: Use NumPy for Mathematical and Statistical Calculations

a. Mathematical functions:
Create an array of 10 evenly spaced values between 0 and 2π.
Compute the sine, cosine, and tangent of each value.
Print the results.

b. Statistical functions:
Create a 3x3 array with random integers between 1 and 100.
Compute the mean, median, standard deviation, and variance.
Print the results.
'''

import numpy as np


'''
a. Mathematical functions:
Create an array of 10 evenly spaced values between 0 and 2π.
Compute the sine, cosine, and tangent of each value.
Print the results.
'''
angles = np.array([0, 2 * np.pi , 10])

print("Mathematical Functions:")
print("Sine of angles:", np.sin(angles))
print("Cosine of angles:", np.cos(angles))
print("Tangent of angles:", np.tan(angles))

'''
b. Statistical functions:
Create a 3x3 array with random integers between 1 and 100.
Compute the mean, median, standard deviation, and variance.
Print the results.
'''

data = np.random.randint(1, 101, size=(3, 3))

print("\n\nStatistical Functions:")
print("Mean of data:", np.mean(data))
print("Median of data:", np.median(data))
print("Variance of data:", np.var(data))
print("Standard deviation of data:", np.std(data))