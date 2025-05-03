'''
    Question 4: Descriptive Statistics with NumPy and SciPy

Create a dataset with 20 random values between 1 and 100.
Compute the following statistics for the dataset:
Mean
Median
Standard deviation
Variance
Skewness
Kurtosis
'''

import numpy as np
import scipy as sp


# Generate a dataset with 20 random values between 1 and 100
dataset = np.random.randint(1, 101, size=20)
print("Dataset:",dataset)

# Compute the mean of the dataset
mean = np.mean(dataset)
print("Mean:", mean)

# Compute the median of the dataset
median = np.median(dataset)
print("Median:", median)

# Compute the standard deviation of the dataset
std_dev = np.std(dataset)
print("Standard Deviation:", std_dev)

# Compute the variance of the dataset
variance = np.var(dataset)
print("Variance:", variance)

# Compute the skewness of the dataset
skewness = sp.stats.skew(dataset)
print("Skewness:", skewness)

# Compute the kurtosis of the dataset
kurtosis = sp.stats.kurtosis(dataset)
print("Kurtosis:", kurtosis)
