'''

Question 5: Hypothesis Testing with SciPy
Generate a sample dataset of 30 random values from a normal distribution with mean 50 and standard deviation 5.
Perform a one-sample t-test to check if the sample mean is significantly different from 50.
'''


import numpy as np
from scipy import stats


np.random.seed(42)


mean = 50
std_dev = 5
n_samples = 30

sample = np.random.normal(mean, std_dev, n_samples)


population_mean = 50


t_stat, p_value = stats.ttest_1samp(sample, population_mean)

print("Sample:", sample)
print("Sample Mean:", np.mean(sample))
print("Population Mean:", population_mean)
print("t-statistic:", t_stat)
print("p-value:", p_value)