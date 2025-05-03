'''
    Question 1: One-Sample t-Test

Perform a one-sample t-test to determine if the sample mean is significantly different from a known population mean.

Generate a sample dataset of 30 random values from a normal distribution with a mean of 60 and a standard deviation of 10.
Perform a one-sample t-test to check if the sample mean is significantly different from 50.

'''




import numpy as np
from scipy import stats


np.random.seed(42)


mean = 60
std_dev = 10
n_samples = 30

sample = np.random.normal(mean, std_dev, n_samples)


population_mean = 50


t_stat, p_value = stats.ttest_1samp(sample, population_mean)

print("Sample:", sample)
print("Sample Mean:", np.mean(sample))
print("Population Mean:", population_mean)
print("t-statistic:", t_stat)
print("p-value:", p_value)