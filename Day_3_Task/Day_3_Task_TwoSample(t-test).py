'''
    Question 2: Two-Sample t-Test
Perform a two-sample t-test to compare the means of two independent samples.

Generate two sample datasets each with 25 random values from normal distributions with means of 55 and 60, and a standard deviation of 8.
Perform an independent two-sample t-test to check if the means of the two samples are significantly different.
'''


import numpy as np
from scipy import stats


np.random.seed(42)


mean1, mean2 = 55, 60
std_dev = 8
n_samples = 25

sample1 = np.random.normal(mean1, std_dev, n_samples)
sample2 = np.random.normal(mean2, std_dev, n_samples)


t_stat, p_value = stats.ttest_ind(sample1, sample2)

print("Sample 1:", sample1)
print("Sample 2:", sample2)
print("t-statistic:", t_stat)
print("p-value:", p_value)