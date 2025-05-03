'''Question 4: One-Way ANOVA
Objective: Perform a one-way ANOVA to compare means across multiple groups.

Generate three sample datasets each with 20 random values from normal distributions with means of 50, 55, and 60, and a standard deviation of 10.
Perform a one-way ANOVA to check if there are any significant differences in means across the three groups.
'''


import numpy as np
import scipy.stats as stats


np.random.seed(2)
group1 = np.random.normal(50, 10, 20)
group2 = np.random.normal(55, 10, 20)
group3 = np.random.normal(60, 10, 20)

# Perform a one-way ANOVA to check if there are any significant differences in means across the three groups.
f_stat, p_value = stats.f_oneway(group1, group2, group3)
print(f"One-Way ANOVA:\nF-statistic: {f_stat:.4f}, P-value: {p_value:.4f}\n")