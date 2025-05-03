'''
Question 5: Post-hoc Test using Tukey's HSD
Objective: Perform a post-hoc test using Tukey's HSD to identify which groups are significantly different.

Use the same datasets generated in the one-way ANOVA exercise.
Perform Tukey's HSD test to find out which pairs of group means are significantly different.
'''

import numpy as np

from statsmodels.stats.multicomp import pairwise_tukeyhsd
np.random.seed(2)
group1 = np.random.normal(50, 10, 20)
group2 = np.random.normal(55, 10, 20)
group3 = np.random.normal(60, 10, 20)

data = np.concatenate([group1, group2, group3])
labels = ['Group1'] * 20 + ['Group2'] * 20 + ['Group3'] * 20


tukey_result = pairwise_tukeyhsd(endog=data, groups=labels, alpha=0.05)
print("Tukey's HSD Test:\n", tukey_result)