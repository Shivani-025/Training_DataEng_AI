'''Question 3: Chi-Squared Test
Objective: Perform a Chi-Squared test for independence.

Create a contingency table with observed frequencies for two categorical variables.

|-----------| Category A | Category B |
| Group 1 |     10   	    |     20   	 |
| Group 2 |     15    	    |     25     	 |

Perform a Chi-Squared test to determine if there is a significant association between the two categorical variables.

'''

import numpy as np
import scipy.stats as stats

observed = np.array([[10, 20], [15, 25]])


chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)
print(f"Chi-Squared Test:\nChi2 Statistic: {chi2_stat:.4f}, P-value: {p_value:.4f}, Degrees of Freedom: {dof}\nExpected Frequencies:\n{expected}\n")