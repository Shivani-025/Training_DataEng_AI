'''
Exercise 4: Group and Aggregate Data for Analysis

# Create the DataFrame
data_group = pd.DataFrame({
    'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'C', 'A'],
    'value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
})


Load a dataset with columns category and value.
Group the data by category and compute the sum of value for each category.
Display the aggregated DataFrame.

Load a dataset with columns category and value.
Group the data by category and compute the mean and standard deviation of value for each category.
Display the aggregated DataFrame.


# Create the DataFrame
data_pivot = pd.DataFrame({
    'category': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'subcategory': ['X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z'],
    'value': [1, 2, 3, 4, 5, 6, 7, 8, 9]
})

# Save DataFrame to CSV file
data_pivot.to_csv('data_pivot.csv', index=False)


Load a dataset with columns category, subcategory, and value.
Create a pivot table that shows the sum of value for each combination of category and subcategory.
Display the pivot table.
'''


import pandas as pd

# Load dataset
data = pd.read_csv('data_group.csv')

# Group by category and compute the sum of value
grouped_sum_df = data.groupby('category')['value'].sum().reset_index()

print("Grouped and summed DataFrame:")
print(grouped_sum_df)

# Group by category and compute the mean and standard deviation of value
grouped_stats_df = data.groupby('category')['value'].agg(['mean', 'std']).reset_index()

print("Grouped DataFrame with mean and standard deviation:")
print(grouped_stats_df)


# Load dataset
data_pivot = pd.read_csv('data_pivot.csv')

# Create a pivot table
pivot_table = data_pivot.pivot_table(values='value', index='category', columns='subcategory', aggfunc='sum')

print("Pivot table:")
print(pivot_table)
