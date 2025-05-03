'''
Exercise 2: Perform Data Cleaning and Transformation Tasks
Load a dataset with missing values.
Identify the columns with missing values.
Fill the missing values with the mean of the column.



Load a dataset with duplicate rows.
Remove the duplicate rows and display the cleaned DataFrame.
'''



import pandas as pd
# Sample DataFrame with missing values
data = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': [None, 2, 3, 4],
    'C': [1, 2, 3, None]
})

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# Identify the columns with missing values
missing_values = df.isnull().sum()
print("\nColumns with missing values:")
print(missing_values[missing_values > 0])

# Fill the missing values with the mean of the column
df_filled = df.fillna(df.mean())

print("\nDataFrame after filling missing values with column means:")
print(df_filled)



# Sample DataFrame with duplicates
data = pd.DataFrame({
    'A': [1, 2, 2, 4],
    'B': [5, 6, 6, 8]
})
# Removing duplicate rows
cleaned_data = data.drop_duplicates()
print("After duplicate remove:\n",cleaned_data)