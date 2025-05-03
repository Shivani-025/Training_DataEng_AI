'''
Exercise 3: Merge and Join Multiple DataFrames
Load two DataFrames, df1 and df2, with a common column id.
Perform an inner join on the id column.
Display the merged DataFrame.

Load two DataFrames, df1 and df2, with a common column id.
Perform an outer join on the id column.
Display the merged DataFrame.

Load two DataFrames with the same columns.
Concatenate the DataFrames vertically.
Display the concatenated DataFrame.

df1 = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'name': ['Shivani', 'Shivu', 'Shiv', 'Shivi'],
    'age': [24, 27, 22, 32]
})

# Create the second DataFrame
df2 = pd.DataFrame({
    'id': [3, 4, 5, 6],
    'city': ['Indore', 'Delhi', 'Bhopal', 'Ujjain'],
    'salary': [70000, 80000, 60000, 75000]
})

'''

import pandas as pd

# Load two DataFrames with a common column id
df1 = pd.read_csv('df1.csv')
df2 = pd.read_csv('df2.csv')

# Perform an inner join on the id column
inner_join_df = pd.merge(df1, df2, on='id', how='inner')

print("Inner join DataFrame:")
print(inner_join_df)


# Perform an outer join on the id column
outer_join_df = pd.merge(df1, df2, on='id', how='outer')

print("\n\nOuter join DataFrame:")
print(outer_join_df)


# Concatenate the DataFrames vertically
concat_df = pd.concat([df1, df2], ignore_index=True)

print("\n\nConcatenated DataFrame:")
print(concat_df)
