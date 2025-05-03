'''
Assignment 2: Encoding Categorical Variables and Feature Engineering Load the Iris dataset from a CSV file.
Perform one-hot encoding on the Species column. Perform label encoding on the Species column and compare the results.
Create a new feature PetalArea by multiplying PetalLength and PetalWidth.
Create a new feature SepalArea by multiplying SepalLength and SepalWidth.
'''


import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load the Iris dataset from a CSV file
df = pd.read_csv('Iris.csv')

# Perform one-hot encoding on the Species column
one_hot_encoder = OneHotEncoder()
species_one_hot = one_hot_encoder.fit_transform(df[['Species']])

# Perform label encoding on the Species column
label_encoder = LabelEncoder()
df['Species_label'] = label_encoder.fit_transform(df['Species'])

# Create a new feature PetalArea by multiplying PetalLength and PetalWidth
df['PetalArea'] = df['PetalLength'] * df['PetalWidth']

# Create a new feature SepalArea by multiplying SepalLength and SepalWidth
df['SepalArea'] = df['SepalLength'] * df['SepalWidth']

print("Original DataFrame:")
print(df.head())

print("\nOne-Hot Encoded Species:")
print(species_one_hot.toarray())

print("\nLabel Encoded Species:")
print(df['Species_label'])