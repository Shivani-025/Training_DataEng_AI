'''
Assignment 3: Data Visualization Load the Titanic dataset.
Create box plots to identify outliers in the Age and Fare columns.
Create histograms and KDE plots to visualize the distribution of Age and Fare.
Create scatter plots to visualize the relationship between Age and Fare, and Pclass and Survived.
Use pair plots to visualize the relationships between multiple numerical features.
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

titanic_df = pd.read_csv('titanic.csv')

print(titanic_df.head())

print(titanic_df.isnull().sum())

titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)

titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)

titanic_df['Cabin'].fillna('Unknown', inplace=True)

scaler = StandardScaler()
titanic_df[['Age_standardized', 'Fare_standardized']] = scaler.fit_transform(titanic_df[['Age', 'Fare']])

minmax_scaler = MinMaxScaler()
titanic_df[['Age_normalized', 'Fare_normalized']] = minmax_scaler.fit_transform(titanic_df[['Age', 'Fare']])

fig, axs = plt.subplots(3, 2, figsize=(12, 18))

axs[0, 0].hist(titanic_df['Age'], bins=30, color='blue', alpha=0.7)
axs[0, 0].set_title('Original Age Distribution')
axs[0, 1].hist(titanic_df['Fare'], bins=30, color='blue', alpha=0.7)
axs[0, 1].set_title('Original Fare Distribution')

axs[1, 0].hist(titanic_df['Age_standardized'], bins=30, color='green', alpha=0.7)
axs[1, 0].set_title('Standardized Age Distribution')
axs[1, 1].hist(titanic_df['Fare_standardized'], bins=30, color='green', alpha=0.7)
axs[1, 1].set_title('Standardized Fare Distribution')

axs[2, 0].hist(titanic_df['Age_normalized'], bins=30, color='red', alpha=0.7)
axs[2, 0].set_title('Normalized Age Distribution')
axs[2, 1].hist(titanic_df['Fare_normalized'], bins=30, color='red', alpha=0.7)
axs[2, 1].set_title('Normalized Fare Distribution')

plt.tight_layout()
plt.show()