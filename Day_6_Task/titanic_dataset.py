'''
Assignment 1: Handling Missing Data and Scaling Features Load the Titanic dataset from a CSV file.
Identify and handle missing values in the Age, Embarked, and Cabin columns using different imputation methods.
Standardize the numerical features (Age, Fare) using StandardScaler.
Normalize the numerical features using MinMaxScaler. Compare the distributions of the scaled features using histograms.
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

titanic = pd.read_csv('Titanic-Dataset.csv')

titanic.columns = [col.capitalize() for col in titanic.columns]

print(titanic.head())

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x=titanic['Age'])
plt.title('Box Plot of Age')

plt.subplot(1, 2, 2)
sns.boxplot(x=titanic['Fare'])
plt.title('Box Plot of Fare')

plt.show()

plt.figure(figsize=(14, 12))

plt.subplot(2, 2, 1)
sns.histplot(titanic['Age'].dropna(), kde=True)
plt.title('Histogram and KDE of Age')

plt.subplot(2, 2, 2)
sns.histplot(titanic['Fare'], kde=True)
plt.title('Histogram and KDE of Fare')

plt.show()

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=titanic['Age'], y=titanic['Fare'])
plt.title('Scatter Plot of Age vs Fare')

plt.subplot(1, 2, 2)
sns.scatterplot(x=titanic['Pclass'], y=titanic['Survived'])
plt.title('Scatter Plot of Pclass vs Survived')

plt.show()

sns.pairplot(titanic[['Age', 'Fare', 'Pclass', 'Survived']], hue='Survived')
plt.show()


