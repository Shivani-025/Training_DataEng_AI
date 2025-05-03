'''Assignment 1: Implement a Bagging Regressor Using PyTorch
Dataset: Diabetes Dataset

Objective: Implement a Bagging regressor to predict diabetes progression using the Diabetes dataset. Evaluate the model using various regression metrics (MAE, MSE, RMSE, R-squared).

Steps:

Load the Diabetes dataset:
Use sklearn.datasets.load_diabetes to load the dataset.

Preprocess the data:
Normalize the features using StandardScaler.

Define the model:
Create a simple linear regression model using PyTorch.

Implement Bagging:
Use sklearn.ensemble.BaggingRegressor to implement the Bagging ensemble method.
Train multiple models on different subsets of the training data.

Evaluate the model:
Calculate and interpret regression metrics (MAE, MSE, RMSE, R-squared) on the test set.'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the Diabetes dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Implement Bagging with the custom regressor

bagging_regressor = BaggingRegressor(n_estimators=10, random_state=42)
bagging_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = bagging_regressor.predict(X_test)

# Calculate and print regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
