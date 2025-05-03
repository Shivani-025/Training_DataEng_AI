"""

Assignment 3: Applying Ridge and Lasso 9.Regression

Problem 1: Implement Ridge regression on the same dataset used in Assignment 1. Use cross-validation to select the best regularization parameter (alpha).
Problem 2: Implement Lasso regression on the same dataset. Use cross-validation to select the best regularization parameter (alpha).
Problem 3: Compare the performance of Ridge, Lasso, and standard linear regression models in terms of MSE and interpret the results.
"""


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV

# Problem 1: Implement Ridge regression on the same dataset used in Assignment 1. Use cross-validation to select the
# best regularization parameter (alpha).
# Load the Boston Housing dataset
data = np.loadtxt("housing.data")
X = data[:, :-1]
y = data[:, -1]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Ridge regression model with cross-validation
ridge_model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)

# Fit the Ridge regression model
ridge_model.fit(X_train, y_train)

# Best alpha selected by cross-validation
best_alpha_ridge = ridge_model.alpha_
print(f"\nBest alpha for Ridge 9.Regression: {best_alpha_ridge}")

# Predictions
predicted_train_ridge = ridge_model.predict(X_train)
predicted_test_ridge = ridge_model.predict(X_test)

# Calculate MSE for Ridge regression model
mse_ridge_train = mean_squared_error(y_train, predicted_train_ridge)
mse_ridge_test = mean_squared_error(y_test, predicted_test_ridge)
print(f"Ridge 9.Regression MSE (Train): {mse_ridge_train}")
print(f"Ridge 9.Regression MSE (Test): {mse_ridge_test}")


# Problem 2: Implement Lasso regression on the same dataset. Use cross-validation to select the best regularization parameter (alpha).

from sklearn.linear_model import LassoCV

# Initialize Lasso regression model with cross-validation
lasso_model = LassoCV(alphas=[0.1, 1.0, 10.0], cv=5)

# Fit the Lasso regression model
lasso_model.fit(X_train, y_train)

# Best alpha selected by cross-validation
best_alpha_lasso = lasso_model.alpha_
print(f"\nBest alpha for Lasso 9.Regression: {best_alpha_lasso}")

# Predictions
predicted_train_lasso = lasso_model.predict(X_train)
predicted_test_lasso = lasso_model.predict(X_test)

# Calculate MSE for Lasso regression model
mse_lasso_train = mean_squared_error(y_train, predicted_train_lasso)
mse_lasso_test = mean_squared_error(y_test, predicted_test_lasso)
print(f"Lasso 9.Regression MSE (Train): {mse_lasso_train}")
print(f"Lasso 9.Regression MSE (Test): {mse_lasso_test}")


# Problem 3: Compare the performance of Ridge, Lasso, and standard linear regression models in terms of MSE and interpret the results.

# Standard Linear 9.Regression
from sklearn.linear_model import LinearRegression

# Initialize the standard linear regression model
linear_model = LinearRegression()

# Fit the standard linear regression model
linear_model.fit(X_train, y_train)

# Predictions
predicted_train_linear = linear_model.predict(X_train)
predicted_test_linear = linear_model.predict(X_test)

# Calculate MSE for standard linear regression model
mse_linear_train = mean_squared_error(y_train, predicted_train_linear)
mse_linear_test = mean_squared_error(y_test, predicted_test_linear)
print(f"\nLinear 9.Regression MSE (Train): {mse_linear_train}")
print(f"Linear 9.Regression MSE (Test): {mse_linear_test}")

# Comparison
print("\nComparing Performance:\n")
print(f"Linear 9.Regression MSE (Train): {mse_linear_train}")
print(f"Ridge 9.Regression MSE (Train): {mse_ridge_train}")
print(f"Lasso 9.Regression MSE (Train): {mse_lasso_train}")
print(f"\nLinear 9.Regression MSE (Test): {mse_linear_test}")
print(f"Ridge 9.Regression MSE (Test): {mse_ridge_test}")
print(f"Lasso 9.Regression MSE (Test): {mse_lasso_test}")