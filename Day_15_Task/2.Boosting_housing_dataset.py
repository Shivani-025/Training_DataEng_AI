"""

Assignment 2: Implement a Gradient Boosting Regressor Using PyTorch
Dataset: Any Housing Dataset

Objective: Implement a Gradient Boosting regressor to predict housing prices using the  Housing dataset. Evaluate the model using various regression metrics (MAE, MSE, RMSE, R-squared).

Steps:
Load the Housing dataset:

Preprocess the data:
Normalize the features using StandardScaler.

Define the model:
Create a simple linear regression model using PyTorch.

Implement Gradient Boosting:
Use sklearn.ensemble.GradientBoostingRegressor to implement the Gradient Boosting ensemble method.
Train the model on the training data.

Evaluate the model:
Calculate and interpret regression metrics (MAE, MSE, RMSE, R-squared) on the test set.

"""

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load California housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

# Initialize Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Train the model
gbr.fit(X_train_scaled, y_train)

# Predictions on the test set
y_pred = gbr.predict(X_test_scaled)


# Calculate regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2 Score): {r2:.4f}")