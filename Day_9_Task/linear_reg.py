"""
Assignment 1: Implementing Linear 9.Regression with PyTorch
Problem 1: Load a dataset (such as the Boston Housing dataset) and implement a simple linear regression model to predict housing prices.
Use Mean Squared Error (MSE) as the loss function.
Problem 2: Plot the predicted vs actual values for the training and test sets.
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from urllib.request import urlretrieve
from pathlib import Path
import matplotlib.pyplot as plt


# Problem 1: Load a dataset (such as the Boston Housing dataset) and implement a simple linear regression model to predict housing prices.
# Use Mean Squared Error (MSE) as the loss function.
# Define a function to download the Boston Housing dataset
def download_boston_housing_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
    data_file = Path("housing.data")
    if not data_file.is_file():
        urlretrieve(url, data_file)
    return str(data_file)

# Load the Boston Housing dataset
data_file = download_boston_housing_data()
data = np.loadtxt(data_file)
X = data[:, :-1]
y = data[:, -1]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # reshape to column vector
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)  # reshape to column vector


# Define the linear regression model
class LinearRegression(nn.Module):
    def __init__(self,input_size,output_size):
        super(LinearRegression,self).__init__()
        self.linear = nn.Linear(input_size,output_size)

    def forward(self,x):
        return self.linear(x)


# Initialize the model
input_size = X.shape[1]
output_size = 1
model = LinearRegression(input_size,output_size)

# Define the loss function (Mean Squared Error)
criterion = nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

# Training the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs,y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Problem 2: Plot the predicted vs actual values for the training and test sets.
# Plotting predicted vs actual values
with torch.no_grad():
    predicted_train = model(X_train_tensor).numpy()
    predicted_test = model(X_test_tensor).numpy()

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Training Set')
plt.scatter(y_train, predicted_train, color='blue')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.subplot(1, 2, 2)
plt.title('Test Set')
plt.scatter(y_test, predicted_test, color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.tight_layout()
plt.show()