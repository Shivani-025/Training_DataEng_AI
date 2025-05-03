"""

Assignment 2: Performing Polynomial 9.Regression and Feature Engineering

Problem 1: Load a dataset and perform polynomial regression. Add polynomial features up to degree 3 and train the model using these features.
Problem 2: Compare the performance of the polynomial regression model with the linear regression model using MSE.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
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

# Add polynomial features up to degree 3
poly_features = PolynomialFeatures(degree=3)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)


# Define the linear regression model
class LinearRegression(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearRegression,self).__init__()
        self.linear = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        return self.linear(x)


# Initialize the model, loss function, and optimizer
model = LinearRegression(input_dim=13,output_dim=1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

# Train the model
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(torch.tensor(X_train,dtype=torch.float32))
    loss = criterion(outputs,torch.tensor(y_train,dtype=torch.float32))
    loss.backward()
    optimizer.step()
    print("Epoch:",epoch + 1,"Loss:",loss.item())

# Evaluate the model on the test set
model.eval()
test_outputs = model(torch.tensor(X_test,dtype=torch.float32))
test_loss = criterion(test_outputs,torch.tensor(y_test,dtype=torch.float32))
print("Test Loss:",test_loss.item())

# Train the polynomial regression model
model_poly = LinearRegression(input_dim=13*3+1, output_dim=1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model_poly.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model_poly(torch.tensor(X_train_poly, dtype=torch.float32))
    loss = criterion(outputs, torch.tensor(y_train, dtype=torch.float32))
    loss.backward()
    optimizer.step()
    print("Epoch:", epoch+1, "Loss:", loss.item())


# Evaluate the polynomial regression model on the test set
model_poly.eval()
test_outputs = model_poly(torch.tensor(X_test_poly, dtype=torch.float32))
test_loss = criterion(test_outputs, torch.tensor(y_test, dtype=torch.float32))
print("Test Loss (Polynomial 9.Regression):", test_loss.item())

# Compare the performance of the polynomial regression model with the linear regression model
print("MSE (Linear 9.Regression):", test_loss.item())
print("MSE (Polynomial 9.Regression):", test_loss.item())