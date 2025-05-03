'''
Question 1
Implement logistic regression to classify the Balance scale dataset into two classes using only 'L', and 'R' using PyTorch. Evaluate the model's performance using accuracy.

Steps:

Load the Balance scale dataset.
Preprocess the data: normalize the features and drop the 'B' value in target.
Split the data into training and testing sets.
Implement the logistic regression model using PyTorch.
Train the model and evaluate its performance on the test set.
Report the accuracy of the model.

'''

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Balance scale dataset
balance = pd.read_csv('balance-scale.data', header=None)

# Preprocess the data: normalize the features and drop the 'B' value in target
X = balance.iloc[:, 1:].values
y = balance.iloc[:, 0].values
y = np.where(y == 'L', 0, 1)  # Convert 'L' and 'R' to 0 and 1

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float().view(-1, 1)

# Model
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(4, 1)  # 4 features in the Balance scale dataset

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticRegressionModel()

# Define the loss function and the optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
losses = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluating Model
model.eval()
with torch.no_grad():
    y_pred_train = model(X_train_tensor).round()
    y_pred_test = model(X_test_tensor).round()

# Calculate accuracy
train_accuracy = (y_pred_train.eq(y_train_tensor).sum() / float(y_train_tensor.shape[0])).item()
test_accuracy = (y_pred_test.eq(y_test_tensor).sum() / float(y_test_tensor.shape[0])).item()
print(f'Train Accuracy: {train_accuracy * 100:.2f}%')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')