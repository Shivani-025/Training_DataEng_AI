'''
Exercise 1: Implement Grid Search on a 9.Regression Model Using PyTorch
Dataset: California Housing Dataset

Objective: Implement grid search to tune hyperparameters of a regression model using the California Housing dataset. Evaluate the model using various regression metrics (MAE, MSE, RMSE, R-squared).

Steps:
Load the California Housing dataset:
Use sklearn.datasets.fetch_california_housing to load the dataset.

Preprocess the data:
Normalize the features using StandardScaler.

Define the model and hyperparameters:
Create a simple linear regression model using PyTorch.
Define a grid of hyperparameters to search (e.g., learning rate, number of epochs).

Implement grid search:
Use sklearn.model_selection.GridSearchCV to perform grid search.
Evaluate the model using cross-validation metrics.

Report the best hyperparameters and model performance:
Calculate and interpret regression metrics (MAE, MSE, RMSE, R-squared) for the best model.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid

# Load the California Housing dataset
california_housing = fetch_california_housing()
X = california_housing.data
y = california_housing.target

# Preprocess the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model and hyperparameters
class SimpleLinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleLinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

input_size = X.shape[1]
output_size = 1
model = SimpleLinearRegression(input_size, output_size)

param_grid = {
    'lr': [0.01, 0.001],
    'batch_size': [16, 32],
    'epochs': [300, 600]
}

# Implement grid search
def train_model(model, criterion, optimizer, dataloader):
    model.train()
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            preds = outputs.squeeze()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    return mean_squared_error(all_labels, all_preds)

best_params = None
best_score = float('inf')
results = []
best_params_list = []

# Perform grid search
for params in ParameterGrid(param_grid):
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # Train the model
    for epoch in range(params['epochs']):
        train_model(model, criterion, optimizer, train_loader)

    # Evaluate the model
    mse = evaluate_model(model, test_loader)

    # Store the results
    results.append((params, mse))
    print(f' Res :  \n {results}')

    if mse < best_score:
        best_score = mse
        best_params_list = [params]
    elif mse == best_score:
        best_params_list.append(params)

# Print all results
for params, mse in results:
    print(f'Params: {params} => MSE: {mse:.4f}')

print(f'Best MSE: {best_score:.4f}')
print('Best parameter combinations:')
for params in best_params_list:
    print(params)

# Evaluate the best model
best_model = SimpleLinearRegression(input_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params_list[0]['lr'])

for epoch in range(best_params_list[0]['epochs']):
    train_model(best_model, criterion, optimizer, train_loader)

y_pred = best_model(torch.from_numpy(X_test).float()).detach().numpy()
y_pred = y_pred.squeeze()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)