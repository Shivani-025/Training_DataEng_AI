'''
DAY 17

Assignment Question: Building and Evaluating a Neural Network for Housing Price Prediction

Tasked with building a neural network model to predict housing prices based on a dataset provided.

Dataset Description:
Dataset containing various features of houses (e.g., number of bedrooms, size of the house, location, etc.) and their corresponding prices.

Model Building:

Neural Network Architecture: Design a neural network architecture using PyTorch with the following specifications:
Input layer that matches the number of features in the dataset.
At least two hidden layers with advanced activation functions such as Leaky ReLU or ELU.
Output layer for predicting the house price.

Regularization Techniques: Implement at least two regularization techniques:
L1.or L2 regularization
Apply dropout regularization to one of the hidden layers.

Training and Optimization:
Optimization Algorithms: Train your neural network using the following optimization algorithms:
Adam
SGD with momentum
Compare their performance in terms of convergence speed and final prediction accuracy.

Evaluation:
Performance Metrics: Evaluate your trained model using appropriate metrics such as Mean Squared Error (MSE) or R-squared score.
Visualization: Visualize the predicted house prices against the actual prices to understand the model's predictive capabilities.
'''
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# 1. Data Preparation
# Load dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)


# Create custom dataset
class HousingDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create DataLoader for training and validation
train_dataset = HousingDataset(X_train, y_train)
val_dataset = HousingDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# 2. Neural Network Architecture
class HousingPriceNN(nn.Module):
    def __init__(self, input_dim):
        super(HousingPriceNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)  # Dropout regularization

    def forward(self, x):
        x = F.elu(self.fc1(x))  # Hidden layer with ELU activation
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)  # Hidden layer with Leaky ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = self.fc3(x)  # Output layer
        return x


input_dim = X_tensor.shape[1]
model = HousingPriceNN(input_dim)

# 3. Training and Optimization
# Loss function
criterion = nn.MSELoss()

# Optimizers
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
optimizer_sgd = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # Lower learning rate for SGD


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50, grad_clip=None):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if grad_clip:  # Apply gradient clipping
                torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')


# Train with Adam
print("Training with Adam Optimizer")
train_model(model, train_loader, val_loader, criterion, optimizer_adam, epochs=50)

# Reset model parameters before training with SGD
model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

# Train with SGD with momentum
print("Training with SGD Optimizer")
train_model(model, train_loader, val_loader, criterion, optimizer_sgd, epochs=50,
            grad_clip=1.0)  # Apply gradient clipping


# 4. Evaluation and Visualization
def evaluate_model(model, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            predictions.append(outputs.numpy())
            actuals.append(targets.numpy())
    predictions = np.concatenate(predictions).flatten()
    actuals = np.concatenate(actuals).flatten()
    return predictions, actuals


# Get predictions and actual values
pred_train, actual_train = evaluate_model(model, train_loader)
pred_val, actual_val = evaluate_model(model, val_loader)

# Calculate performance metrics
mse_train = mean_squared_error(actual_train, pred_train)
r2_train = r2_score(actual_train, pred_train)
mse_val = mean_squared_error(actual_val, pred_val)
r2_val = r2_score(actual_val, pred_val)

print(f'Training MSE: {mse_train:.4f}, R2: {r2_train:.4f}')
print(f'Validation MSE: {mse_val:.4f}, R2: {r2_val:.4f}')

# Visualization
plt.figure(figsize=(10, 5))
plt.scatter(actual_val, pred_val, label='Predicted vs Actual')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.show()


#------------------------OUTPUT--------------------------------
'''
Training with Adam Optimizer
Epoch 1/50, Loss: 0.8300, Val Loss: 0.4579
Epoch 2/50, Loss: 0.5808, Val Loss: 0.4347
Epoch 3/50, Loss: 0.5384, Val Loss: 0.3991
Epoch 4/50, Loss: 0.5168, Val Loss: 0.3894
Epoch 5/50, Loss: 0.4796, Val Loss: 0.3759
Epoch 6/50, Loss: 0.4566, Val Loss: 0.3783
Epoch 7/50, Loss: 0.4435, Val Loss: 0.3684
Epoch 8/50, Loss: 0.4289, Val Loss: 0.3618
Epoch 9/50, Loss: 0.4216, Val Loss: 0.3570
Epoch 10/50, Loss: 0.4094, Val Loss: 0.3527
Epoch 11/50, Loss: 0.4137, Val Loss: 0.3568
Epoch 12/50, Loss: 0.4011, Val Loss: 0.3483
Epoch 13/50, Loss: 0.3919, Val Loss: 0.3669
Epoch 14/50, Loss: 0.3919, Val Loss: 0.3478
Epoch 15/50, Loss: 0.3927, Val Loss: 0.3494
Epoch 16/50, Loss: 0.3941, Val Loss: 0.3508
Epoch 17/50, Loss: 0.3896, Val Loss: 0.3378
Epoch 18/50, Loss: 0.3867, Val Loss: 0.3413
Epoch 19/50, Loss: 0.3840, Val Loss: 0.3599
Epoch 20/50, Loss: 0.3851, Val Loss: 0.3380
Epoch 21/50, Loss: 0.3839, Val Loss: 0.3386
Epoch 22/50, Loss: 0.3856, Val Loss: 0.3379
Epoch 23/50, Loss: 0.3807, Val Loss: 0.3323
Epoch 24/50, Loss: 0.3880, Val Loss: 0.3359
Epoch 25/50, Loss: 0.3697, Val Loss: 0.3303
Epoch 26/50, Loss: 0.3745, Val Loss: 0.3376
Epoch 27/50, Loss: 0.3766, Val Loss: 0.3418
Epoch 28/50, Loss: 0.3690, Val Loss: 0.3201
Epoch 29/50, Loss: 0.3699, Val Loss: 0.3196
Epoch 30/50, Loss: 0.3751, Val Loss: 0.3227
Epoch 31/50, Loss: 0.3721, Val Loss: 0.3276
Epoch 32/50, Loss: 0.3693, Val Loss: 0.3185
Epoch 33/50, Loss: 0.3703, Val Loss: 0.3340
Epoch 34/50, Loss: 0.3691, Val Loss: 0.3221
Epoch 35/50, Loss: 0.3635, Val Loss: 0.3238
Epoch 36/50, Loss: 0.3685, Val Loss: 0.3182
Epoch 37/50, Loss: 0.3676, Val Loss: 0.3237
Epoch 38/50, Loss: 0.3674, Val Loss: 0.3201
Epoch 39/50, Loss: 0.3737, Val Loss: 0.3165
Epoch 40/50, Loss: 0.3664, Val Loss: 0.3198
Epoch 41/50, Loss: 0.3644, Val Loss: 0.3224
Epoch 42/50, Loss: 0.3629, Val Loss: 0.3147
Epoch 43/50, Loss: 0.3639, Val Loss: 0.3263
Epoch 44/50, Loss: 0.3694, Val Loss: 0.3119
Epoch 45/50, Loss: 0.3619, Val Loss: 0.3181
Epoch 46/50, Loss: 0.3568, Val Loss: 0.3244
Epoch 47/50, Loss: 0.3630, Val Loss: 0.3182
Epoch 48/50, Loss: 0.3618, Val Loss: 0.3099
Epoch 49/50, Loss: 0.3610, Val Loss: 0.3210
Epoch 50/50, Loss: 0.3570, Val Loss: 0.3156
Training with SGD Optimizer
Epoch 1/50, Loss: 0.7936, Val Loss: 0.4583
Epoch 2/50, Loss: 0.5196, Val Loss: 0.4349
Epoch 3/50, Loss: 0.4747, Val Loss: 0.4112
Epoch 4/50, Loss: 0.4562, Val Loss: 0.4054
Epoch 5/50, Loss: 0.4414, Val Loss: 0.3998
Epoch 6/50, Loss: 0.4340, Val Loss: 0.3836
Epoch 7/50, Loss: 0.4315, Val Loss: 0.3849
Epoch 8/50, Loss: 0.4177, Val Loss: 0.3744
Epoch 9/50, Loss: 0.4100, Val Loss: 0.3756
Epoch 10/50, Loss: 0.4126, Val Loss: 0.3641
Epoch 11/50, Loss: 0.4021, Val Loss: 0.3675
Epoch 12/50, Loss: 0.4041, Val Loss: 0.3588
Epoch 13/50, Loss: 0.3944, Val Loss: 0.3647
Epoch 14/50, Loss: 0.3968, Val Loss: 0.3674
Epoch 15/50, Loss: 0.3988, Val Loss: 0.3567
Epoch 16/50, Loss: 0.3914, Val Loss: 0.3571
Epoch 17/50, Loss: 0.3877, Val Loss: 0.3499
Epoch 18/50, Loss: 0.3880, Val Loss: 0.3482
Epoch 19/50, Loss: 0.3894, Val Loss: 0.3460
Epoch 20/50, Loss: 0.3895, Val Loss: 0.3448
Epoch 21/50, Loss: 0.4008, Val Loss: 0.3441
Epoch 22/50, Loss: 0.3865, Val Loss: 0.3458
Epoch 23/50, Loss: 0.3865, Val Loss: 0.3419
Epoch 24/50, Loss: 0.3834, Val Loss: 0.3419
Epoch 25/50, Loss: 0.3871, Val Loss: 0.3379
Epoch 26/50, Loss: 0.3989, Val Loss: 0.3382
Epoch 27/50, Loss: 0.3898, Val Loss: 0.3421
Epoch 28/50, Loss: 0.3864, Val Loss: 0.3407
Epoch 29/50, Loss: 0.3825, Val Loss: 0.3486
Epoch 30/50, Loss: 0.3786, Val Loss: 0.3377
Epoch 31/50, Loss: 0.3845, Val Loss: 0.3420
Epoch 32/50, Loss: 0.3755, Val Loss: 0.3360
Epoch 33/50, Loss: 0.3737, Val Loss: 0.3324
Epoch 34/50, Loss: 0.3800, Val Loss: 0.3421
Epoch 35/50, Loss: 0.3772, Val Loss: 0.3370
Epoch 36/50, Loss: 0.3783, Val Loss: 0.3348
Epoch 37/50, Loss: 0.3765, Val Loss: 0.3295
Epoch 38/50, Loss: 0.3859, Val Loss: 0.3297
Epoch 39/50, Loss: 0.3790, Val Loss: 0.3302
Epoch 40/50, Loss: 0.3756, Val Loss: 0.3351
Epoch 41/50, Loss: 0.3743, Val Loss: 0.3436
Epoch 42/50, Loss: 0.3718, Val Loss: 0.3301
Epoch 43/50, Loss: 0.3685, Val Loss: 0.3285
Epoch 44/50, Loss: 0.3726, Val Loss: 0.3292
Epoch 45/50, Loss: 0.3685, Val Loss: 0.3294
Epoch 46/50, Loss: 0.3730, Val Loss: 0.3288
Epoch 47/50, Loss: 0.3716, Val Loss: 0.3344
Epoch 48/50, Loss: 0.3743, Val Loss: 0.3272
Epoch 49/50, Loss: 0.3697, Val Loss: 0.3327
Epoch 50/50, Loss: 0.3714, Val Loss: 0.3218
Training MSE: 0.3134, R2: 0.7656
Validation MSE: 0.3218, R2: 0.7545
'''