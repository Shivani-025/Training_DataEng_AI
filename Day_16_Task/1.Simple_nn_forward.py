'''
Question 1:

Install PyTorch and set up the environment.
Build a simple neural network with one hidden layer using PyTorch.
Implement forward propagation for the neural network.
Print the architecture and output of the network for a given input tensor.

Implement a method to monitor the training progress by plotting the loss curve.
Experiment with different learning rates and batch sizes to improve the performance of the neural network.
Implement an early stopping mechanism to halt training when the validation loss stops improving.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class SimpleNN(nn.Module):
    def _init_(self, input_size, hidden_size, output_size):
        super(SimpleNN, self)._init_()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Example: Instantiate the network with input size of 3, hidden layer size of 5, and output size of 1
input_size = 3
hidden_size = 5
output_size = 1

model = SimpleNN(input_size, hidden_size, output_size)
print(model)

# Create a dummy input tensor with the appropriate input size
dummy_input = torch.randn(1, input_size)

# Perform forward propagation
output = model(dummy_input)
print(f"Input: {dummy_input}")
print(f"Output: {output}")

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy dataset (inputs and targets)
inputs = torch.randn(100, input_size)
targets = torch.randn(100, output_size)

# Training loop
num_epochs = 100
losses = []

for epoch in range(num_epochs):
    model.train()

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    losses.append(loss.item())

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Plot the loss curve
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()


def train_model(lr, batch_size):
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Training loop with mini-batches
    num_epochs = 100
    losses = []
    n_samples = inputs.size(0)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        # Mini-batch gradient descent
        for i in range(0, n_samples, batch_size):
            batch_inputs = inputs[i:i + batch_size]
            batch_targets = targets[i:i + batch_size]

            # Forward pass
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss / (n_samples / batch_size))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / (n_samples / batch_size):.4f}")

    # Plot the loss curve
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curve (lr={lr}, batch_size={batch_size})')
    plt.show()


# Example: Train with different learning rates and batch sizes
train_model(lr=0.01, batch_size=10)
train_model(lr=0.1, batch_size=20)


def train_model_with_early_stopping(lr, batch_size, patience):
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Split data into training and validation sets
    val_split = 0.2
    val_size = int(n_samples * val_split)
    train_inputs, val_inputs = inputs[:-val_size], inputs[-val_size:]
    train_targets, val_targets = targets[:-val_size], targets[-val_size:]

    # Training loop with early stopping
    num_epochs = 100
    losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        # Mini-batch gradient descent
        for i in range(0, train_inputs.size(0), batch_size):
            batch_inputs = train_inputs[i:i + batch_size]
            batch_targets = train_targets[i:i + batch_size]

            # Forward pass
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss / (train_inputs.size(0) / batch_size))

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets).item()
            val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / (train_inputs.size(0) / batch_size):.4f}, Val Loss: {val_loss:.4f}")

    # Plot the loss curve
    plt.plot(losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss Curve (lr={lr}, batch_size={batch_size})')
    plt.legend()
    plt.show()


# Example: Train with early stopping
train_model_with_early_stopping(lr=0.01, batch_size=10, patience=10)

#--------------------------------------OUTPUT-----------------------------------

