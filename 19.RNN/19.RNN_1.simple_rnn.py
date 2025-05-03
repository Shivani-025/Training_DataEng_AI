import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# Sample dataset
class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# Define the 19.RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Get the output from the last time step
        return out


# Hyperparameters
input_size = 10  # Number of features in the input
hidden_size = 20
output_size = 2  # Number of output classes
num_layers = 2
num_epochs = 10
learning_rate = 0.001
# Generate some dummy data
sequences = torch.randn(100, 5, input_size)  # 100 sequences of length 5
targets = torch.randint(0, 2, (100,))  # Binary classification
# Create dataset and dataloader
dataset = SequenceDataset(sequences, targets)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
# Instantiate the model, loss function, and optimizer
model = SimpleRNN(input_size, hidden_size, output_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Training loop
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
print("Training complete")
# Save the model
torch.save(model.state_dict(), 'simple_rnn_model.pth')
# Load the model
modele = SimpleRNN(input_size, hidden_size, output_size, num_layers)
modele.load_state_dict(torch.load('simple_rnn_model.pth'))
modele.eval()
# Make a prediction
with torch.no_grad():
    sample_input = torch.randn(1, 5, input_size)  # Single sequence of length 5
    print(f' SI :  {sample_input}')
    prediction = modele(sample_input)
    print(f' Pred : {prediction}')
    predicted_class = torch.argmax(prediction, dim=1)
    print(f'Predicted class: {predicted_class.item()}')

'''
Training Loop:
For each epoch, the model processes the training data in batches.
The loss is calculated using the criterion (CrossEntropyLoss).
The optimizer updates the model parameters to minimize the loss.
After each epoch, the average loss for that epoch is printed.
Model Saving:
After training, the model's state dictionary is saved to a file named simple_rnn_model.pth.
Model Loading:
The saved model state dictionary is loaded back into the model.
Prediction:
A random sample input sequence is generated.
The model makes a prediction for this input.
The predicted class is printed.
'''
#--------------------output--------------------------
'''
Epoch [1/10], Loss: 0.6060
Epoch [2/10], Loss: 0.6095
Epoch [3/10], Loss: 0.6426
Epoch [4/10], Loss: 0.5473
Epoch [5/10], Loss: 0.5868
Epoch [6/10], Loss: 0.6377
Epoch [7/10], Loss: 0.5870
Epoch [8/10], Loss: 0.5992
Epoch [9/10], Loss: 0.5250
Epoch [10/10], Loss: 0.3926
Training complete
 SI :  tensor([[[-1.1933e+00, -9.6122e-01,  1.5089e+00, -3.8471e-01, -6.6960e-01,
          -2.1611e+00,  3.8230e-01,  1.8958e+00,  1.6136e-01,  1.4844e+00],
         [ 5.0352e-01, -5.5937e-01, -1.6156e+00,  1.0619e+00, -1.0058e+00,
           7.9100e-01, -6.9221e-01,  3.9725e-01, -9.8556e-01, -9.0800e-02],
         [ 8.9681e-01, -1.3169e+00,  4.4923e-01, -1.2312e+00,  3.6022e-02,
           7.3896e-01, -1.0136e+00,  1.4505e-03,  6.5635e-01, -1.4017e+00],
         [-1.8461e+00, -3.0257e-01, -2.1200e-02, -2.4435e-01, -2.3152e-01,
          -1.1050e-01,  1.5852e-01, -2.0592e+00,  2.1136e+00,  2.9914e-01],
         [-6.4705e-01,  7.7020e-01, -2.9205e-01, -6.7657e-01,  4.5194e-01,
           3.4706e-01,  1.0696e+00, -1.0878e-01,  8.0408e-02,  1.0416e+00]]])
 Pred : tensor([[-0.3605,  0.2264]])
Predicted class: 1

'''