'''
Question 2:

Extend the simple neural network to include a backward propagation method.
Use a loss function and optimizer to train the neural network on a sample dataset.
Implement the training loop and monitor the loss during training.

Train the neural network on a real dataset (e.g., CIFAR-10).
Implement evaluation metrics to monitor the performance of the network.
Save and load the trained model.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Hyperparameters
input_size = 3072  # 32x32 images
hidden_size = 128
num_classes = 10  # 10 classes for CIFAR-10
num_epochs = 10
batch_size = 256
learning_rate = 0.001

# Data preprocessing and loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = SimpleNN(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
losses = []
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.view(-1, 3072)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            losses.append(loss.item())

# Plot the loss curve
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()

# Evaluating the model
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(-1, 3072)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

accuracy = 100 * correct / total
print(f'Accuracy on the test set: {accuracy:.2f}%')

# Metrics
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Visualize the first 10 test images and their predictions
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(test_loader)
images, labels = next(dataiter)
images = images[:10]
labels = labels[:10]
outputs = model(images.view(-1, 3072))
_, predicted = torch.max(outputs, 1)

# Plot the images
fig, axes = plt.subplots(1, 10, figsize=(12, 2))
for idx in range(10):
    ax = axes[idx]
    img = images[idx].numpy().squeeze()
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Pred: {predicted[idx].item()}')
    ax.axis('off')

plt.show()

# Save the model checkpoint
torch.save(model.state_dict(), 'CIFAR_model.ckpt')

#--------------------------OUTPUT---------------------------------

'''
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data\cifar-10-python.tar.gz
100.0%
Extracting ./data\cifar-10-python.tar.gz to ./data
Files already downloaded and verified
Epoch [1/10], Step [100/196], Loss: 1.6771
Epoch [2/10], Step [100/196], Loss: 1.3876
Epoch [3/10], Step [100/196], Loss: 1.3682
Epoch [4/10], Step [100/196], Loss: 1.2720
Epoch [5/10], Step [100/196], Loss: 1.3415
Epoch [6/10], Step [100/196], Loss: 1.2661
Epoch [7/10], Step [100/196], Loss: 1.1503
Epoch [8/10], Step [100/196], Loss: 1.0227
Epoch [9/10], Step [100/196], Loss: 1.0570
Epoch [10/10], Step [100/196], Loss: 1.0443
Accuracy on the test set: 52.70%
Precision: 0.5288
Recall: 0.5270
F1 Score: 0.5268
'''