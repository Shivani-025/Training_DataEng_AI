'''# Implementing Bagging
import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim


# Load the dataset
cali = load_iris()
X, y = cali.data, cali.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Function to train a single model
def train_model(model, X_train, y_train, epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()
    return model


# Train multiple models on bootstrapped samples
def bootstrap_sample(X, y):
    indices = torch.randint(0, len(X), (len(X),))
    return X[indices], y[indices]


num_models = 2
models = []
for _ in range(num_models):
    X_bootstrap, y_bootstrap = bootstrap_sample(X_train, y_train)
    model = Net()
    model = train_model(model, X_bootstrap, y_bootstrap)
    models.append(model)


# Function to make predictions with an ensemble of models
def predict(models, X):
    predictions = [model(X).detach().numpy() for model in models]
    avg_predictions = np.mean(predictions, axis=0)
    return torch.tensor(avg_predictions)


# Make predictions on the test set
y_pred = predict(models, X_test)



from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# Define a simple neural network for classification
class ClassifierNet(nn.Module):
    def __init__(self):
        super(ClassifierNet, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Function to train a single model
def train_model(model, X_train, y_train, epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    return model


# Train multiple models on bootstrapped samples
num_models = 2
models = []
for _ in range(num_models):
    X_bootstrap, y_bootstrap = bootstrap_sample(X_train, y_train)
    model = ClassifierNet()
    model = train_model(model, X_bootstrap, y_bootstrap)
    models.append(model)


# Function to make predictions with an ensemble of models
def predict(models, X):
    predictions = [model(X).detach().numpy() for model in models]
    avg_predictions = np.mean(predictions, axis=0)
    return torch.tensor(np.argmax(avg_predictions, axis=1))


# Make predictions on the test set
y_pred = predict(models, X_test)


from sklearn.metrics import accuracy_score, classification_report

# Evaluate the ensemble model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"10.Classification Report:\n{report}")

'''

#-----------------------------------------------------------------------------------------------------------
#=================================PART 2================================================


import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = load_breast_cancer()
X, y = data.data, data.target
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample


# Function to perform bagging and return trained classifiers
def bagging_classifiers(X, y, n_samples=100):
    classifiers = []
    for _ in range(n_samples):
        X_resampled, y_resampled = resample(X, y, random_state=np.random.randint(10000))
        # print(f'Res count : {len(X_resampled)}')
        clf1 = LogisticRegression(max_iter=10000)
        clf2 = DecisionTreeClassifier()
        clf1.fit(X_resampled, y_resampled)
        clf2.fit(X_resampled, y_resampled)
        classifiers.append((clf1, clf2))
    return classifiers


# Train classifiers
classifiers = bagging_classifiers(X_train, y_train, n_samples=10)


# Function to predict using the trained classifiers
def predict_with_classifiers(classifiers, X):
    predictions1 = np.zeros((len(classifiers), X.shape[0]))
    predictions2 = np.zeros((len(classifiers), X.shape[0]))
    for i, (clf1, clf2) in enumerate(classifiers):
        predictions1[i, :] = clf1.predict(X)
        predictions2[i, :] = clf2.predict(X)
    return predictions1, predictions2


# Make predictions on the test set
predictions1, predictions2 = predict_with_classifiers(classifiers, X_test)


# Function to aggregate predictions by voting
def aggregate_predictions(predictions):
    return np.round(np.mean(predictions, axis=0))


# Aggregate predictions for each classifier
aggregated_predictions1 = aggregate_predictions(predictions1)
aggregated_predictions2 = aggregate_predictions(predictions2)
# Combine predictions from both classifiers
combined_predictions = np.round((aggregated_predictions1 + aggregated_predictions2) / 2)


# Function to evaluate the model
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1


# Evaluate individual classifiers
accuracy1, precision1, recall1, f11 = evaluate_model(y_test, aggregated_predictions1)
accuracy2, precision2, recall2, f12 = evaluate_model(y_test, aggregated_predictions2)
# Evaluate combined model
accuracy_combined, precision_combined, recall_combined, f1_combined = evaluate_model(y_test, combined_predictions)
# Print metrics
print("Logistic 9.Regression Bagging:")
print(f"Accuracy: {accuracy1}, Precision: {precision1}, Recall: {recall1}, F1 Score: {f11}")
print("\nDecision Tree Bagging:")
print(f"Accuracy: {accuracy2}, Precision: {precision2}, Recall: {recall2}, F1 Score: {f12}")
print("\nCombined Model:")
print(
    f"Accuracy: {accuracy_combined}, Precision: {precision_combined}, Recall: {recall_combined}, F1 Score: {f1_combined}")

