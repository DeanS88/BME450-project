import os
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import requests
from io import StringIO
import joblib 

# Load data from CSV file
github_url = 'https://raw.githubusercontent.com/DeanS88/BME450-project/main/cardio_train.csv'
response = requests.get(github_url)

data = np.loadtxt(StringIO(response.text), delimiter=';', skiprows=1)

github_url = 'https://raw.githubusercontent.com/DeanS88/BME450-project/main/cardio_train.csv'
file_path = 'cardio_train.csv'
data = np.loadtxt(file_path, delimiter=';', skiprows=1)

# Separate features and labels
X = data[:, 1:-1]  # All columns except the last one are features
y = data[:, -1]   # Last column is the target

# Standardize input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Convert data to PyTorch tensors
train_inputs = torch.tensor(X_train, dtype=torch.float32)
test_inputs = torch.tensor(X_test, dtype=torch.float32)
train_targets = torch.tensor(y_train, dtype=torch.long)  # Assuming targets are integers
test_targets = torch.tensor(y_test, dtype=torch.long)

train_indices = train_inputs[:,0]

# Create PyTorch datasets and dataloaders
train_dataset = TensorDataset(train_inputs, train_targets)
test_dataset = TensorDataset(test_inputs, test_targets)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(11, 128),  # Change input size to 11
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),   # Assuming 2 output classes for binary classification
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

sample_num = 12
print('Input Index: ', train_indices[sample_num].tolist(),'\n')
print('Expected Output: ', train_targets[sample_num].tolist(), '\n')

# Initialize model, loss function, and optimizer
model = NeuralNetwork()
learning_rate = 1e-2
loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

# Training loop
def train_loop(dataloader, model, loss_fn, optimizer):
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

# Testing loop
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")

# Training the model
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(test_loader, model, loss_fn)
print("Done!")

with torch.no_grad():
    # Get the output from the model for the sample
    output = model(train_inputs[sample_num, :].unsqueeze(0))  # Unsqueeze to add batch dimension
    # Apply softmax to get probabilities
    probabilities = torch.softmax(output, dim=1)
    
    # Get the predicted class label (0 or 1)
    predicted_class = torch.argmax(output, dim=1).item()
    
    # Print the predicted class label
    print("Predicted class:", predicted_class)

model_state_dict = model.state_dict()

# Save the state dictionary to a file
torch.save(model_state_dict, 'final_weights.pth')
joblib.dump(scaler, 'scaler.pkl')
