
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. Device Configuration
# Check if CUDA is available, else use CPU
device = torch.device("cpu")
print(f'Using device: {device}')

# 2. Data Loading and Preprocessing

# Define transformations: Convert images to tensors and normalize
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL images or numpy arrays to tensors
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std for MNIST
])

# Specify the local directory where MNIST will be stored
local_data_dir = '.'

# Download and load the training dataset (set download=True only once)
train_dataset = datasets.MNIST(root=local_data_dir, train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Download and load the testing dataset (set download=True only once)
test_dataset = datasets.MNIST(root=local_data_dir, train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 3. Define the Enhanced Convolutional Neural Network (CNN) Model

class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        # First Convolutional Layer: 1 input channel (grayscale), 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second Convolutional Layer: 32 input channels, 64 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third Convolutional Layer: 64 input channels, 128 output channels, 3x3 kernel
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully Connected Layer 1: 128*3*3 input features, 256 output features
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout1 = nn.Dropout(0.5)
        
        # Fully Connected Layer 2: 256 input features, 128 output features
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        
        # Output Layer: 128 input features, 10 output features (classes)
        self.fc3 = nn.Linear(128, 10)
        
        # Max Pooling Layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Activation Function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # First Convolutional Block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # Downsample to 14x14
        
        # Second Convolutional Block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)  # Downsample to 7x7
        
        # Third Convolutional Block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)  # Downsample to 3x3
        
        # Flatten the tensor for Fully Connected Layers
        x = x.view(-1, 128 * 3 * 3)
        
        # First Fully Connected Layer with Dropout
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        # Second Fully Connected Layer with Dropout
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        # Output Layer
        x = self.fc3(x)
        return x

# Instantiate the CNN model and move it to the appropriate device
model = EnhancedCNN().to(device)
print(model)

# 4. Define the Loss Function and Optimizer

# Cross-Entropy Loss for multi-class classification
criterion = nn.CrossEntropyLoss()

# Adam Optimizer for efficient training
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training the Model

num_epochs = 1  # Number of epochs to train

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)  # Move data to device
        
        # Forward pass: Compute predicted outputs by passing inputs to the model
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward pass: Compute gradient of the loss with respect to model parameters
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()        # Backpropagation
        optimizer.step()       # Update the parameters
        
        running_loss += loss.item()
        
        if (batch_idx + 1) % 100 == 0:
            avg_loss = running_loss / 100
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {avg_loss:.4f}')
            running_loss = 0.0

# 6. Evaluating the Model

model.eval()  # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # No need to compute gradients during evaluation
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')
