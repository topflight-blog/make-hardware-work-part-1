import torch
from torchvision import datasets, transforms

# Define transformations: Convert images to tensors and normalize
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std for MNIST
])

# Specify the local directory where MNIST will be stored
local_data_dir = '.'

# Download and load the training dataset
train_dataset = datasets.MNIST(root=local_data_dir, train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Download and load the testing dataset
test_dataset = datasets.MNIST(root=local_data_dir, train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
