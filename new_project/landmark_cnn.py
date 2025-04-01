import torch
import torch.nn as nn
import torch.nn.functional as F

class LandmarkCNN(nn.Module):
    def __init__(self):
        super(LandmarkCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 28 * 28, 1024)  # Adjust based on input size
        self.fc2 = nn.Linear(1024, 42)  # Output: 42 landmarks (x, y) coordinates

    def forward(self, x):
        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # Max pooling after conv1
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # Max pooling after conv2
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)  # Max pooling after conv3
        
        # Flatten the output from the convolutional layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        landmarks = self.fc2(x)
        
        return landmarks
