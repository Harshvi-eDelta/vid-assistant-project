# landmark_cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        # Squeeze operation: global average pooling
        self.fc1 = nn.Linear(channels, channels // reduction)  # Reduced dimension
        self.fc2 = nn.Linear(channels // reduction, channels)  # Restored to original dimension

    def forward(self, x):
        # Global average pooling (squeeze step)
        y = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        # Fully connected layers (excitation step)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(x.size(0), x.size(1), 1, 1)  # Reshape to match input size
        # Recalibrate the input with the learned attention weights
        return x * y

class LandmarkCNN(nn.Module):
    def __init__(self):
        super(LandmarkCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2, dilation=2)

        # BatchNorm layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Squeeze-and-Excitation Block
        self.se_block = SEBlock(128)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 136)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Apply Squeeze-and-Excitation block
        x = self.se_block(x)

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output layer (for landmark coordinates)
        x = torch.sigmoid(x)  # Normalize output between 0 and 1
        return x
