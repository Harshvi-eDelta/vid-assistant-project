import torch
import torch.nn as nn

class LandmarkCNN(nn.Module):
    def __init__(self):
        super(LandmarkCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 56 * 56, 512),
            nn.ReLU(),
            nn.Linear(512, 136)  # Output: 68 landmarks * 2 (x, y)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        #print("Feature map shape before flattening:", x.shape)  # Debugging step
        x = x.view(x.shape[0], -1)
        #print("Flattened feature size:", x.shape)  # Should match FC input size
        x = self.fc_layers(x)
        return x


def get_model():
    return LandmarkCNN()
