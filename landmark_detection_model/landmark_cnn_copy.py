# original
'''import torch
import torch.nn as nn
import torch.nn.functional as F

class LandmarkCNN(nn.Module):
    def __init__(self, num_landmarks=68):
        super(LandmarkCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_landmarks * 2)  # 68 landmarks × 2 (x, y)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 128x128
        x = self.pool(F.relu(self.conv2(x)))  # 64x64
        x = self.pool(F.relu(self.conv3(x)))  # 32x32
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x'''

# trying new 
import torch
import torch.nn as nn
import torch.nn.functional as F

class LandmarkCNN(nn.Module):
    def __init__(self, num_landmarks=68):
        super(LandmarkCNN, self).__init__()

        # Shared feature extractor (Backbone)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   # Input channels: 3 (RGB), Output: 32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                  # Downsample to 128x128

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                  # Downsample to 64x64

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Stage 1 Head – First prediction of heatmaps
        self.stage1_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),  # Refine features
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_landmarks, 1)    # Final output shape: (B, 68, 64, 64)
        )

        # Stage 2 Head – Second refinement stage
        self.stage2_refine = nn.Sequential(
            nn.Conv2d(128 + num_landmarks, 64, 3, padding=1),  # Concatenate features + heatmap1
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_landmarks, 1)                     # Output shape: (B, 68, 64, 64)
        )

    def forward(self, x):
        features = self.backbone(x)            # Shared CNN features → shape: [B, 128, 64, 64]

        heatmap1 = self.stage1_head(features)  # Stage 1 prediction → shape: [B, 68, 64, 64]

        # Concatenate features and heatmap1 along channel dimension
        concat = torch.cat([features, heatmap1], dim=1)  # Shape: [B, 128 + 68, 64, 64]

        heatmap2 = self.stage2_refine(concat)  # Stage 2 refined heatmap → shape: [B, 68, 64, 64]

        return heatmap1, heatmap2



