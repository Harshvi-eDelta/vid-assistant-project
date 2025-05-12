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

import torch
import torch.nn as nn

'''class LandmarkCNN(nn.Module):
    def __init__(self, num_landmarks=68):
        super(LandmarkCNN, self).__init__()
        self.num_landmarks = num_landmarks

        # Shared Feature Extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 256x256
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x128

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Stage 1: Takes only features
        self.stage1 = self.make_stage(256, num_landmarks)

        # Stages 2-5: Take [features + previous heatmap]
        self.stage2 = self.make_stage(256 + num_landmarks, num_landmarks)
        self.stage3 = self.make_stage(256 + num_landmarks, num_landmarks)
        self.stage4 = self.make_stage(256 + num_landmarks, num_landmarks)
        self.stage5 = self.make_stage(256 + num_landmarks, num_landmarks)

    def make_stage(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )

    def forward(self, x):
        features = self.features(x)

        out1 = self.stage1(features)
        out2 = self.stage2(torch.cat([features, out1], dim=1))
        out3 = self.stage3(torch.cat([features, out2], dim=1))
        out4 = self.stage4(torch.cat([features, out3], dim=1))
        out5 = self.stage5(torch.cat([features, out4], dim=1))

        return out1, out2, out3, out4, out5'''

class LandmarkCNN(nn.Module):
    def __init__(self, num_landmarks=68):
        super(LandmarkCNN, self).__init__()
        self.num_landmarks = num_landmarks

        # Shared Feature Extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 256x256
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x128

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Stage 1: Takes only features
        self.stage1 = self.make_stage(256, num_landmarks)

        # Stages 2-5: Take [features + previous heatmap], resize previous heatmaps to match feature map size
        self.stage2 = self.make_stage(256 + num_landmarks, num_landmarks)
        self.stage3 = self.make_stage(256 + num_landmarks, num_landmarks)
        self.stage4 = self.make_stage(256 + num_landmarks, num_landmarks)
        self.stage5 = self.make_stage(256 + num_landmarks, num_landmarks)

    def make_stage(self, in_channels, out_channels, target_size=128):
        return nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, out_channels, kernel_size=1),
            nn.Upsample(size=(target_size, target_size), mode='bilinear', align_corners=False)  # Upsample to 128x128
        )

    def forward(self, x):
        # 1. Extract features from the input image (x)
        features = self.features(x)

        # 2. Pass the features through stage 1 to predict the first heatmap
        out1 = self.stage1(features)

        # 3. Upsample out1 to ensure it matches the size we want for concatenation (128x128)
        out1_resized = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)(out1)

        # 4. Resize features to match the size of out1_resized (128x128) for concatenation
        features_resized = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)(features)

        # 5. Concatenate the resized features with the first heatmap (out1_resized) along the channel dimension
        out2 = self.stage2(torch.cat([features_resized, out1_resized], dim=1))

        # 6. Repeat the same process for stages 3, 4, and 5
        out2_resized = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)(out2)
        out3 = self.stage3(torch.cat([features_resized, out2_resized], dim=1))

        out3_resized = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)(out3)
        out4 = self.stage4(torch.cat([features_resized, out3_resized], dim=1))

        out4_resized = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)(out4)
        out5 = self.stage5(torch.cat([features_resized, out4_resized], dim=1))

        # 7. Return the outputs of all 5 stages
        return out1, out2, out3, out4, out5







