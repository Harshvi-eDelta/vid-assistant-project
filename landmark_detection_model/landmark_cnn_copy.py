'''import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(x.size(0), x.size(1), 1, 1)
        return x * y

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 53 * 53, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # def forward(self, x):
    #     xs = self.localization(x)
    #     xs = xs.view(-1, 10 * 53 * 53)
    #     theta = self.fc_loc(xs)
    #     theta = theta.view(-1, 2, 3)
    #     grid = F.affine_grid(theta, x.size(), align_corners=False)
    #     x = F.grid_sample(x, grid, align_corners=False)
    #     return x

    def forward(self, x):
        xs = self.localization(x)
        b, c, h, w = xs.shape
        xs = xs.view(b, -1)
        if self.fc_loc[0].in_features != c * h * w:
            # Re-initialize with correct shape
            self.fc_loc[0] = nn.Linear(c * h * w, 32).to(x.device)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

class LandmarkCNN(nn.Module):
    def __init__(self):
        super(LandmarkCNN, self).__init__()
        self.stn = STN()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.se_block = SEBlock(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 136)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        #x = self.stn(x)

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = self.se_block(x)
        #print("Before flatten:", x.shape) # 👈 ADD THIS LINE
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x'''

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
        return x

'''import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

class DeepLandmarkCNN(nn.Module):
    def __init__(self, num_landmarks=68):
        super(DeepLandmarkCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = ResidualBlock(64, 128, downsample=True)
        self.layer3 = ResidualBlock(128, 128)
        self.layer4 = ResidualBlock(128, 256, downsample=True)
        self.layer5 = ResidualBlock(256, 256)

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_landmarks * 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.fc(x)
        return x'''

