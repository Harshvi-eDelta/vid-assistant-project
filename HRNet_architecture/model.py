import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50

class HRNet_LandmarkDetector(nn.Module):
    def __init__(self, num_landmarks=68):
        super(HRNet_LandmarkDetector, self).__init__()
        self.hrnet = deeplabv3_resnet50(pretrained=True).backbone
        self.final_layer = nn.Conv2d(2048, num_landmarks, kernel_size=1)  # Output heatmaps

    def forward(self, x):
        #x = self.hrnet(x)['out']
        features = self.hrnet(x)
        if isinstance(features, dict):
            x = features['out']  # if exists
        else:
            x = features

        x = self.final_layer(x)
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)  # Upsample to 64x64
        return x

if __name__ == "__main__":
    model = HRNet_LandmarkDetector()
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    print(output.shape)  # Should now be [1, 68, 64, 64]
