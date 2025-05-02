# original
'''import os
import torch
import numpy as np
import torchfile  # This is key for .t7 files
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class LandmarkDataset(Dataset):
    def __init__(self, img_dir, t7_dir, transform=None):
        self.img_dir = img_dir
        self.t7_dir = t7_dir
        self.image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image_path = os.path.join(self.img_dir, img_name)
        t7_path = os.path.join(self.t7_dir, os.path.splitext(img_name)[0] + '.t7')

        # Load image
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size

        # Load .t7 landmark
        landmark = torchfile.load(t7_path)  # (68, 2)
        landmark = np.array(landmark).astype(np.float32)

        # Normalize landmarks by original image size
        landmark[:, 0] /= original_width   # normalize x
        landmark[:, 1] /= original_height  # normalize y

        # Transform image
        if self.transform:
            image = self.transform(image)

        landmark = torch.tensor(landmark, dtype=torch.float32).view(-1)  # flatten to (136,)
        return image, landmark


def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
'''

# trying new 
import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchfile
import torchvision.transforms as transforms
import cv2

def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

def generate_heatmap(landmark, heatmap_size=64, image_size=256, sigma=1.5):          # 1.5,2.0
    num_landmarks = landmark.shape[0]
    heatmaps = np.zeros((num_landmarks, heatmap_size, heatmap_size), dtype=np.float32)

    for i in range(num_landmarks):
        x = int(landmark[i][0] / 256 * heatmap_size)
        y = int(landmark[i][1] / 256 * heatmap_size)


        if x < 0 or y < 0 or x >= heatmap_size or y >= heatmap_size:
            continue

        # Create meshgrid
        xx, yy = np.meshgrid(np.arange(heatmap_size), np.arange(heatmap_size))
        heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        heatmap = heatmap.astype(np.float32)

        # Normalize to [0,1]
        heatmap = heatmap / np.max(heatmap)
        heatmaps[i] = heatmap

    return heatmaps

class LandmarkHeatmapDataset(Dataset):
    def __init__(self, img_dir, t7_dir, transform=None):
        self.img_dir = img_dir
        self.t7_dir = t7_dir
        self.image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image_path = os.path.join(self.img_dir, img_name)
        t7_path = os.path.join(self.t7_dir, os.path.splitext(img_name)[0] + '.t7')

        # Load image and landmarks
        image = Image.open(image_path).convert("RGB")
        landmark = torchfile.load(t7_path)
        #landmark = np.array(landmark).astype(np.float32)

        landmark = np.array(landmark).astype(np.float32)

        # Resize landmarks from original scale to 256x256 scale
        original_width, original_height = 256, 256  # target image size
        input_width, input_height = image.size      # original image size BEFORE transform

        scale_x = original_width / input_width
        scale_y = original_height / input_height

        landmark[:, 0] *= scale_x
        landmark[:, 1] *= scale_y

        if self.transform:
            image = self.transform(image)

        # Generate heatmaps
        heatmaps = generate_heatmap(landmark)

        return image, torch.tensor(heatmaps, dtype=torch.float32)
        if idx == 0:
            print("Landmark sample values (raw):", landmark.shape)
            print("First 5 landmarks:", landmark[:5])


