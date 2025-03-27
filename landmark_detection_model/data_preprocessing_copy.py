import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
import torchvision.transforms as transforms

class LandmarkDataset(Dataset):
    def __init__(self, img_dir, t7_dir, transform=None):
        self.img_dir = img_dir
        self.t7_dir = t7_dir
        self.transform = transform

        # Collect all image files
        self.image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # Load corresponding .t7 file
        t7_name = os.path.splitext(img_name)[0] + ".t7"  # Match image name
        t7_path = os.path.join(self.t7_dir, t7_name)

        if not os.path.exists(t7_path):
            raise FileNotFoundError(f"Landmark file not found for image: {img_name}")

        landmarks = torch.load(t7_path)  # Load landmarks from .t7 file
        landmarks = landmarks.numpy().astype("float32").reshape(-1, 2)

        # Normalize landmarks
        landmarks[:, 0] /= orig_w
        landmarks[:, 1] /= orig_h

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(landmarks.flatten(), dtype=torch.float32)

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset
dataset = LandmarkDataset(
    img_dir="/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/original_jpg",
    t7_dir="/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/t7",
    transform=transform
)

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
