import torch
import cv2
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import glob
import torch.nn.functional as F

def generate_gaussian_heatmap(x, y, heatmap_size=64, sigma=1.5):
            xx, yy = np.meshgrid(np.arange(heatmap_size), np.arange(heatmap_size))
            heatmap = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
            return heatmap

class FacialLandmarkDataset(Dataset):
    def __init__(self, data_dir, img_size=256, heatmap_size=64, num_landmarks=68, transform=None):
        self.data_dir = data_dir
        self.img_size = img_size
        self.heatmap_size = heatmap_size  # Ensure GT matches model output
        self.num_landmarks = num_landmarks
        self.transform = transform
        
        # Find all JPG images
        self.images = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
        
        # Extract corresponding .mat files
        self.mat_files = [img_path.replace(".jpg", ".mat") for img_path in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mat_path = self.mat_files[idx]

        # Load image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0  # Normalize
        img = (img - 0.5) / 0.5

        # Load landmarks from .mat file
        mat_data = sio.loadmat(mat_path)
        landmarks = mat_data['pt2d']  # Shape: (68, 2)
        landmarks = landmarks * (self.heatmap_size / self.img_size)  # Scale to heatmap size
        landmarks = landmarks.T  # Convert (2, 68) → (68, 2)

            # Convert landmarks to heatmap
        heatmaps = np.zeros((self.num_landmarks, self.heatmap_size, self.heatmap_size), dtype=np.float32)

        for i, (x, y) in enumerate(landmarks):
            if 0 <= x < self.heatmap_size and 0 <= y < self.heatmap_size:
                heatmaps[i] = generate_gaussian_heatmap(x, y, self.heatmap_size)

        heatmaps = torch.tensor(heatmaps, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)

        return img, heatmaps

def get_dataloader(data_dir, batch_size=16):
    dataset = FacialLandmarkDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=3)
