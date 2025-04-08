import os
import torch
import torchfile
import numpy as np
from torch.utils.data import Dataset
import cv2
from PIL import Image  # ✅ Needed to convert NumPy to PIL

class LandmarkDataset(Dataset):
    def __init__(self, image_dir, landmark_dir, transform=None):
        self.image_dir = image_dir
        self.landmark_dir = landmark_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        self.landmark_files = sorted(os.listdir(landmark_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        landmark_path = os.path.join(self.landmark_dir, self.landmark_files[idx])

        # Load image using OpenCV (BGR to RGB)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert image from NumPy to PIL
        image = Image.fromarray(image)

        # Apply transform (e.g., Resize + ToTensor)
        if self.transform:
            image = self.transform(image)

        # Load .t7 landmark file
        landmarks = torchfile.load(landmark_path)
        landmarks = torch.tensor(landmarks, dtype=torch.float32)  # shape: (68, 2)

        # Flatten landmarks to (136,) and normalize (assuming original image was 256x256)
        landmarks /= 256.0
        landmarks = landmarks.view(-1)  # Shape: (136,)

        return image, landmarks
