'''import os
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

    # def __getitem__(self, idx):
    #     img_name = self.image_files[idx]
    #     img_path = os.path.join(self.img_dir, img_name)

    #     # Load image
    #     image = cv2.imread(img_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     orig_h, orig_w = image.shape[:2]

    #     # Load corresponding .t7 file
    #     t7_name = os.path.splitext(img_name)[0] + ".t7"
    #     t7_path = os.path.join(self.t7_dir, t7_name)

    #     if not os.path.exists(t7_path):
    #         raise FileNotFoundError(f"Landmark file not found for image: {img_name}")

    #     # Load landmarks
    #     landmarks = torch.load(t7_path)
    #     if isinstance(landmarks, torch.Tensor):
    #         landmarks = landmarks.numpy()  # Convert Tensor to NumPy if needed
    #     elif not isinstance(landmarks, np.ndarray):
    #         raise ValueError(f"Unexpected data format in {t7_path}, expected NumPy array or Tensor.")

    #     landmarks = landmarks.astype("float32").reshape(-1, 2)
    #     landmarks = landmarks / 256.0
    #     # Normalize landmarks
    #     landmarks[:, 0] /= orig_w
    #     landmarks[:, 1] /= orig_h

    #     print(f" Image: {img_name}, Landmarks: {landmarks[:5]}")  # Print first 5 landmarks for debugging

    #     # Apply transformations
    #     if self.transform:
    #         image = self.transform(image)

    #     return image, torch.tensor(landmarks.flatten(), dtype=torch.float32)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load landmarks
        t7_name = os.path.splitext(img_name)[0] + ".t7"
        t7_path = os.path.join(self.t7_dir, t7_name)

        if not os.path.exists(t7_path):
            raise FileNotFoundError(f"Landmark file not found for image: {img_name}")

        landmarks = torch.load(t7_path)
        if isinstance(landmarks, torch.Tensor):
            landmarks = landmarks.numpy()
        elif not isinstance(landmarks, np.ndarray):
            raise ValueError(f"Unexpected format in {t7_path}")

        # Just reshape and leave as normalized [0,1]
        landmarks = landmarks.astype("float32").reshape(-1, 2)

        # Resize image
        image = cv2.resize(image, (256, 256))

        # Apply image transform (ToTensor, Normalize)
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(landmarks.flatten(), dtype=torch.float32)


# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset
dataset = LandmarkDataset(
    img_dir="/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/original_jpg_copy",
    t7_dir="/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/t7",
    transform=transform
)

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)'''

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
        transforms.RandomRotation(degrees=10),                  # Small rotation
        transforms.RandomHorizontalFlip(),                      # Flip faces
        transforms.ColorJitter(brightness=0.2, contrast=0.2),   # Vary lighting
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

