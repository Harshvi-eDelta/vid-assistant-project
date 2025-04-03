import os
import cv2
import scipy.io
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class FaceLandmarkDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.data = []
        
        for file in os.listdir(dataset_path):
            if file.endswith(".jpg"):
                img_path = os.path.join(dataset_path, file)
                mat_path = img_path.replace(".jpg", ".mat")

                if os.path.exists(mat_path):
                    mat_data = scipy.io.loadmat(mat_path)

                    # Ensure 'pt2d' key exists
                    if "pt2d" not in mat_data:
                        raise KeyError(f"Landmark key 'pt2d' not found in {mat_path}")

                    landmarks = mat_data["pt2d"].T  # Shape: (68,2)

                    # Load grayscale image
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    # Resize image FIRST to match CNN input size
                    # img_resized = cv2.resize(img, (96, 96))
                    # h, w = 96, 96  # New size after resizing
                    #h, w = img.shape[:2]
                    h,w = 224,224
                    # Normalize landmarks using the resized image size
                    landmarks[:, 0] /= w  # Normalize x
                    landmarks[:, 1] /= h  # Normalize y

                    self.data.append((img, landmarks.flatten()))  # Flatten (68,2) → (136,)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, landmarks = self.data[idx]

        # Convert image to PIL format before applying transforms
        image = Image.fromarray(img)

        if self.transform:
            image = self.transform(image)

        #return image, torch.tensor(landmarks, dtype=torch.float32)
        return image, torch.tensor(landmarks, dtype=torch.float32)  # Keep as (68,2)


# Define Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),  
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def get_dataloader(dataset_path, batch_size=32):
    dataset = FaceLandmarkDataset(dataset_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
