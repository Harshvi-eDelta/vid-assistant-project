import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

image_folder = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/original_jpg"
csv_file = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/landmarks_normalized.csv"

df = pd.read_csv(csv_file)

# Dataset Class for loading images & landmarks
class LandmarkDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None):
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_name = row.iloc[0]
        image_path = os.path.join(self.image_folder, image_name)

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_name}")
            return None  
        orig_h, orig_w = image.shape[:2]  

        # Resize image to 224x224
        image_resized = cv2.resize(image, (224, 224))
        image_resized = image_resized.astype(np.float32) / 255.0  

        image_resized = np.transpose(image_resized, (2, 0, 1))  # Convert to (C, H, W)
        image_resized = torch.tensor(image_resized, dtype=torch.float32)

        # Load normalized landmarks (already in 0-1 range)
        landmarks = row.iloc[1:].values.astype(np.float32).reshape(-1, 2)  # Shape (68, 2)

        landmarks = landmarks.flatten()  
        landmarks = torch.tensor(landmarks, dtype=torch.float32)

        return image_resized, landmarks

# Split dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = LandmarkDataset(train_df, image_folder)
val_dataset = LandmarkDataset(val_df, image_folder)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("Data Preprocessing Completed Successfully!")
