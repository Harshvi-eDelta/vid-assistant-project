import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms
from tqdm import tqdm
from landmark_cnn import LandmarkCNN

# Dataset Class
class LandmarkDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.landmarks_frame.iloc[idx, 0])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # Load landmarks
        landmarks = self.landmarks_frame.iloc[idx, 1:].values.astype("float32").reshape(-1, 2)

        # Normalize landmarks
        #landmarks[:, 0] /= orig_w
        #landmarks[:, 1] /= orig_h

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(landmarks.flatten(), dtype=torch.float32)

# Data Transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset
dataset = LandmarkDataset(csv_file="/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/landmarks_normalized.csv", 
                          img_dir="/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/original_jpg", 
                          transform=transform)

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = LandmarkCNN().to(device)
#criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Training Loop with Progress Bar
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for images, landmarks in progress_bar:
        images, landmarks = images.to(device), landmarks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, landmarks)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN/Inf loss detected at epoch {epoch+1}! Stopping training.")
            break

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        progress_bar.set_postfix(loss=loss.item())

    print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {running_loss/len(train_loader):.4f}")
    scheduler.step(running_loss)

# Save Model
torch.save(model.state_dict(), "landmark_model.pth")
print("Model saved successfully!")
