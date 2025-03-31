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
from PIL import Image
import matplotlib.pyplot as plt

# Dataset Class
class LandmarkDataset(Dataset):
    def __init__(self, img_dir, pth_dir, transform=None):
        self.img_dir = img_dir
        self.pth_dir = pth_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Load image (OpenCV loads as NumPy array)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ✅ Convert OpenCV image (NumPy) to PIL format (Fix for TypeError)
        image = Image.fromarray(image)

        orig_h, orig_w = image.size

        # Load landmarks
        pth_name = os.path.splitext(img_name)[0] + ".pth"
        pth_path = os.path.join(self.pth_dir, pth_name)
        if not os.path.exists(pth_path):
            raise FileNotFoundError(f"Landmark file not found for image: {img_name}")

        landmarks = torch.load(pth_path, weights_only=False).astype("float32").reshape(-1, 2)

        # Normalize landmarks (convert to range [0,1])
        landmarks[:, 0] /= orig_w
        landmarks[:, 1] /= orig_h

        # Apply transforms (PIL image is required)
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(landmarks.flatten(), dtype=torch.float32)

# Data Transformations
'''transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])'''

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset
dataset = LandmarkDataset(
    img_dir="/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/original_jpg_copy",
    pth_dir="/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/t7_fixed",  # Updated to use pth_dir
    transform=transform
)

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = LandmarkCNN().to(device)
criterion = nn.SmoothL1Loss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Training Loop
best_loss = float("inf")
patience = 5  
counter = 0  

num_epochs = 15
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent exploding gradients
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")

    scheduler.step(avg_loss)  

    # Early stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        counter = 0  
        torch.save(model.state_dict(), "best_landmark_model.pth")  # Save best model
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered!")
            break  

# Save Final Model
torch.save(model.state_dict(), "landmark_model.pth")
print("Model saved successfully!")
