'''import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms
from tqdm import tqdm
from landmark_cnn_copy import LandmarkCNN
#from landmark_cnn_copy import DeepLandmarkCNN
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

        # Convert OpenCV image (NumPy) to PIL format (Fix for TypeError)
        image = Image.fromarray(image)

        orig_w, orig_h = image.size

        # Load landmarks
        pth_name = os.path.splitext(img_name)[0] + ".pth"
        pth_path = os.path.join(self.pth_dir, pth_name)
        if not os.path.exists(pth_path):
            raise FileNotFoundError(f"Landmark file not found for image: {img_name}")

        # landmarks = torch.load(pth_path, weights_only=False).astype("float32").reshape(-1, 2)
        landmarks = torch.load(pth_path).astype("float32").reshape(-1, 2)

        # Normalize landmarks (convert to range [0,1])
        landmarks[:, 0] /= 224
        landmarks[:, 1] /= 224

        # Apply transforms (PIL image is required)
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(landmarks.flatten(), dtype=torch.float32)

# Data Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Works with PIL
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# Load dataset
dataset = LandmarkDataset(
    img_dir="/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/original_jpg_copy",
    pth_dir="/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/t7_fixed",  # Updated to use pth_dir
    transform=transform
)

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
# model = LandmarkCNN().to(device)
model = LandmarkCNN().to(device)
criterion = nn.SmoothL1Loss() 
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Training Loop
best_loss = float("inf")
# patience = 5  
patience = 7  
counter = 0  

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for images, landmarks in progress_bar:
        # print("GT min:", landmarks.min().item(), "GT max:", landmarks.max().item())
        # break  # Only need to check one batch       
        #print("Image size:", images.shape)  # 👈 ADD THIS LINE
        images, landmarks = images.to(device), landmarks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        #print("Model output:", outputs[0])
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
        torch.save(model.state_dict(), "best_landmark_model_2.pth")  # Save best model
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered!")
            break  

# Save Final Model
torch.save(model.state_dict(), "landmark_model_2.pth")
print("Model saved successfully!")'''

# best for training less chnage in google image
'''import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_preprocessing_copy import LandmarkDataset, get_transforms
from landmark_cnn_copy import LandmarkCNN
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
img_dir = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/original_jpg_copy"
t7_dir = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/t7"
save_path = "best_model.pth"

# Dataset and Loader
dataset = LandmarkDataset(img_dir, t7_dir, transform=get_transforms())
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model, Loss, Optimizer
model = LandmarkCNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_loss = float("inf")

epochs = 30

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}")
    for images, landmarks in loop:
        images, landmarks = images.to(device), landmarks.to(device)

        outputs = model(images)
        loss = criterion(outputs, landmarks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), save_path)
        print("Best model saved.")'''

# trying new 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_preprocessing_copy import LandmarkDataset, get_transforms
from landmark_cnn_copy import LandmarkCNN
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
img_dir = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/original_jpg_copy"
t7_dir = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/t7"
save_path = "best_model.pth"

# Dataset and Loader
dataset = LandmarkDataset(img_dir, t7_dir, transform=get_transforms())
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model, Loss, Optimizer
model = LandmarkCNN().to(device)
#criterion = nn.MSELoss()
criterion = torch.nn.SmoothL1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_loss = float("inf")

epochs = 35

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}")
    for images, landmarks in loop:
        images, landmarks = images.to(device), landmarks.to(device)

        outputs = model(images)
        loss = criterion(outputs, landmarks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), save_path)
        print("Best model saved.")



# using validation set
'''import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data_preprocessing_copy import LandmarkDataset, get_transforms
from landmark_cnn_copy import LandmarkCNN

# Paths to your dataset
img_dir = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/original_jpg"
t7_dir = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/t7"
save_path = "best_model.pth"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
full_dataset = LandmarkDataset(img_dir, t7_dir, transform=get_transforms())

# Validation split
val_ratio = 0.2
train_size = int((1 - val_ratio) * len(full_dataset))
val_size = len(full_dataset) - train_size
#train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize model, loss, optimizer
model = LandmarkCNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
best_val_loss = float("inf")
num_epochs = 12

for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for images, landmarks in loop:
        images, landmarks = images.to(device), landmarks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, landmarks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(train_loss=loss.item())

    avg_train_loss = running_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, landmarks in val_loader:
            images, landmarks = images.to(device), landmarks.to(device)
            outputs = model(images)
            loss = criterion(outputs, landmarks)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"\nEpoch {epoch} ➤ Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        print(" Best model saved.")

print("\n Training Complete!")'''

