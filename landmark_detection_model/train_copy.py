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
from torch.utils.data import DataLoader, random_split
from data_preprocessing_copy import LandmarkDataset, get_transforms
from landmark_cnn_copy import LandmarkCNN
from tqdm import tqdm
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
img_dir = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/original_jpg_copy"
t7_dir = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/t7"
save_path = "best_model.pth"

# Load full dataset
full_dataset = LandmarkDataset(img_dir, t7_dir, transform=get_transforms())

# Split into train & validation
val_ratio = 0.1
val_size = int(val_ratio * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Model, Loss, Optimizer
model = LandmarkCNN().to(device)
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
best_val_loss = float("inf")
epochs = 30

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] - Training")

    for images, landmarks in loop:
        images, landmarks = images.to(device), landmarks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, landmarks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix(train_loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, landmarks in val_loader:
            images, landmarks = images.to(device), landmarks.to(device)
            outputs = model(images)
            loss = criterion(outputs, landmarks)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"\nEpoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        print("Best model saved.\n")


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

