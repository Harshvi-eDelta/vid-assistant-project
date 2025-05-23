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
from torch.utils.data import DataLoader,random_split
from tqdm import tqdm
import os
from landmark_cnn_copy import LandmarkCNN  # Your model with 2 outputs: heatmap1 and heatmap2
from data_preprocessing_copy import LandmarkHeatmapDataset, get_transforms
from torch.utils.tensorboard import SummaryWriter

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Paths
train_img_dir = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/original_jpg_copy"
train_t7_dir = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/t7"
save_path = 'best_model_4.pth'

full_dataset = LandmarkHeatmapDataset(train_img_dir, train_t7_dir, transform=get_transforms())

# Validation split
val_ratio = 0.2
train_size = int((1 - val_ratio) * len(full_dataset))
val_size = len(full_dataset) - train_size
#train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# Model
model = LandmarkCNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

writer = SummaryWriter(log_dir="runs/landmark_experiment")
best_val_loss = float('inf')
num_epochs = 50     # 45

for epoch in range(num_epochs):
    model.train()       # sets model in the training mode
    running_loss = 0.0

    for images, heatmaps in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images = images.to(device)
        heatmaps = heatmaps.to(device)

        optimizer.zero_grad()        # clears all the gradients before passing to the forward 
        output1, output2, output3, output4, output5 = model(images)     # forward pass 

        loss1 = criterion(output1, heatmaps)
        loss2 = criterion(output2, heatmaps)
        loss3 = criterion(output3, heatmaps)
        loss4 = criterion(output4, heatmaps)
        loss5 = criterion(output5, heatmaps)  # FIXED

        # loss = loss1 + loss2 + loss3 + loss4 + loss5
        loss = 0.05 * loss1 + 0.1 * loss2 + 0.2 * loss3 + 0.3 * loss4 + 0.35 * loss5
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, heatmaps in val_loader:
            images = images.to(device)
            heatmaps = heatmaps.to(device)

            output1, output2, output3, output4, output5 = model(images)
            loss1 = criterion(output1, heatmaps)
            loss2 = criterion(output2, heatmaps)
            loss3 = criterion(output3, heatmaps)
            loss4 = criterion(output4, heatmaps)
            loss5 = criterion(output5, heatmaps)

            #  Adjusted weights to favor later stages
            # loss = 0.05 * loss1 + 0.1 * loss2 + 0.2 * loss3 + 0.3 * loss4 + 0.35 * loss5
            loss = 0.05 * loss1 + 0.1 * loss2 + 0.2 * loss3 + 0.3 * loss4 + 0.35 * loss5
            val_loss += loss.item()

    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")
    writer.add_scalars("Loss", {"Train": train_loss, "Validation": val_loss}, epoch + 1)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
        print("Saved Best Model")

writer.close()
