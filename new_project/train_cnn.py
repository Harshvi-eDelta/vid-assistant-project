import os
import torch
import scipy.io as sio
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from landmark_cnn import LandmarkCNN
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import scipy

class AFLWDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_landmarks=21):
        self.root_dir = root_dir
        self.transform = transform
        self.num_landmarks = num_landmarks

        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.jpg')])
        self.landmark_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.mat')])

        assert len(self.image_files) == len(self.landmark_files), "Mismatch between images and landmarks!"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mat_name = self.landmark_files[idx]

        img_path = os.path.join(self.root_dir, img_name)
        mat_path = os.path.join(self.root_dir, mat_name)

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        orig_w, orig_h = image.size

        # Load landmarks
        mat_data = sio.loadmat(mat_path)

        if 'pt2d' in mat_data:
            landmarks = mat_data['pt2d'].T  # Convert shape from (2, N) to (N,2)
        else:
            raise KeyError(f"Landmarks not found in {mat_path}")

         # Convert landmarks to float before division
        landmarks = landmarks.astype(np.float32) 

        # 🔹 Ensure fixed number of landmarks (21 points)
        if landmarks.shape[0] > self.num_landmarks:
            landmarks = landmarks[:self.num_landmarks]  # Crop extra points
        elif landmarks.shape[0] < self.num_landmarks:
            pad = np.zeros((self.num_landmarks - landmarks.shape[0], 2))
            landmarks = np.vstack((landmarks, pad))  # Pad with zeros

        # Ensure valid landmarks before normalization
        landmarks[:, 0] = np.clip(landmarks[:, 0], 1e-6, orig_w)  # Avoid zero values
        landmarks[:, 1] = np.clip(landmarks[:, 1], 1e-6, orig_h)

        # Normalize
        landmarks[:, 0] /= orig_w
        landmarks[:, 1] /= orig_h

        # Clip again to strictly stay within [0,1]
        landmarks = np.clip(landmarks, 0, 1)


        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # print(f"Original landmarks: {landmarks * np.array([orig_w, orig_h])}")  # Should be in pixel coordinates
        # print(f"Normalized landmarks: {landmarks}")  # Should be in [0,1]

        return image, torch.tensor(landmarks.flatten(), dtype=torch.float32)

# Updated DataLoader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

#  Step 3: Create Dataset & DataLoader
dataset = AFLWDataset(root_dir="/Users/edelta076/Desktop/Project_VID_Assistant/new_project/AFLW2000", transform=transform)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

#  Step 4: Initialize Model & Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LandmarkCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
#criterion = nn.SmoothL1Loss()
criterion = nn.MSELoss()


# Step 5: Training Loop
num_epochs = 25
best_loss = float("inf")
patience = 5
counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
    for images, landmarks in progress_bar:
        images, landmarks = images.to(device), landmarks.to(device)
        
        # Zero gradients, forward pass, compute loss
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, landmarks)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")

    # Adjust learning rate
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

# Step 6: Save Final Model
torch.save(model.state_dict(), "landmark_model_2.pth")
print("Model saved successfully!")

# mat_path = "/Users/edelta076/Desktop/Project_VID_Assistant/new_project/AFLW2000/image00004.mat"  # Replace with an actual file path
# mat_data = scipy.io.loadmat(mat_path)

# # Print all keys in the .mat file
# print(mat_data.keys())

'''import scipy.io as sio

# Load the .mat file
mat_data = sio.loadmat("/Users/edelta076/Desktop/Project_VID_Assistant/new_project/AFLW2000/image00004.mat")

# Print all available keys
print("Keys in MAT file:", mat_data.keys())

# Print shape and type of each key (to understand data structure)
for key in mat_data:
    if not key.startswith("__"):  # Ignore meta keys
        print(f"{key}: Type={type(mat_data[key])}, Shape={mat_data[key].shape}")

#print("pt3d_68:", mat_data['pt3d_68'])  # 2D landmarks
pt2d = np.array(mat_data['pt2d']).T  # Convert (2, N) → (N, 2)
print("Reshaped pt2d:", pt2d.shape) 
print("Landmarks with -1 values:\n", pt2d[pt2d == -1])'''
