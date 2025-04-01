
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import cv2
import scipy.io as sio
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class LandmarkDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.jpg')])  # Get only JPG files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        # Load image and convert to RGB
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)  # Convert to PIL format

        # Get original dimensions
        orig_w, orig_h = image.size

        # Load corresponding .mat file
        mat_name = os.path.splitext(img_name)[0] + ".mat"
        mat_path = os.path.join(self.root_dir, mat_name)

        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"Landmark file not found: {mat_name}")

        # Load landmarks from .mat file
        mat_data = sio.loadmat(mat_path)

        if "pt2d" in mat_data:
            landmarks = mat_data["pt2d"]  # Shape: (2, N)
            landmarks = landmarks.T  # Convert to (N,2)
        else:
            raise KeyError(f"Landmark key 'pt2d' not found in {mat_name}")

        # Normalize landmarks
        landmarks[:, 0] /= orig_w  # Normalize X
        landmarks[:, 1] /= orig_h  # Normalize Y

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(landmarks.flatten(), dtype=torch.float32)

# Define Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure images are resized properly
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load Dataset from a **single** folder
dataset = LandmarkDataset(
    root_dir="/Users/edelta076/Desktop/Project_VID_Assistant/new_project/AFLW2000",
    transform=transform
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

def visualize_landmarks(image, landmarks):
    # Convert PyTorch tensor to NumPy
    image = image.permute(1, 2, 0).cpu().numpy()  # (3, 224, 224) → (224, 224, 3)

    # Reverse normalization
    image = (image * 0.5) + 0.5  
    image = np.clip(image, 0, 1)  # Ensure values are valid for imshow()

    # Ensure landmarks are a NumPy array
    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.detach().cpu().numpy()

    # Reshape from (42,) → (21,2)
    landmarks = landmarks.reshape(-1, 2)

    # Convert normalized landmarks back to pixel space
    h, w, _ = image.shape
    landmarks[:, 0] *= w
    landmarks[:, 1] *= h

    # Plot image & landmarks
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=10)
    plt.show()

# Test with dataset sample
sample_image, sample_landmarks = dataset[0]
visualize_landmarks(sample_image, sample_landmarks)

# to view .mat file
'''mat_path = "/Users/edelta076/Desktop/Project_VID_Assistant/new_project/AFLW2000/image00002.mat"  # Replace with an actual file path
mat_data = scipy.io.loadmat(mat_path)

# Print all keys in the .mat file
print(mat_data.keys())'''