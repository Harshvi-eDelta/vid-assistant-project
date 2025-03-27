'''import torch
import cv2
import numpy as np
import os
from torchvision.transforms import Compose, ToTensor, Normalize

input_folder = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/original_jpg"  
output_folder = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/depth_maps"

os.makedirs(output_folder, exist_ok=True)

# Load MiDaS model
model_type = "DPT_Large"  # Use "MiDaS_large" for better quality
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

# Define transformation (without resizing)
transform = Compose([
    ToTensor(),  
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# Process images
for image_name in os.listdir(input_folder):
    if image_name.endswith(".jpg"):
        image_path = os.path.join(input_folder, image_name)

        # Read image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Convert to tensor
        input_tensor = transform(image_rgb).unsqueeze(0)

        # Predict depth
        with torch.no_grad():
            depth_map = midas(input_tensor)

        # Convert depth map to numpy
        depth_map = depth_map.squeeze().cpu().numpy()

        # Normalize depth values to range 0-255
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
        depth_map = depth_map.astype(np.uint8)

        # Save depth map
        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, depth_map)
        print(f"Saved depth map: {output_path}")

print("All depth maps have been generated successfully!")'''

import os
import cv2
import torch
import numpy as np
import timm
# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MiDaS (pre trained )model
model_type = "DPT_Hybrid"  # Change to "DPT_Large" or "MiDaS_small" if needed
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device)
midas.eval()

# Load MiDaS transform
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform if "DPT" in model_type else midas_transforms.small_transform

# Input and output folders
input_folder = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/original_jpg"  
output_folder = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/depth_maps_1"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(input_folder, filename)

        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Warning: Could not load {filename}, skipping...")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = cv2.resize(image, (384, 384))  # Resize to 384x384 (DPT models expect this size)
        input_tensor = transform(image).to(device)

        # Generate depth map
        with torch.no_grad():
            depth_map = midas(input_tensor)  # Directly get the depth output

        depth_map = depth_map.squeeze().cpu().numpy()  # Convert to numpy

        # Normalize depth map for visualization
        depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

        # Save depth map
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, (depth_map_normalized * 255).astype(np.uint8))
        print(f"Saved depth map: {output_path}")

print("All depth maps generated successfully!")




