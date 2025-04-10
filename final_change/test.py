import torch
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
from model import LandmarkCNN
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Trained Model ===
model = LandmarkCNN().to(device)
model.load_state_dict(torch.load("checkpoints/landmark_epoch49.pth", map_location=device))  # adjust epoch as needed
model.eval()

# === Transform (same as training) ===
transform = transforms.Compose([
    transforms.ToTensor()
])

# === Test on your own image ===
test_image_path = "/Users/edelta076/Desktop/Project_VID_Assistant/face_Images/fimg2.jpg"  #  your custom image path
img_bgr = cv2.imread(test_image_path)

if img_bgr is None:
    raise FileNotFoundError(f"Image not found at: {test_image_path}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (256, 256))
h_original, w_original = img_rgb.shape[:2]

input_tensor = transform(img_resized).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor).cpu().numpy().reshape(-1, 2)

# === Denormalize landmarks to original image size ===
output[:, 0] *= w_original / 256.0
output[:, 1] *= h_original / 256.0

# === Visualize ===
plt.figure(figsize=(4, 4))
plt.imshow(img_rgb)
plt.scatter(output[:, 0], output[:, 1], c='lime', s=15)
plt.title("Predicted Landmarks")
plt.axis('off')
plt.show()
