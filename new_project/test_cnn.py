import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from landmark_cnn import LandmarkCNN  # Import your model definition

#  Load Trained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LandmarkCNN().to(device)
model.load_state_dict(torch.load("best_landmark_model_2.pth", map_location=device))
model.eval()  # Set to evaluation mode

#  Define Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def visualize_landmarks(image, landmarks, orig_w, orig_h):
    """Plot predicted landmarks on the image."""
    image = image.permute(1, 2, 0).numpy()  # Convert (3, 224, 224) → (224, 224, 3)
    image = (image * 0.5) + 0.5  # Denormalize image
    image = np.clip(image, 0, 1)

    # Reshape landmarks from (42,) → (21,2)
    landmarks = landmarks.reshape(-1, 2)

    # Convert normalized landmarks to pixel coordinates (original image size)
    landmarks[:, 0] *= orig_w  
    landmarks[:, 1] *= orig_h  

    print("Predicted Landmarks (Scaled to Original Image):", landmarks)

    # Plot image & landmarks
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='green', s=8)
    plt.show()

# Load Test Image
test_image_path = "/Users/edelta076/Desktop/Project_VID_Assistant/face_images/fimg12.jpg"  # Change to your test image path
image = Image.open(test_image_path).convert("RGB")
orig_w, orig_h = image.size

#  Preprocess Image
image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension  

#  Predict Landmarks
with torch.no_grad():
    pred_landmarks = model(image_tensor)
    pred_landmarks = pred_landmarks.cpu().numpy().flatten()  # Convert to numpy

# Visualize Result
visualize_landmarks(image_tensor.squeeze(0), pred_landmarks,orig_w,orig_h)
