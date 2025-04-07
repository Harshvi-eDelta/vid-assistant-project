import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from landmark_cnn_w import get_model

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)
model.load_state_dict(torch.load("landmark_cnn_w.pth"))
model.eval()

# Define Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def predict_landmarks(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    image_resized = cv2.resize(image, (96, 96))

    # Convert to Tensor
    image_tensor = transform(image_resized).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        pred = model(image_tensor).cpu().numpy().reshape(-1, 2)
    
    h,w = 224,224
    # Denormalize landmarks properly
    pred[:, 0] = pred[:, 0] * w  # Convert back to original width
    pred[:, 1] = pred[:, 1] * h  # Convert back to original height

    # Show image with landmarks
    plt.imshow(image, cmap="gray")
    plt.scatter(pred[:, 0], pred[:, 1], c="green", s=8)
    plt.show()

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.scatter(pred[:, 0], pred[:, 1], c="green", s=8)
    plt.title("Landmarks on Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(image_resized, cmap="gray")
    plt.scatter(pred[:, 0] * (96 / w), pred[:, 1] * (96 / h), c="green", s=8)
    plt.title("Landmarks on Resized Image")

    plt.show()


# Test on a new image
predict_landmarks("/Users/edelta076/Desktop/Project_VID_Assistant/face_images/image00520.jpg")
