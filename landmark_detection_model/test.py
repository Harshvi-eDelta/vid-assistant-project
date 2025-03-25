'''import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from landmark_cnn import LandmarkCNN  # Import your trained model
from torchvision import transforms
from PIL import Image 


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # No need for ToPILImage()
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

model_path = "/Users/edelta076/Desktop/Project_VID_Assistant/landmark_model.pth" 
model = LandmarkCNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Function to detect and visualize landmarks
def predict_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    orig_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = orig_image.shape  # Original image dimensions

    # Convert OpenCV image (NumPy array) to PIL Image
    image_pil = Image.fromarray(image)  
    # Directly apply transform (without ToPILImage())
    image_resized = transform(image_pil).unsqueeze(0)  # Apply transforms

  

    with torch.no_grad():
        output = model(image_resized)  # Get landmark predictions
    landmarks = output.cpu().numpy().reshape(-1, 2)

    # Scale landmarks back to original image size
    landmarks[:, 0] *= w  
    landmarks[:, 1] *= h  

    # Draw landmarks on image
    for (x, y) in landmarks:
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h:  # Ensure landmarks are inside image
            cv2.circle(orig_image, (x, y), 2, (0, 255, 0), -1)  

    #  Debugging: Print landmark coordinates
    print("Predicted landmarks:", landmarks)

    #  Display the image with landmarks
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

image_path = "/Users/edelta076/Desktop/Project_VID_Assistant/face_images/fimg2.png"
predict_landmarks(image_path)'''


import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from landmark_cnn import LandmarkCNN  # Import your trained model

# 🔹 Load Trained Model
model_path = "/Users/edelta076/Desktop/Project_VID_Assistant/landmark_model.pth"
model = LandmarkCNN()  # Initialize model
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()  # Set model to evaluation mode

# 🔹 Define Transform
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert to PIL image first
    transforms.Resize((224, 224)),  # Resize first!
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def predict_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    orig_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = orig_image.shape  # Get original image size

    # 🔹 Convert image to tensor
    image_pil = Image.fromarray(image)  # Convert to PIL image
    image_tensor = transform(image_pil).unsqueeze(0)  # Apply transform

    # 🔹 Predict landmarks
    with torch.no_grad():
        output = model(image_tensor)

    landmarks = output.cpu().numpy().reshape(-1, 2)  # Reshape into (num_points, 2)

    # 🔹 Debugging: Print raw landmark values before scaling
    print("\n🔹 Raw model output (before scaling):")
    print(landmarks)

    # 🔄 Scale landmarks to original image size
    landmarks[:, 0] *= w  # Scale x-coordinates
    landmarks[:, 1] *= h  # Scale y-coordinates

    # 🔹 Debugging: Print scaled landmark values
    print("\n🔹 Scaled landmarks (after multiplying with image size):")
    print(landmarks)

    # 🔹 Draw landmarks on image
    for (x, y) in landmarks:
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h:  # Ensure landmarks are inside the image
            cv2.circle(orig_image, (x, y), 2, (0, 255, 0), -1)  # Draw green dots

    # 🔹 Show image with landmarks
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

# 🔹 Run on a sample image
image_path = "/Users/edelta076/Desktop/Project_VID_Assistant/face_images/fimg2.png"
predict_landmarks(image_path)

