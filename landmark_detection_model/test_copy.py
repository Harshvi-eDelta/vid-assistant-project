import torch
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

    # Convert OpenCV image to PIL Image
    image_pil = Image.fromarray(image)  
    image_resized = transform(image_pil).unsqueeze(0)  # Apply transforms
    #image_resized = image_resized.to(torch.float32)  # Ensure correct dtype

    with torch.no_grad():
        output = model(image_resized)  # Get landmark predictions
    landmarks = output.cpu().numpy().reshape(-1, 2)

    landmarks[:, 0] *= 224  # Convert back to 224x224 coordinates
    landmarks[:, 1] *= 224  

    landmarks[:, 0] = (landmarks[:, 0] / 224) * w  # Scale back to original width
    landmarks[:, 1] = (landmarks[:, 1] / 224) * h  # Scale back to original height

    # Draw landmarks on image
    for (x, y) in landmarks:
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h:  # Ensure landmarks are inside image
            cv2.circle(orig_image, (x, y), 2, (0, 255, 0), -1)  

    #  Debugging: Print landmark coordinates
    print("Predicted landmarks:", landmarks)

    #  Display the image with landmarks
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

image_path = "/Users/edelta076/Desktop/Project_VID_Assistant/face_images/fimg3.png"
predict_landmarks(image_path)


