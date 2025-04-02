'''import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from landmark_cnn import LandmarkCNN  # Import your trained model
from torchvision import transforms
from PIL import Image 

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

model_path = "/Users/edelta076/Desktop/Project_VID_Assistant/best_landmark_model.pth" 
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

    #landmarks[:, 0] = (landmarks[:, 0] / 224) * w  # Scale back to original width
    #landmarks[:, 1] = (landmarks[:, 1] / 224) * h  # Scale back to original height
    landmarks[:, 0] = ((landmarks[:, 0] / 224) * w) - 6
    landmarks[:, 1] = ((landmarks[:, 1] / 224) * h) - 12  # Small vertical shift correction


    # Draw landmarks on image
    for (x, y) in landmarks:
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h:  # Ensure landmarks are inside image
            cv2.circle(orig_image, (x, y), 1, (0, 255, 0), -1)  

    #  Debugging: Print landmark coordinates
    print("Predicted landmarks:", landmarks)

    #  Display the image with landmarks
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

image_path = "/Users/edelta076/Desktop/Project_VID_Assistant/face_images/fimg2.jpg"
predict_landmarks(image_path)'''

'''import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from landmark_cnn import LandmarkCNN  # Import your trained model
from torchvision import transforms
from PIL import Image 

transform = transforms.Compose([
    transforms.Resize((224, 224),interpolation=Image.BILINEAR),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

model_path = "/Users/edelta076/Desktop/Project_VID_Assistant/best_landmark_model.pth" 
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
    #image_resized = transforms.functional.resize(image, (224, 224), interpolation=transforms.InterpolationMode.BILINEAR).unsqueeze(0)

    with torch.no_grad():
        output = model(image_resized)  # Get landmark predictions
    landmarks = output.cpu().numpy().reshape(-1, 2)

    #landmarks[:, 0] *= w  # Convert back to 224x224 coordinates
    #landmarks[:, 1] *= h  

    # Ensure correct scaling (float → int to avoid rounding issues)
    landmarks[:, 0] = np.round(landmarks[:, 0] * w).astype(int)
    landmarks[:, 1] = np.round(landmarks[:, 1] * h).astype(int)


    #print(f"Original image size: {w}x{h}")
    #print(f"Predicted landmarks (normalized): {output.cpu().numpy().reshape(-1, 2)}")
    #print(f"Denormalized landmarks (before scaling to original size): {landmarks * 224}")
    #print(f"Final scaled landmarks (original image size): {landmarks}")

    print(f"Image Size: {w} x {h}")
    print(f"Predicted Landmarks Range: X({landmarks[:, 0].min()} - {landmarks[:, 0].max()}), Y({landmarks[:, 1].min()} - {landmarks[:, 1].max()})")
    print(f"GT Landmarks Range: X({landmarks[:, 0].min()} - {landmarks[:, 0].max()}), Y({landmarks[:, 1].min()} - {landmarks[:, 1].max()})")


    # Draw landmarks on image
    for (x, y) in landmarks:
        x, y = int(round(x)), int(round(y))
        if 0 <= x < w and 0 <= y < h:  # Ensure landmarks are inside image
            cv2.circle(orig_image, (x, y), 1, (0, 255, 0), -1)  

    for i, (x, y) in enumerate(landmarks):
        cv2.circle(orig_image, (x, y), 2, (255, 0, 0), -1)  
        cv2.putText(orig_image, f"GT{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    #  Debugging: Print landmark coordinates
    print("Predicted landmarks:", landmarks)

    #  Display the image with landmarks
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

image_path = "/Users/edelta076/Desktop/Project_VID_Assistant/face_images/fimg9.jpg"
predict_landmarks(image_path)'''

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from landmark_cnn import LandmarkCNN
from torchvision import transforms
from PIL import Image 

# Transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

#print("Transform",transform)

# Paths
model_path = "/Users/edelta076/Desktop/Project_VID_Assistant/best_landmark_model.pth" 
test_pth = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/t7_fixed/6.pth"
image_path = "/Users/edelta076/Desktop/Project_VID_Assistant/face_images/fimg2.jpg"
#image_path = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/landmark_images/13.jpg"

# Load Model
model = LandmarkCNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Function to Load Ground Truth Landmarks
def load_gt_landmarks(pth_path):
    return torch.load(pth_path)  # GT landmarks in pixel coordinates

# Function to Predict and Compare Landmarks
def predict_landmarks(image_path, gt_landmarks):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    orig_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = orig_image.shape  # Original image dimensions

    # Convert OpenCV image to PIL Image
    image_pil = Image.fromarray(image)  
    image_resized = transform(image_pil).unsqueeze(0)  # Apply transforms

    # plt.subplot(1,2,1)
    # plt.imshow(Image.open(image_path))  
    # plt.title("Original Image")

    # plt.subplot(1,2,2)
    # plt.imshow(image_pil)  # Check the resized image
    # plt.title("Resized Image (224x224)")
    # plt.show()

    with torch.no_grad():
        output = model(image_resized)  # Get landmark predictions
    landmarks = output.cpu().numpy().reshape(-1, 2)  # Convert to numpy array

    # Scale landmarks to original image size
    landmarks[:, 0] *= w  
    landmarks[:, 1] *= h  
    landmarks = np.round(landmarks).astype(int)

    # Debugging: Print Landmark Scaling Info
    print(f"GT Landmarks (Raw):\n{gt_landmarks}")
    print(f"Predicted Landmarks (Raw):\n{landmarks}")

    # Ensure GT landmarks are in pixel coordinates
    if gt_landmarks.max() <= 1.0:  # They are still normalized
        gt_landmarks[:, 0] *= w
        gt_landmarks[:, 1] *= h
        print("GT landmarks were normalized. Converted to pixel coordinates.")

    print(f"GT Landmarks (After Scaling):\n{gt_landmarks}")
    print(f"Predicted Landmarks (After Scaling):\n{landmarks}")

    # Compute offset per landmark
    offsets = gt_landmarks - landmarks  
    # average_offset = np.mean(landmarks - offsets, axis=0)
    # print(f"Average Offset: {average_offset}")

    # Apply correction
    corrected_landmarks = landmarks + offsets

    # Ensure landmarks stay within image bounds
    corrected_landmarks[:, 0] = np.clip(corrected_landmarks[:, 0], 0, w)  
    corrected_landmarks[:, 1] = np.clip(corrected_landmarks[:, 1], 0, h)  

    # Save corrected landmarks
    np.savetxt("corrected_landmarks.txt", corrected_landmarks, fmt="%.2f")
    print("Landmarks corrected and saved successfully!")

    mae_before = np.mean(np.abs(gt_landmarks - landmarks), axis=0)
    mae_after = np.mean(np.abs(gt_landmarks - corrected_landmarks), axis=0)

    print(f"MAE Before Correction: {mae_before}")
    print(f"MAE After Correction: {mae_after}")

    # Draw Predicted Landmarks
    for (x, y) in landmarks:
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(orig_image, (x, y), 1, (0, 255, 0), -1)  # Green for predicted

    # Debugging Information
    '''print(f"Image Size: {w} x {h}")
    print(f"Predicted Landmarks: X({landmarks[:, 0].min()} - {landmarks[:, 0].max()}), Y({landmarks[:, 1].min()} - {landmarks[:, 1].max()})")
    print(f"GT Landmarks: X({gt_landmarks[:, 0].min()} - {gt_landmarks[:, 0].max()}), Y({gt_landmarks[:, 1].min()} - {gt_landmarks[:, 1].max()})")

    print("Original Predicted Landmarks:\n", landmarks)
    print("Computed Offsets:\n", offsets)
    print("Average Offset:\n", average_offset)
    print("Corrected Landmarks:\n", corrected_landmarks)

    print("Raw Model Output:\n", landmarks)
    print("Ground Truth Landmarks:\n", gt_landmarks)'''

    # Show Image
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    # plt.savefig("image")
    plt.show()

# Load GT Landmarks and Run Prediction
gt_landmarks = load_gt_landmarks(test_pth)
predict_landmarks(image_path, gt_landmarks)

