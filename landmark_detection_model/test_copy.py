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

model_path = "/Users/edelta076/Desktop/Project_VID_Assistant/best_landmark_model_2.pth" 
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

image_path = "/Users/edelta076/Desktop/Project_VID_Assistant/face_images/fimg12.jpg"
predict_landmarks(image_path)'''

# original
'''import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from landmark_cnn_copy import LandmarkCNN
from data_preprocessing_copy import get_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = LandmarkCNN()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Load custom image
image_path = "/Users/edelta076/Desktop/Project_VID_Assistant/face_images/fimg2.jpg"
original_img = cv2.imread(image_path)

if original_img is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

# Convert BGR (OpenCV) to RGB
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

# Resize for visualization later
resized_img = cv2.resize(original_img, (256, 256))

# Convert to PIL for transform
pil_img = Image.fromarray(original_img)

# Apply transform
transform = get_transforms()
input_tensor = transform(pil_img).unsqueeze(0).to(device)

# Predict landmarks
with torch.no_grad():
    output = model(input_tensor).cpu().numpy().reshape(-1, 2)

# Denormalize using the final display image size (256x256)
output[:, 0] *= 256  # x
output[:, 1] *= 256  # y

# Draw landmarks
for (x, y) in output:
    cv2.circle(resized_img, (int(x), int(y)), 2, (0, 255, 0), -1)

# Show image with landmarks
plt.figure(figsize=(4,4))
plt.imshow(resized_img)
plt.title("Predicted Landmarks")
plt.axis("off")
plt.show()'''

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from landmark_cnn_copy import LandmarkCNN
from data_preprocessing_copy import get_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = LandmarkCNN()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Load image
image_path = "/Users/edelta076/Desktop/Project_VID_Assistant/face_images/4.jpg"
original_img = cv2.imread(image_path)
if original_img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# Convert to RGB
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

# Resize for inference
resized_img = cv2.resize(original_img, (256, 256))

# Prepare for model input
pil_img = Image.fromarray(original_img)
transform = get_transforms()
input_tensor = transform(pil_img).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    output = model(input_tensor).cpu().numpy().reshape(-1, 2)

# Denormalize
output[:, 0] *= 256
output[:, 1] *= 256

shift_x = 4   # Tune this: try -5 to +5
shift_y = 2   # Tune this: try -5 to +5
output[:, 0] += shift_x
output[:, 1] += shift_y

# Draw
for (x, y) in output:
    cv2.circle(resized_img, (int(x), int(y)), 2, (0, 255, 0), -1)

plt.imshow(resized_img)
plt.title("Predicted Landmarks")
plt.axis("off")
plt.show()


'''import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from landmark_cnn_copy import DeepLandmarkCNN
from torchvision import transforms
from PIL import Image

# Paths
model_path = "/Users/edelta076/Desktop/Project_VID_Assistant/best_deep_landmark_model.pth"
image_path = "/Users/edelta076/Desktop/Project_VID_Assistant/face_images/fimg11.jpg"

# Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load model
model = DeepLandmarkCNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Function to predict and visualize landmarks
def predict_landmarks_only(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    orig_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = orig_image.shape

    # Convert to PIL and transform
    image_pil = Image.fromarray(image)
    image_resized = transform(image_pil).unsqueeze(0)

    with torch.no_grad():
        output = model(image_resized)
        print("Raw model output (first 5 landmarks):", output[0][:5])

    # Reshape and scale landmarks to original image
    landmarks = output.cpu().numpy().reshape(-1, 2)
    landmarks[:, 0] *= w
    landmarks[:, 1] *= h
    landmarks = np.round(landmarks).astype(int)

    # Draw predicted landmarks (green)
    for (x, y) in landmarks:
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(orig_image, (x, y), 1, (0, 255, 0), -1)

    # Show result
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Predicted Landmarks (Green Only)")
    plt.show()

# Run prediction
predict_landmarks_only(image_path)'''




