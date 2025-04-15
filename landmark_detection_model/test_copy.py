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
image_path = "/Users/edelta076/Desktop/Project_VID_Assistant/face_images/fimg12.jpg"
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

# trying new 
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

# Load test image
image_path = "/Users/edelta076/Desktop/Project_VID_Assistant/face_images/fimg4.jpg"
original_img = cv2.imread(image_path)

if original_img is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

# Convert to RGB
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

# Resize for display (not for inference)
resized_img = cv2.resize(original_img, (256, 256))

# Convert to PIL for transform
pil_img = Image.fromarray(original_img)

# Apply transform
transform = get_transforms()
input_tensor = transform(pil_img).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(input_tensor).cpu().numpy().reshape(-1, 2)

# # Denormalize landmarks using display size (256x256)
# output[:, 0] *= 256
# output[:, 1] *= 256

original_width = original_img.shape[1]
original_height = original_img.shape[0]

output[:, 0] *= original_width
output[:, 1] *= original_height


# Draw landmarks on resized image
for (x, y) in output:
    cv2.circle(resized_img, (int(x), int(y)), 2, (0, 255, 0), -1)

# Show result
plt.figure(figsize=(4, 4))
plt.imshow(resized_img)
plt.title("Predicted Landmarks")
plt.axis("off")
plt.show()





