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
import collections
import scipy.ndimage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = LandmarkCNN()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Load test image
image_path = "/Users/edelta076/Desktop/Project_VID_Assistant/face_images/nfimg2.jpg"
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

# Decode heatmaps to (x, y) coordinates
def heatmaps_to_landmarks_argmax(heatmaps):
    landmarks = []
    for i in range(heatmaps.shape[0]):
        h = heatmaps[i]
        y, x = np.unravel_index(np.argmax(h), h.shape)
        landmarks.append([x, y])
    return np.array(landmarks)

with torch.no_grad():
    _,_,output = model(input_tensor)
    print(type(output))              # Shape: (1, 68, 64, 64)
    
    # If output is a tuple, unpack the actual tensor
    if isinstance(output, tuple):
        output = output[0]

    output = output.squeeze(0).cpu().numpy()  # Shape: (68, 64, 64)

landmarks = heatmaps_to_landmarks_argmax(output)

# # Convert heatmaps to coordinates
# landmarks = heatmaps_to_landmarks(output)
# print(f"Total landmarks: {len(landmarks)}")

# Convert to int for checking duplicates at pixel level
int_landmarks = np.round(landmarks).astype(int)

# Count each (x, y) pair
counter = collections.Counter(map(tuple, int_landmarks))
duplicates = [pt for pt, count in counter.items() if count > 1]

print(f"\nDetected {len(duplicates)} overlapping landmark positions:")
for pt in duplicates:
    print(f" - {pt}")

# Scale landmarks from heatmap size (64x64) → image size (256x256)
landmarks *= 4  # (256 / 64)

# original_width = original_img.shape[1]
# original_height = original_img.shape[0]

# output[:, 0] *= original_width
# output[:, 1] *= original_height
print("Landmarks shape:", landmarks.shape)
print("First 5 landmarks:", landmarks[:5])


# # # Draw landmarks on resized image
# for (x, y) in landmarks:
#     x = int(np.clip(x, 0, 255))
#     y = int(np.clip(y, 0, 255))
#     cv2.circle(resized_img, (int(x), int(y)), 2, (0, 255, 0), -1)

print("Landmarks shape:", landmarks.shape)
for i, (x, y) in enumerate(landmarks):
    print(f"Landmark {i}: ({x:.2f}, {y:.2f})")


# Draw all 68 landmarks on the image
for i, (x, y) in enumerate(landmarks.astype(int)):
    # Check if the point is within bounds before drawing
    if 0 <= x < resized_img.shape[1] and 0 <= y < resized_img.shape[0]:
        cv2.circle(resized_img, (x, y), 2, (0, 255, 0), -1)  # Green dot
        #cv2.putText(resized_img, str(i), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    else:
        print(f"Skipped landmark {i}: ({x}, {y}) out of bounds")


# for idx, (x, y) in enumerate(landmarks):
#     print(f"Landmark {idx}: ({x}, {y})")

# Show result
plt.figure(figsize=(3, 3))
plt.imshow(resized_img)
plt.title("Predicted Landmarks")
plt.axis("off")
plt.show()
