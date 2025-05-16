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
'''import torch
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
model.load_state_dict(torch.load("best_model_3.pth", map_location=device))
model.to(device)
model.eval()

# Load test image
image_path = "/Users/edelta076/Desktop/Project_VID_Assistant/face_images/fimg15.jpg"    # 1,14,16,_1,13,16,18,23,24,27,28,29,04,026,046,060,088,0133,0143,0520
original_img = cv2.imread(image_path)
                                                                                   # 1,14,16,20,3,7,16,23,24,25,0871,04,026,051,088,0133,0143,0168,0520,0801
if original_img is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")              # 1,4,14,16,20,1,3,4,5,7,11,13,15,16,17,18,20,24,25,27,28,29,026,041,042,046,050,133,168
                                                                                    # 520,801
# Convert to RGB
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

# Resize for display (not for inference)
resized_img = cv2.resize(original_img, (256, 256))          # 1,16,20,1,3,4,5,13
print(resized_img.shape)

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
    _,_,_,_, output = model(input_tensor)
    print(type(output))              # Shape: (1, 68, 64, 64)

    # If output is a tuple, unpack the actual tensor
    if isinstance(output, tuple):
        output = output[0]

    output = output.squeeze(0).cpu().numpy()  # Shape: (68, 64, 64)

landmarks = heatmaps_to_landmarks_argmax(output)

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

print("Landmarks shape:", landmarks.shape)
print("First 5 landmarks:", landmarks[:5])

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

# Show result
plt.figure(figsize=(4, 4))
plt.imshow(resized_img)
plt.title("Predicted Landmarks")
plt.axis("off")
plt.savefig("abc")
plt.show()'''

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from landmark_cnn_copy import LandmarkCNN
from data_preprocessing_copy import get_transforms
import scipy.ndimage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = LandmarkCNN()
model.load_state_dict(torch.load("best_model_4.pth", map_location=device))
model.to(device)
model.eval()

# Load test image
image_path = "/Users/edelta076/Desktop/Project_VID_Assistant/face_images/fimg1.jpg"     # 1,14,16,20,_1,3,13,24,27,29,30,026,042,046,0133,0520,0801
original_img = cv2.imread(image_path)
if original_img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

# Convert to PIL and apply transform
pil_img = Image.fromarray(original_img)
transform = get_transforms()
input_tensor = transform(pil_img).unsqueeze(0).to(device)

# Decode with Gaussian-smoothed argmax
def heatmaps_to_landmarks_argmax(heatmaps):
    landmarks = []
    for i in range(heatmaps.shape[0]):
        smoothed = scipy.ndimage.gaussian_filter(heatmaps[i], sigma=1)
        y, x = np.unravel_index(np.argmax(smoothed), smoothed.shape)
        landmarks.append([x, y])
    return np.array(landmarks)

# Run inference
with torch.no_grad():
    _, _, _, _, output = model(input_tensor)
    if isinstance(output, tuple):
        output = output[0]
    output = output.squeeze(0).cpu().numpy()  # (68, 64, 64)

landmarks = heatmaps_to_landmarks_argmax(output)
landmarks *= 2  # Scale from 64x64 → 256x256        # *= 4 causing and error !
print(landmarks)

# Resize original image for visualization
resized_img = cv2.resize(original_img, (256, 256))

# Draw landmarks
for i, (x, y) in enumerate(landmarks.astype(int)):
    if 0 <= x < 256 and 0 <= y < 256:
        cv2.circle(resized_img, (x, y), 2, (0, 255, 0), -1)
    else:
        print(f"Skipped out-of-bounds landmark {i}: ({x}, {y})")

# Display
plt.figure(figsize=(4, 4))
plt.imshow(resized_img)
plt.title("Predicted Landmarks")
plt.axis("off")
plt.savefig("abc.png")
plt.show()


