import torch
import cv2
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import HRNet_LandmarkDetector

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HRNet_LandmarkDetector().to(device)
model.load_state_dict(torch.load("hrnet_landmarks.pth", map_location=device))
model.eval()

# Load Image
def load_image(image_path, img_size=256):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5  # Normalize to [-1, 1] if used during training
    img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)
    return img

# Convert Heatmaps to Coordinates
def heatmap_to_landmarks(heatmaps, image_size=256, heatmap_size=64):
    """
    Converts heatmaps (B, 68, 64, 64) to coordinates in the original image scale.
    """
    coords = []
    heatmaps = heatmaps.squeeze(0).detach().cpu().numpy()  # Shape: [68, 64, 64]

    for heatmap in heatmaps:
        # Find max location in heatmap
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

        # Scale coords to original image size
        x = x * (image_size / heatmap_size)
        y = y * (image_size / heatmap_size)

        coords.append((int(x), int(y)))
    return np.array(coords)

# Run Inference
image_path = "/Users/edelta076/Desktop/Project_VID_Assistant/face_images/fimg9.jpg"
image = load_image(image_path)
output_heatmaps = model(image)  # Output: [1, 68, 64, 64]
landmarks = heatmap_to_landmarks(output_heatmaps)

# Plot Image & Landmarks
img = cv2.imread(image_path)
img = cv2.resize(img, (256, 256))

for (x, y) in landmarks:
    cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
    #cv2.putText(img, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
