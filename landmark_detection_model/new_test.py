import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from landmark_cnn import LandmarkCNN
from torchvision import transforms
from PIL import Image

# Load model
model_path = "/Users/edelta076/Desktop/Project_VID_Assistant/best_landmark_model.pth"
model = LandmarkCNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load landmarks from .pth (ground truth)
def load_gt_landmarks(pth_path):
    gt_landmarks = torch.load(pth_path)  # Convert to numpy array
    return gt_landmarks  # Already in pixel coordinates

# Predict & visualize landmarks
def test_landmarks(image_path, pth_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    orig_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = orig_image.shape  # Get original size

    # Transform and predict
    image_pil = Image.fromarray(image)
    image_resized = transform(image_pil).unsqueeze(0)

    with torch.no_grad():
        output = model(image_resized)
    pred_landmarks = output.cpu().numpy().reshape(-1, 2)

    # Convert normalized landmarks to pixel coordinates
    pred_landmarks[:, 0] = pred_landmarks[:, 0] * w  # Scale X
    pred_landmarks[:, 1] = pred_landmarks[:, 1] * h  # Scale Y

    # Load ground truth landmarks
    gt_landmarks = load_gt_landmarks(pth_path)

    # 🔹 Compare order of predicted vs. ground truth landmarks
    print("\n🔍 Checking Landmark Order:")
    for i in range(len(gt_landmarks)):
        print(f"GT {i}: {gt_landmarks[i]}  |  Pred {i}: {pred_landmarks[i]}")

    # Draw predicted landmarks (Green)
    for i, (x, y) in enumerate(pred_landmarks):
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(orig_image, (x, y), 2, (0, 255, 0), -1)  # Green dots
            cv2.putText(orig_image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Draw ground truth landmarks (Red)
    for i, (x, y) in enumerate(gt_landmarks):
        x, y = int(x), int(y)
        cv2.circle(orig_image, (x, y), 2, (255, 0, 0), -1)  # Red dots
        cv2.putText(orig_image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # Show image
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.savefig("123")
    plt.show()

# Run test
test_image = "/Users/edelta076/Desktop/Project_VID_Assistant/face_images/fimg6.jpg"
test_pth = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/t7_fixed/6.pth"
 # Change this
test_landmarks(test_image, test_pth)
