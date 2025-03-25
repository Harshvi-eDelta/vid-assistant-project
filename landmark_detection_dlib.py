import torchfile
import cv2
import numpy as np
import os

t7_folder = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset 2/t7"  
image_folder = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset 2/original_jpg" 
output_folder = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset 2/landmark_images"

os.makedirs(output_folder, exist_ok=True)

for t7_file in os.listdir(t7_folder):
    if t7_file.endswith(".t7"):
        t7_path = os.path.join(t7_folder, t7_file)
        image_name = os.path.splitext(t7_file)[0] + ".jpg"
        image_path = os.path.join(image_folder, image_name)

        if not os.path.exists(image_path):
            print(f"Warning: Image {image_name} not found. Skipping...")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image {image_name}.")
            continue

        # Load landmarks coordinates from .t7 file
        landmarks = torchfile.load(t7_path)

        if not isinstance(landmarks, np.ndarray) or landmarks.shape[1] != 2:
            print(f"Error: Invalid landmarks in {t7_file}. Skipping...")
            continue

        # Check if landmarks are out of range
        if np.max(landmarks) > 224:
            print(f"Warning: Scaling landmarks in {t7_file} (original max={np.max(landmarks)})")
            landmarks[:, 0] = np.clip(landmarks[:, 0], 0, 224)
            landmarks[:, 1] = np.clip(landmarks[:, 1], 0, 224)

        for (x, y) in landmarks:
            cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)

        # Save the image with landmarks
        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, image)
        print(f"Saved: {output_path}")

print("All landmark images have been saved.")
