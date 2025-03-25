import os
import shutil
import torchfile
import cv2
import numpy as np

# Define dataset paths
dataset_folder = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset"
jpg_folder = os.path.join(dataset_folder, "original_jpg")
t7_folder = os.path.join(dataset_folder, "t7")
output_folder = os.path.join(dataset_folder, "landmark_images")

# Create necessary directories
os.makedirs(jpg_folder, exist_ok=True)
os.makedirs(t7_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Step 1: Separate original images and .t7 files
all_files = os.listdir(dataset_folder)
for file in all_files:
    file_path = os.path.join(dataset_folder, file)
    if file.endswith('.jpg'):
        shutil.move(file_path, os.path.join(jpg_folder, file))
    elif file.endswith('.t7'):
        shutil.move(file_path, os.path.join(t7_folder, file))

print("Step 1: Images and .t7 files separated.")

# Step 2: Perform landmark detection and resizing
for t7_file in os.listdir(t7_folder):
    if t7_file.endswith(".t7"):
        t7_path = os.path.join(t7_folder, t7_file)
        image_name = os.path.splitext(t7_file)[0] + ".jpg"
        image_path = os.path.join(jpg_folder, image_name)

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Image {image_name} not found. Skipping...")
            continue

        # Load landmarks from .t7 file
        landmark_data = torchfile.load(t7_path)
        landmarks = np.array(landmark_data)  # Convert to NumPy array

        # Draw landmarks on image
        for (x, y) in landmarks:
            cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)  # Green dots

        # Resize image and landmarks to 224x224
        original_height, original_width = image.shape[:2]
        image_resized = cv2.resize(image, (224, 224))
        
        # Scale landmarks to match resized image
        landmarks[:, 0] = (landmarks[:, 0] / original_width) * 224  # X-coordinates
        landmarks[:, 1] = (landmarks[:, 1] / original_height) * 224  # Y-coordinates

        # Save the image with landmarks
        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, image_resized)
        print(f"Saved: {output_path}")

print("Step 2: Landmark detection and resizing completed.")
