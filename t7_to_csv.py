import os
import torchfile
import csv
import cv2
import numpy as np

# Paths
t7_folder = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/t7"
image_folder = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/original_jpg"  # Update this path
output_csv = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/landmarks_normalized.csv"

# Open CSV file for writing
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["image_name"] + [f"x{i+1}" for i in range(68)] + [f"y{i+1}" for i in range(68)]
    writer.writerow(header)

    # Process each .t7 file
    for t7_file in sorted(os.listdir(t7_folder)):  
        if t7_file.endswith(".t7"):
            t7_path = os.path.join(t7_folder, t7_file)
            landmark_data = torchfile.load(t7_path)

            # Ensure correct shape
            if not isinstance(landmark_data, np.ndarray) or landmark_data.shape != (68, 2):
                print(f"Skipping {t7_file}, invalid landmark format")
                continue

            image_name = os.path.splitext(t7_file)[0] + ".jpg"
            image_path = os.path.join(image_folder, image_name)

            # 🔹 Get actual image size
            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠️ Warning: Image {image_name} not found. Skipping...")
                continue

            h, w, _ = image.shape  # Get actual height and width

            # 🔹 Normalize landmarks using actual image size
            x_coords = landmark_data[:, 0] / w  # Normalize x-coordinates
            y_coords = landmark_data[:, 1] / h  # Normalize y-coordinates

            # 🔹 Ensure values are clipped to [0,1] (to prevent outliers)
            x_coords = np.clip(x_coords, 0, 1)
            y_coords = np.clip(y_coords, 0, 1)

            # Write to CSV
            writer.writerow([image_name] + x_coords.tolist() + y_coords.tolist())

print(f"✅ Normalized CSV file saved at: {output_csv}")
