import os
import torchfile
import csv
import numpy as np

t7_folder = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/t7"
output_csv = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/landmarks.csv"

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["image_name"] + [f"x{i+1}" for i in range(68)] + [f"y{i+1}" for i in range(68)]
    writer.writerow(header)

    for t7_file in sorted(os.listdir(t7_folder)):  
        if t7_file.endswith(".t7"):
            t7_path = os.path.join(t7_folder, t7_file)

            landmark_data = torchfile.load(t7_path)

            if not isinstance(landmark_data, np.ndarray) or landmark_data.shape != (68, 2):
                print(f"Skipping {t7_file}, invalid landmark format")
                continue

            x_coords = landmark_data[:, 0].tolist()  
            y_coords = landmark_data[:, 1].tolist()  
            image_name = os.path.splitext(t7_file)[0] + ".jpg"

            writer.writerow([image_name] + x_coords + y_coords)

print(f"CSV file saved at: {output_csv}")



'''import pandas as pd
csv_path = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/landmarks.csv"
df = pd.read_csv(csv_path)

print(df.head())
print("Data type of first column:", df.iloc[:, 0].dtype)'''

