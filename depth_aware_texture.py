import os
import shutil

# Base folder where original datasets are stored
base_folder = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset"

# Define paths for input folders
input_folders = {
    "images": os.path.join(base_folder, "original_jpg"),        
    "depth_maps": os.path.join(base_folder, "depth_maps_1"),      
    "meshes": os.path.join(base_folder, "generated_meshes1"),    
    "uv_meshes": os.path.join(base_folder, "uv_generated_meshes"), 
    "landmarks": os.path.join(base_folder, "t7"),  
                
}

# Define NEW output folder for structured data
structured_folder = os.path.join(base_folder, "structured_dataset")  # New folder
os.makedirs(structured_folder, exist_ok=True)

# Step 1: Collect all filenames (without extensions)
file_sets = {key: set() for key in input_folders}

for key, folder in input_folders.items():
    for file in os.listdir(folder):
        if key == "uv_meshes":
            name = file.replace("processed_", "").rsplit(".", 1)[0]  # Normalize UV mesh filenames
        else:
            name = file.rsplit(".", 1)[0]  # Remove extension
        file_sets[key].add(name)

# Step 2: Find common names (files that exist in ALL categories)
common_files = set.intersection(*file_sets.values())

# Step 3: Copy files into new structured dataset
for person in common_files:
    person_folder = os.path.join(structured_folder, person)
    os.makedirs(person_folder, exist_ok=True)  # Create a folder for each person

    # Copy only files that belong to the complete set
    for key, folder in input_folders.items():
        if key == "uv_meshes":
            possible_files = [
                os.path.join(folder, f"processed_{person}.obj"),
                os.path.join(folder, f"processed_{person}.obj.mtl")
            ]
        elif key == "depth_maps":
            possible_files = [os.path.join(folder, f"{person}.jpg")]
            new_filenames = [os.path.join(person_folder, f"{person}_depth.jpg")]  # Rename depth map
        else:
            possible_files = [
                os.path.join(folder, f"{person}.jpg"),
                os.path.join(folder, f"{person}.png"),
                os.path.join(folder, f"{person}.obj"),
                os.path.join(folder, f"{person}.t7")
            ]
            new_filenames = None  # No renaming needed

        for i, possible_file in enumerate(possible_files):
            if os.path.exists(possible_file):
                destination = new_filenames[i] if new_filenames else os.path.join(person_folder, os.path.basename(possible_file))
                shutil.copy(possible_file, destination)  # Use copy instead of move

print(f"{len(common_files)} Complete Sets Copied to {structured_folder}! No original files were moved.")
