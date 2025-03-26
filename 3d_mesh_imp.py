import os
import numpy as np
import cv2
import torchfile
import trimesh
import pymeshfix
from scipy.spatial import Delaunay

# Input folders
landmark_folder = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/t7"
depth_folder = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/depth_maps"

# Output folder for fixed 3D meshes
mesh_output_folder = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/fixed_meshes"
'''os.makedirs(mesh_output_folder, exist_ok=True)

# Camera intrinsics (for 224x224 images)
fx, fy = 300, 300  
cx, cy = 112, 112  

# Process each landmark file
landmark_files = [f for f in os.listdir(landmark_folder) if f.endswith(".t7")]

for landmark_file in landmark_files:
    name = os.path.splitext(landmark_file)[0]  

    # Load 2D landmarks
    landmarks_2d = torchfile.load(os.path.join(landmark_folder, landmark_file))
    landmarks_2d = np.array(landmarks_2d)

    # Load depth map
    depth_map_path = os.path.join(depth_folder, name + ".jpg")
    if not os.path.exists(depth_map_path):
        print(f"Skipping {name}, depth map not found.")
        continue

    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth_map /= np.max(depth_map)  

    # Convert 2D landmarks to 3D
    landmarks_3d = []
    for x, y in landmarks_2d:
        if 0 <= int(y) < depth_map.shape[0] and 0 <= int(x) < depth_map.shape[1]:
            z = depth_map[int(y), int(x)]
            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            landmarks_3d.append((X, Y, z))

    landmarks_3d = np.array(landmarks_3d)
    if len(landmarks_3d) < 4:
        print(f"Skipping {name}: Not enough valid 3D points for meshing.")
        continue

    # Create a 3D mesh using Delaunay triangulation
    tri = Delaunay(landmarks_3d[:, :2])
    faces = tri.simplices

    # Apply PyMeshFix to improve the mesh
    meshfix = pymeshfix.MeshFix(landmarks_3d, faces)
    meshfix.repair(verbose=True)
    fixed_vertices, fixed_faces = meshfix.v, meshfix.f

    # Save the fixed mesh
    fixed_mesh = trimesh.Trimesh(vertices=fixed_vertices, faces=fixed_faces)
    mesh_output_path = os.path.join(mesh_output_folder, name + "_fixed.obj")
    fixed_mesh.export(mesh_output_path)

    print(f"Fixed mesh saved at: {mesh_output_path}")

print("All images processed! Fixed 3D meshes stored in:", mesh_output_folder)'''

mesh = trimesh.load_mesh("/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/fixed_meshes/485_fixed.obj")
mesh.show()