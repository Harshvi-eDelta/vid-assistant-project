'''import os
import numpy as np
import cv2
import torchfile
import open3d as o3d
from scipy.spatial import Delaunay
import trimesh

# Define Input and Output Folders
landmark_folder = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/t7"
depth_folder = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/depth_maps_1"
mesh_output_folder = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/generated_meshes1"

# Ensure Output Folder Exists
os.makedirs(mesh_output_folder, exist_ok=True)

# Get All Landmark Files
landmark_files = [f for f in os.listdir(landmark_folder) if f.endswith(".t7")]
if not landmark_files:
    print("No landmark (.t7) files found! Check your folder path.")
    exit()

# Camera Intrinsics for 224x224 Images
fx, fy = 300, 300  # Adjust based on your camera
cx, cy = 112, 112  # Optical center (center of 224x224 image)

# Process Each Landmark File
for landmark_file in landmark_files:
    name = os.path.splitext(landmark_file)[0]  # Remove .t7 extension

    # Load 2D Landmarks
    landmarks_2d = torchfile.load(os.path.join(landmark_folder, landmark_file))
    landmarks_2d = np.array(landmarks_2d)
    
    if landmarks_2d.shape[1] != 2:
        print(f"Skipping {name}: Invalid landmark format.")
        continue

    # Load Depth Map
    depth_map_path = os.path.join(depth_folder, name + ".jpg")
    if not os.path.exists(depth_map_path):
        print(f"Skipping {name}: Depth map not found.")
        continue

    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if depth_map is None or depth_map.size == 0:
        print(f"Skipping {name}: Could not read depth map.")
        continue

    depth_map /= np.max(depth_map)  # Normalize depth (0 to 1)

    # Convert 2D Landmarks to 3D
    landmarks_3d = []
    for x, y in landmarks_2d:
        x, y = int(x), int(y)
        if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
            z = depth_map[y, x]  # Get depth value
            X = (x - cx) * z / fx  # Convert to 3D space
            Y = (y - cy) * z / fy
            landmarks_3d.append([X, Y, z])
        else:
            print(f"Skipping point ({x}, {y}) - Out of bounds")

    landmarks_3d = np.array(landmarks_3d)
    
    # Ensure we have enough unique points
    if landmarks_3d.shape[0] < 4:
        print(f"Skipping {name}: Not enough valid 3D points for meshing.")
        continue

    # Ensure points are not collinear or coplanar
    unique_points = np.unique(landmarks_3d, axis=0)
    if unique_points.shape[0] < 4:
        print(f"Skipping {name}: 3D points are collinear or too close.")
        continue

    # Create 3D Mesh Using Delaunay Triangulation
    try:
        tri = Delaunay(unique_points[:, :2])  # Triangulate only XY (2D projection)
        faces = tri.simplices  # Indices of triangles

        # Create a Mesh using Open3D
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(unique_points)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        # Compute Normals
        mesh.compute_vertex_normals()

        # Compute Basic UV Mapping (Planar Projection)
        uv_coords = np.array([[v[0], v[1]] for v in unique_points])  # Use X, Y as UV
        uv_coords = (uv_coords - uv_coords.min(axis=0)) / (uv_coords.max(axis=0) - uv_coords.min(axis=0))  # Normalize UV to [0,1]

        # Save Mesh with UV Mapping
        mesh_output_path = os.path.join(mesh_output_folder, name + ".obj")
        o3d.io.write_triangle_mesh(mesh_output_path, mesh)

        print(f"Mesh with UV mapping saved: {mesh_output_path}")

    except Exception as e:
        print(f"Error processing {name}: {e}")

print(f"\nAll meshes stored in: {mesh_output_folder}")

#mesh = o3d.io.read_triangle_mesh("/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/generated_meshes/1.obj")
#print(mesh)

#mesh = trimesh.load("/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/generated_meshes/1.obj")
#mesh.show()


# Count .obj files in the mesh output folder
#mesh_files = [f for f in os.listdir(mesh_output_folder) if f.endswith(".obj")]
#print(f"\nTotal 3D mesh files generated: {len(mesh_files)}")    (6374)'''

import os
import numpy as np
import cv2
import torchfile
import open3d as o3d

# Define Input and Output Folders
landmark_folder = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/t7"
depth_folder = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/original_jpg"
mesh_output_folder = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/generated_meshes_new"

# Ensure Output Folder Exists
os.makedirs(mesh_output_folder, exist_ok=True)

# Get All Landmark Files
landmark_files = [f for f in os.listdir(landmark_folder) if f.endswith(".t7")]
if not landmark_files:
    print("No landmark (.t7) files found! Check your folder path.")
    exit()

# Camera Intrinsics for 224x224 Images
fx, fy = 300, 300  # Adjust based on your camera
cx, cy = 112, 112  # Optical center (center of 224x224 image)

# Process Each Landmark File
for landmark_file in landmark_files:
    name = os.path.splitext(landmark_file)[0]  # Remove .t7 extension

    # Load 2D Landmarks
    landmarks_2d = torchfile.load(os.path.join(landmark_folder, landmark_file))
    landmarks_2d = np.array(landmarks_2d)

    if landmarks_2d.shape[1] != 2:
        print(f"Skipping {name}: Invalid landmark format.")
        continue

    # Load Depth Map
    depth_map_path = os.path.join(depth_folder, name + ".jpg")
    if not os.path.exists(depth_map_path):
        print(f"Skipping {name}: Depth map not found.")
        continue

    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    if depth_map is None or depth_map.size == 0:
        print(f"Skipping {name}: Could not read depth map.")
        continue

    #depth_map /= np.max(depth_map)  # Normalize depth (0 to 1)
    depth_map = depth_map / 255.0 * 1.5  # Scale depth properly


    # Convert entire depth map to 3D point cloud
    height, width = depth_map.shape
    points_3d = []

    for y in range(height):
        for x in range(width):
            z = depth_map[y, x]
            if 0.01 < z < 1.5:  # Ignore invalid depth values
                Z = z * 2.0  # Scale depth (adjust this factor based on real data)
                X = (x - cx) * Z / fx
                Y = (y - cy) * Z / fy
                points_3d.append([X, Y, z])

    points_3d = np.array(points_3d)

    # Ensure we have enough unique points
    if points_3d.shape[0] < 10:
        print(f"Skipping {name}: Not enough valid 3D points for meshing.")
        continue

    # Ensure points are not collinear or coplanar
    unique_points = np.unique(points_3d, axis=0)  # Remove duplicates
    if unique_points.shape[0] < 4:
        print(f"Skipping {name}: 3D points are collinear or too close.")
        continue

    # Convert to Open3D Point Cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(unique_points)

    # Remove outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)

    # Downsample (optional)
    pcd = pcd.voxel_down_sample(voxel_size=0.05)

    # Estimate better normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=10)

    # Visualize point cloud (debugging)
    #o3d.visualization.draw_geometries([pcd], window_name="Check Point Cloud")

    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        if len(mesh.triangles) == 0:
            raise ValueError("Poisson mesh is empty")
    except Exception as e:
        print(f"Poisson failed for {name}, switching to Ball-Pivoting")
        radii = [0.005, 0.01, 0.02, 0.04]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )

    # Save mesh
    mesh_output_path = os.path.join(mesh_output_folder, name + ".obj")
    o3d.io.write_triangle_mesh(mesh_output_path, mesh)
    print(f"Mesh saved: {mesh_output_path}")

print(f"\nAll meshes stored in: {mesh_output_folder}")



