import scipy.io
import numpy as np
import open3d as o3d

# Load the .mat file
mat_file = scipy.io.loadmat("/Users/edelta076/Desktop/Project_VID_Assistant/300W-3D-Face/HELEN/14403172_1.mat")

# Extract 'Fitted_Face' data
fitted_face = mat_file['Fitted_Face']  # Shape: (3, 53215)

# Convert to (N, 3) format
points_3d = fitted_face.T  # Shape: (53215, 3)

# Create Open3D Point Cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d)

# Save point cloud
o3d.io.write_point_cloud("point_cloud.ply", pcd)

# Visualize point cloud
o3d.visualization.draw_geometries([pcd], window_name="3D Point Cloud")

# Remove outliers
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# Downsample (optional)
pcd = pcd.voxel_down_sample(voxel_size=0.002)

# Estimate normals (important for meshing)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(k=10)

# Save the processed point cloud
o3d.io.write_point_cloud("cleaned_point_cloud.ply", pcd)

# Try Poisson Reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6)

# Save the mesh
o3d.io.write_triangle_mesh("output_mesh.obj", mesh)

# Visualize the mesh
o3d.visualization.draw_geometries([mesh], window_name="Generated 3D Mesh")

