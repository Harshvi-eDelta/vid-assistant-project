'''import open3d as o3d

# Load the mesh
mesh = o3d.io.read_triangle_mesh("/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/uv_generated_meshes/processed_4.obj")
print("Mesh loaded:", mesh)

# Check if the mesh has UVs
if len(mesh.triangle_uvs) == 0:
    print("Mesh does not have UVs!")
else:
    print("Mesh has UVs")

# Load the texture image
texture_image = o3d.io.read_image("/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/original_jpg_copy/4.jpg")

# Create a texture object and apply it to the mesh
mesh.textures = [texture_image]

# Display the mesh with texture
o3d.visualization.draw_geometries([mesh])'''


import open3d as o3d

# Load the mesh
mesh = o3d.io.read_triangle_mesh("/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/uv_generated_meshes/processed_4.obj")

# Check if mesh is valid
if not mesh.is_empty():
    print("Mesh is valid")
else:
    print("Mesh is empty or invalid")

# Visualize the mesh without texture
o3d.visualization.draw_geometries([mesh])

