import os
import pymeshlab
import open3d as o3d


# Input and output directories
input_folder = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/generated_meshes1"
output_folder = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/uv_generated_meshes"
# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get all OBJ files in the folder
mesh_files = [f for f in os.listdir(input_folder) if f.endswith('.obj') or f.endswith('.ply')]

# Process each mesh file
for mesh_file in mesh_files:
    print(f"Processing: {mesh_file}")

    # Load the mesh
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(os.path.join(input_folder, mesh_file))

    # Apply processing steps
    ms.meshing_remove_duplicate_vertices()  # Remove duplicate vertices
    ms.meshing_remove_unreferenced_vertices()  # Clean unreferenced vertices
    ms.meshing_repair_non_manifold_edges()  # Fix non-manifold edges
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=5000)  # Reduce poly count

    # **Step 1: UV Unwrapping** (Better than "per-wedge" method)
    ms.compute_texcoord_parametrization_triangle_trivial_per_wedge()

    # **Step 2: UV Mapping** (Assigns UV coordinates)
    ms.compute_texcoord_transfer_wedge_to_vertex()
    
    # Save the processed mesh
    output_file = os.path.join(output_folder, f"processed_{mesh_file}")
    ms.save_current_mesh(output_file)

    print(f"Saved: {output_file}")

print(" Batch Processing Completed!")



# Load the mesh
#mesh = o3d.io.read_triangle_mesh("/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/uv_generated_meshes/processed_4.obj")

# Display the mesh
#o3d.visualization.draw_geometries([mesh])

