import os
import trimesh
import numpy as np
from scipy.spatial import cKDTree
import concurrent.futures

def get_file_names(directory):
    # List to store file names
    file_names = []

    # Iterate over all files in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".obj"):
                file_names.append(file)

    return file_names

def partition_mesh(mesh, chunk_size):
    """
    Partition the mesh into chunks using a k-d tree and return the chunks.
    """
    # Get the vertices and create a k-d tree
    vertices = mesh.vertices
    tree = cKDTree(vertices)
    
    # Define the ranges for partitioning
    x_min, x_max = int(np.floor(vertices[:, 0].min() / chunk_size) * chunk_size), int(np.ceil(vertices[:, 0].max() / chunk_size) * chunk_size)
    y_min, y_max = int(np.floor(vertices[:, 1].min() / chunk_size) * chunk_size), int(np.ceil(vertices[:, 1].max() / chunk_size) * chunk_size)
    z_min, z_max = int(np.floor(vertices[:, 2].min() / chunk_size) * chunk_size), int(np.ceil(vertices[:, 2].max() / chunk_size) * chunk_size)

    x_range = np.arange(x_min, x_max + chunk_size, chunk_size)
    y_range = np.arange(y_min, y_max + chunk_size, chunk_size)
    z_range = np.arange(z_min, z_max + chunk_size, chunk_size)

    chunks = {}
    
    # Partition the vertices into chunks
    for x in x_range:
        for y in y_range:
            for z in z_range:
                # Define the bounding box for the chunk
                min_bound = np.array([x, y, z])
                max_bound = min_bound + chunk_size
                
                # Find the vertices within the bounding box
                indices = tree.query_ball_point((min_bound + max_bound) / 2, chunk_size / 2)
                
                if indices:
                    chunk_key = (x, y, z)
                    chunks[chunk_key] = indices

    return chunks

def save_mesh_chunks(mesh, chunks, current_directory, chunk_size, i):
    """
    Save the mesh chunks to disk.
    """
    for chunk_key, indices in chunks.items():
        filtered_vertices = mesh.vertices[indices]
        
        # Create a mapping from old vertex indices to new ones
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}

        # Filter faces to only include those whose all vertices are within the chunk
        filtered_faces = []
        for face in mesh.faces:
            if all(vertex in vertex_map for vertex in face):
                filtered_faces.append([vertex_map[vertex] for vertex in face])

        if filtered_faces:
            chunk_mesh = trimesh.Trimesh(vertices=filtered_vertices, faces=np.array(filtered_faces))
            
            if mesh.visual.kind == 'texture' and mesh.visual.uv is not None:
                chunk_mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv[indices])
            
            output_dir = f"{current_directory}/output/clipped_meshes/Scroll1/{chunk_key[0]}_{chunk_key[1]}_{chunk_key[2]}_xyz"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            chunk_mesh.export(output_dir + f"/mesh_{i}_{chunk_key[0]}_{chunk_key[1]}_{chunk_key[2]}_xyz.obj")

def process_mesh(mesh, current_directory, chunk_size, i):
    chunks = partition_mesh(mesh, chunk_size)
    save_mesh_chunks(mesh, chunks, current_directory, chunk_size, i)

def main(meshes, current_directory, chunk_size):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_mesh, meshes[i], current_directory, chunk_size, i) for i in range(len(meshes))]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Task generated an exception: {e}")

if __name__ == "__main__":
    obj_path = "/Volumes/16TB_RAID_0/Scroll1/segments/objs"#path to folder with obj files
    obj_files = get_file_names(obj_path)
    print(obj_files)
    meshes = []
    for obj in obj_files:
        meshes.append(trimesh.load_mesh(f"{obj_path}/{obj}"))
    print("meshes loaded")
    current_directory = os.getcwd()
    chunk_size = 256  # Replace with your chunk size in units

    main(meshes, current_directory, chunk_size)