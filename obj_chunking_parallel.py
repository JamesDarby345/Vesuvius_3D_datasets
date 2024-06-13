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

def partition_mesh(mesh, chunk_size, padding=0):
    """
    Partition the mesh into chunks using a k-d tree and return the chunks.
    """
    # Get the vertices and create a k-d tree
    vertices = mesh.vertices
    # tree = cKDTree(vertices)
    
    # Define the ranges for partitioning
    x_min, x_max = int(np.floor(vertices[:, 0].min() / chunk_size) * chunk_size), int(np.ceil(vertices[:, 0].max() / chunk_size) * chunk_size)
    y_min, y_max = int(np.floor(vertices[:, 1].min() / chunk_size) * chunk_size), int(np.ceil(vertices[:, 1].max() / chunk_size) * chunk_size)
    z_min, z_max = int(np.floor(vertices[:, 2].min() / chunk_size) * chunk_size), int(np.ceil(vertices[:, 2].max() / chunk_size) * chunk_size)
    
    print("Ranges, x: ", x_min, x_max, "y:", y_min, y_max, "z:", z_min, z_max)
    #testing values
    # x_min= 3328
    # x_max= 3584
    # y_min=2560
    # y_max=2816
    # z_min=4864
    # z_max=5120

    x_range = np.arange(x_min, x_max + chunk_size, chunk_size)
    y_range = np.arange(y_min, y_max + chunk_size, chunk_size)
    z_range = np.arange(z_min, z_max + chunk_size, chunk_size)

    chunks = {}
    
    # Partition the vertices into chunks
    for x in x_range:
        for y in y_range:
            for z in z_range:
                # Define the bounding box for the chunk
                min_bound = np.array([x, y, z]) - padding
                max_bound = min_bound + chunk_size + 2*padding
                
                # Query the vertices within the bounding box
                indices = np.where(
                    (vertices[:, 0] >= min_bound[0]) & (vertices[:, 0] < max_bound[0]) &
                    (vertices[:, 1] >= min_bound[1]) & (vertices[:, 1] < max_bound[1]) &
                    (vertices[:, 2] >= min_bound[2]) & (vertices[:, 2] < max_bound[2])
                )[0]
                
                if len(indices) > 0:
                    chunk_key = (x, y, z)
                    chunks[chunk_key] = indices

    return chunks

def save_mesh_chunks(mesh, chunks, current_directory, chunk_size, padding, i):
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
            
            output_dir = f"{current_directory}/output/chunked_meshes_pad{padding}_chunk_size{chunk_size}/Scroll1/{chunk_key[0]}_{chunk_key[1]}_{chunk_key[2]}_xyz"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            chunk_mesh.export(output_dir + f"/{chunk_key[0]}_{chunk_key[1]}_{chunk_key[2]}_xyz_mesh_{i}.obj")

def process_mesh(mesh, current_directory, chunk_size, i, padding=0):
    chunks = partition_mesh(mesh, chunk_size, padding=padding)
    save_mesh_chunks(mesh, chunks, current_directory, chunk_size, padding, i)

def main(meshes, current_directory, chunk_size, padding):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_mesh, meshes[i], current_directory, chunk_size, i, padding=padding) for i in range(len(meshes))]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Task generated an exception: {e}")

if __name__ == "__main__":
    obj_path = "/Volumes/16TB_RAID_0/Scroll1/segments/objs" #path to folder with obj files
    obj_files = get_file_names(obj_path)
    print(obj_files)
    meshes = []
    for obj in obj_files:
        meshes.append(trimesh.load_mesh(f"{obj_path}/{obj}"))
    print("meshes loaded")
    current_directory = os.getcwd()
    chunk_size = 256  # Replace with chunk size in units
    padding = 50 # Replace with padding in units

    main(meshes, current_directory, chunk_size, padding)