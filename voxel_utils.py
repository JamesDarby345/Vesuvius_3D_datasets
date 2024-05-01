from concurrent.futures import ProcessPoolExecutor
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage.morphology import binary_dilation, ball

def build_face_voxels_lists(obj_paths, vertex_dicts=None, connected_vertices_graphs=None, max_workers=12, roi_x=[0,10000000000], roi_y=[0,10000000000], roi_z=[0,10000000000]):
    if not vertex_dicts or not connected_vertices_graphs:
        vertex_dicts, connected_vertices_graphs = build_vertex_dicts(obj_paths, roi_x, roi_y, roi_z)
    component_faces_list = []
    for connected_vertices_graph in connected_vertices_graphs:
        components, component_faces = find_connected_components_and_faces(connected_vertices_graph)
        print(f'Number of connected components: {len(components)}')
        print("component faces:", len(component_faces))
        # print(component_faces[0])
        component_faces_list.append(component_faces)
    

    connected_faces_in_roi_list = []

    k = 0
    for component_faces in component_faces_list:
        connected_faces_in_roi = [[] for _ in range(len(component_faces))]
        i = 0
        vertex_dict = vertex_dicts[k]
        for connected_component in component_faces:
            for face in connected_component:
                v1, v2, v3 = face
                vx1, vy1, vz1 = vertex_dict[v1]
                vx2, vy2, vz2 = vertex_dict[v2]
                vx3, vy3, vz3 = vertex_dict[v3]
                connected_faces_in_roi[i].append(((vx1, vy1, vz1), (vx2, vy2, vz2), (vx3, vy3, vz3)))
            i += 1
        connected_faces_in_roi_list.append(connected_faces_in_roi)
        k += 1

    # if connected_faces_in_roi_list and connected_faces_in_roi_list[0]:
    #     print("connected faces in roi list:", connected_faces_in_roi_list[0][0])
    # else:
    #     print("connected_faces_in_roi_list is empty or the first element is empty")
    print("length of connected faces in roi:", len(connected_faces_in_roi_list))

    # Assuming connected_faces_in_roi_list is defined above
    face_voxels_list = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for connected_faces_in_roi in connected_faces_in_roi_list:
            # Map the calculate_face_voxels function to all faces in the ROI concurrently
            results = list(executor.map(calculate_face_voxels, connected_faces_in_roi))

            # Flatten the list of sets into a single set and append to face_voxels_list
            for face_voxels in results:
                if face_voxels:  # Skip if the set is empty
                    print("number of face voxels:", len(face_voxels))
                    face_voxels_list.append(face_voxels)

    return face_voxels_list

def build_vertex_dicts(obj_paths, roi_x=[0,10000000000], roi_y=[0,10000000000], roi_z=[0,10000000000], scale_factor=1):
    vertex_dicts = []
    connected_vertices_graphs = []
    for obj_path in obj_paths:
        vertex_dict = {}
        connected_vertices_graph = {}
        with open(obj_path, 'r') as fd:
            obj_vertex_index = 0

            # assumes vertices are listed first in the obj file, before faces
            # this should always be the case
            for line in fd:
                line = line.strip()
                words = line.split()
                if words[0] == 'v':
                    obj_vertex_index += 1 #obj file faces are 1 indexed
                    # Extract the 3D coordinates of the vertex
                    vertex = line.split()
                    x, y, z = map(float, vertex[1:])

                    # Filter the vertices not within the ROI
                    if x < roi_x[0] or x > roi_x[1] or y < roi_y[0] or y > roi_y[1] or z < roi_z[0] or z > roi_z[1]:
                        continue
                    
                    vx = round(x - roi_x[0])
                    vy = round(y - roi_y[0])
                    vz = round(z - roi_z[0])

                    #TODO: is this the best way? -> small quantization error
                    if vx == roi_x[1]:
                        vx = roi_x[1] - 1
                    if vy == roi_y[1]:
                        vy = roi_y[1] - 1
                    if vz == roi_z[1]:
                        vz = roi_z[1] - 1
                        
                    vertex_dict[obj_vertex_index] = (vx, vy, vz)
                elif words[0] == 'f':
                    face = line.split()
                    # Extract the vertex indices of the face
                    v1, v2, v3 = [int(face[i].split('/')[0]) for i in range(1, 4)]

                    # Filter the faces not within the ROI
                    if v1 not in vertex_dict or v2 not in vertex_dict or v3 not in vertex_dict:
                        continue
                    
                    # Add the vertex indices of the face to the connected vertices graph
                    if v1 not in connected_vertices_graph:
                        temp = []
                    else:
                        temp = connected_vertices_graph[v1]
                    temp.append((v2, v3))

                    connected_vertices_graph[v1] = temp

                    if v2 not in connected_vertices_graph:
                        temp = []
                    else:
                        temp = connected_vertices_graph[v2]
                    temp.append((v1, v3))
                    connected_vertices_graph[v2] = temp

                    if v3 not in connected_vertices_graph:
                        temp = []
                    else:
                        temp = connected_vertices_graph[v3]
                    temp.append((v1, v2))
                    connected_vertices_graph[v3] = temp

                    # face_vertices_in_roi.append(((vx1, vy1, vz1), (vx2, vy2, vz2), (vx3, vy3, vz3)))
            connected_vertices_graphs.append(connected_vertices_graph.copy())
            vertex_dicts.append(vertex_dict.copy())
            connected_vertices_graph.clear()
            vertex_dict.clear()
    return vertex_dicts, connected_vertices_graphs

def build_neighbor_graph(face_graph):
    graph = {}
    for key_vertex, face_tuples in face_graph.items():
        for v1, v2 in face_tuples:
            if key_vertex not in graph:
                graph[key_vertex] = set()
            graph[key_vertex].add(v1)
            graph[key_vertex].add(v2)
            
            if v1 not in graph:
                graph[v1] = set()
            graph[v1].add(key_vertex)
            graph[v1].add(v2)
            
            if v2 not in graph:
                graph[v2] = set()
            graph[v2].add(key_vertex)
            graph[v2].add(v1)
    return graph

def dfs(vertex, graph, visited, component):
    visited.add(vertex)
    component.add(vertex)
    for neighbor in graph[vertex]:
        if neighbor not in visited:
            dfs(neighbor, graph, visited, component)

def find_connected_components(graph):
    simplified_graph = build_neighbor_graph(graph)
    visited = set()
    components = []
    
    for vertex in simplified_graph:
        if vertex not in visited:
            component = set()
            dfs(vertex, simplified_graph, visited, component)
            components.append(component)
    
    return components

def find_connected_components_and_faces(graph):
    # Find connected components as before
    simplified_graph = build_neighbor_graph(graph)
    visited = set()
    components = []
    
    for vertex in simplified_graph:
        if vertex not in visited:
            # print(f'Processing vertex {vertex}')
            component = set()
            dfs(vertex, simplified_graph, visited, component)
            components.append(component)
    
    # Reconstruct faces for each component
    component_faces = []
    for component in components:
        faces = set()
        for vertex in component:
            for v1, v2 in graph.get(vertex, []):
                if v1 in component and v2 in component:
                    # Sort the vertices of the face to avoid duplicates
                    face = tuple(sorted([vertex, v1, v2]))
                    faces.add(face)
        component_faces.append(faces)
    
    return components, component_faces

def barycentric_coords(p, a, b, c):
    """
    Calculate barycentric coordinates of the point p with respect to the triangle 
    defined by points a, b, c.
    """
    v0 = [b[i] - a[i] for i in range(3)]
    v1 = [c[i] - a[i] for i in range(3)]
    v2 = [p[i] - a[i] for i in range(3)]
    d00 = sum(v0[i] * v0[i] for i in range(3))
    d01 = sum(v0[i] * v1[i] for i in range(3))
    d11 = sum(v1[i] * v1[i] for i in range(3))
    d20 = sum(v2[i] * v0[i] for i in range(3))
    d21 = sum(v2[i] * v1[i] for i in range(3))
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w

def point_in_triangle(p, a, b, c):
    """
    Check if point p is inside the triangle defined by points a, b, c.
    """
    u, v, w = barycentric_coords(p, a, b, c)
    return (u >= 0) and (v >= 0) and (w >= 0)

def voxels_intersecting_trianglular_face(a, b, c, bounds):
    """
    Return a list of voxel coordinates intersecting with the triangle defined by points a, b, c.
    bounds is a tuple of ((min_x, max_x), (min_y, max_y), (min_z, max_z)) defining the region to check.
    """
    voxels = []
    for x in range(bounds[0][0], bounds[0][1] + 1):
        for y in range(bounds[1][0], bounds[1][1] + 1):
            for z in range(bounds[2][0], bounds[2][1] + 1):
                center = (x + 0.5, y + 0.5, z + 0.5)  # Center of the voxel
                if point_in_triangle(center, a, b, c):
                    voxels.append((x, y, z))
    return voxels


def calculate_face_voxels(connected_face):
    face_voxels = set()
    num_intersecting_faces = len(connected_face)
    if num_intersecting_faces == 0:
        return face_voxels

    for i in range(num_intersecting_faces):
        # Your existing logic to define v1, v2, v3
        v1, v2, v3 = connected_face[i]
        vx1, vy1, vz1 = v1
        vx2, vy2, vz2 = v2
        vx3, vy3, vz3 = v3

        # Calculate bounds to limit the search space, assuming voxels_intersecting_trianglular_face is defined elsewhere
        min_vx, max_vx = min(vx1, vx2, vx3), max(vx1, vx2, vx3)
        min_vy, max_vy = min(vy1, vy2, vy3), max(vy1, vy2, vy3)
        min_vz, max_vz = min(vz1, vz2, vz3), max(vz1, vz2, vz3)

        new_face_voxels = voxels_intersecting_trianglular_face((vx1, vy1, vz1), (vx2, vy2, vz2), (vx3, vy3, vz3), ((min_vx, max_vx), (min_vy, max_vy), (min_vz, max_vz)))
        face_voxels.update(new_face_voxels)

    return face_voxels

def dilate_and_mask(face_voxels, volume_grid, selem_dilation, idx):
    print(f'Starting thread for surface {idx}')
    binary_volume_mask = np.zeros((500, 500, 500), dtype=bool)
    for voxel in face_voxels:
        x, y, z = voxel
        binary_volume_mask[z, y, x] = 1

    binary_volume_mask_dilated = binary_dilation(binary_volume_mask, selem_dilation)
    masked_volume = volume_grid * binary_volume_mask_dilated
    print(f'Finished thread for surface {idx}')
    return binary_volume_mask_dilated, masked_volume, idx

"""
Uses the binary masks to create COCO annotations for each surface in the cell
for every xyz slice, these are combined into a single COCO annotation file
This coco annotation files can get quite big ~300MB for a single cell
Can cause issues with text editors and some json viewers
"""
def update_coco_annotations(binary_mask_list, cell_id, coco_dict, start_end_ignore=20, layer_skip=1):
    img_id = 0
    img_id_set = set()
    annotation_id = 0
    for i in range(binary_mask_list[0].shape[0]):
        if i < start_end_ignore or i > binary_mask_list[0].shape[0] - start_end_ignore or i % layer_skip != 0:
            continue
        # i = 450
        surface_id = 0
        for surface_mask in binary_mask_list:
            contours_x, _ = cv2.findContours(surface_mask[:, :, i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_y, _ = cv2.findContours(surface_mask[:, i, :].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_z, _ = cv2.findContours(surface_mask[i, :, :].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            #visualize the x contours
            # test_img = np.zeros((500, 500), dtype=np.uint8)
            # cv2.drawContours(test_img, contours_x, -1, (255, 255, 255), -1)
            # plt.imshow(test_img)
            # plt.show()

            #Apply DRY? -> create function, nah too specific
            coco_annotations = []
            coco_images = []
            img_id = abs(hash(f"{cell_id}_cmb_x_{i}"))
            for contour in contours_x:
                segmentation_x = contour.flatten().tolist()
                annotation_hash_id = abs(hash(f"{cell_id}_cmb_x_{i}_{annotation_id}"))
                coco_annotations.append({
                    "id": annotation_hash_id,
                    "image_id": img_id,
                    "category_id": 0,
                    "segmentation": [segmentation_x],
                    "area": cv2.contourArea(contour),
                    "bbox": cv2.boundingRect(contour),
                    "iscrowd": 0
                })
                annotation_id += 1

            if len(contours_x) > 0 and img_id not in img_id_set:
                coco_images.append({
                    "id": img_id,
                    "file_name": f"{cell_id}/{cell_id}_cmb_x_{i}.jpg",
                    "width": binary_mask_list[0].shape[0],
                    "height": binary_mask_list[0].shape[1]
                })
                img_id_set.add(img_id)

            img_id = abs(hash(f"{cell_id}_cmb_y_{i}"))
            for contour in contours_y:
                segmentation_y = contour.flatten().tolist()
                annotation_hash_id = abs(hash(f"{cell_id}_cmb_x_{i}_{annotation_id}"))
                coco_annotations.append({
                    "id": annotation_hash_id,
                    "image_id": img_id,
                    "category_id": 0,
                    "segmentation": [segmentation_y],
                    "area": cv2.contourArea(contour),
                    "bbox": cv2.boundingRect(contour),
                    "iscrowd": 0
                })
                annotation_id += 1

            if len(contours_y) > 0 and img_id not in img_id_set:
                coco_images.append({
                    "id": img_id,
                    "file_name": f"{cell_id}/{cell_id}_cmb_y_{i}.jpg",
                    "width": binary_mask_list[0].shape[0],
                    "height": binary_mask_list[0].shape[1]
                })
                img_id_set.add(img_id)

            img_id = abs(hash(f"{cell_id}_cmb_z_{i}"))
            
            for contour in contours_z:
                segmentation_z = contour.flatten().tolist()
                annotation_hash_id = abs(hash(f"{cell_id}_cmb_x_{i}_{annotation_id}"))
                coco_annotations.append({
                    "id": annotation_hash_id,
                    "image_id": img_id,
                    "category_id": 0,
                    "segmentation": [segmentation_z],
                    "area": cv2.contourArea(contour),
                    "bbox": cv2.boundingRect(contour),
                    "iscrowd": 0
                })
                annotation_id += 1

            if len(contours_z) > 0 and img_id not in img_id_set:
                coco_images.append({
                    "id": img_id,
                    "file_name": f"{cell_id}/{cell_id}_cmb_z_{i}.jpg",
                    "width": binary_mask_list[0].shape[0],
                    "height": binary_mask_list[0].shape[1]
                })
                img_id_set.add(img_id)

            coco_dict["annotations"].extend(coco_annotations)
            coco_annotations = []

            coco_dict["images"].extend(coco_images)
            coco_images = []

            surface_id += 1
        #  break #uncomment to test a single slice output
    return coco_dict

"""
Combines all surfaces intersecting with a xyz slice into one combined
slice for each xyz axis, currently saves output to disk matching 
coco annotations json file names. Takes awhile to execute
"""
def combine_and_save_slices(volume_grid, cell_id, binary_mask_list, masked_volume_list,  produce_masked_slices=False, start_end_ignore=20, layer_skip=1):

    # Create the folder structure
    
    folder_path = f"xyz_slices/{cell_id}"
    os.makedirs(folder_path, exist_ok=True)

    for i in range(masked_volume_list[0].shape[0]):
        if i < start_end_ignore or i > masked_volume_list[0].shape[0] - start_end_ignore or i % layer_skip != 0:
            continue
        x_slice_mask = np.zeros((500, 500), dtype=bool)
        y_slice_mask = np.zeros((500, 500), dtype=bool)
        z_slice_mask = np.zeros((500, 500), dtype=bool)
        for surface_mask in binary_mask_list:
            x_slice_mask = np.logical_or(x_slice_mask, surface_mask[:, :, i])
            y_slice_mask = np.logical_or(y_slice_mask, surface_mask[:, i, :])
            z_slice_mask = np.logical_or(z_slice_mask, surface_mask[i, :, :])

        if produce_masked_slices:
            x_slice = volume_grid[:, :, i] * x_slice_mask
            y_slice = volume_grid[:, i, :] * y_slice_mask
            z_slice = volume_grid[i, :, :] * z_slice_mask
        else:
            x_slice = volume_grid[:, :, i]
            y_slice = volume_grid[:, i, :]
            z_slice = volume_grid[i, :, :]

        # Save x slices
        x_slice_path = os.path.join(folder_path, f"{cell_id}_cmb_x_{i}.jpg")
        if np.sum(x_slice) > 0:
            plt.imsave(x_slice_path, x_slice, cmap='gray')
        
        # Save y slices
        y_slice_path = os.path.join(folder_path, f"{cell_id}_cmb_y_{i}.jpg")
        if np.sum(y_slice) > 0:
            plt.imsave(y_slice_path, y_slice, cmap='gray')

        # Save z slices
        z_slice_path = os.path.join(folder_path, f"{cell_id}_cmb_z_{i}.jpg")
        if np.sum(z_slice) > 0:
            plt.imsave(z_slice_path, z_slice, cmap='gray')