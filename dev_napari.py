import nrrd
import napari
import os
import numpy as np
from skimage.segmentation import find_boundaries
from napari.qt import thread_worker

current_directory = os.getcwd()
file_name = 'data/manual_1_raw.nrrd'
file_path = os.path.join(current_directory, file_name)
data, _ = nrrd.read(file_path)
label_name = 'data/manual_1_label.nrrd'
label_path = os.path.join(current_directory, label_name)
label, _ = nrrd.read(label_path)

# Function to generate a borders-only view of the labels
def generate_borders_view(labels):
    # Initialize the array to hold borders
    borders = np.zeros_like(labels, dtype=bool)
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        print(label)
        if label == 0:
            continue  # Skip background
        # Find boundaries for the current label
        label_boundaries = find_boundaries(labels == label, mode='outer')

        # Combine the boundaries of all labels
        borders[label_boundaries] = label
    
    return borders

# Function to display labels with borders
def display_labels_with_borders(viewer, labels, label_name):
    borders = generate_borders_view(labels)
    
    # Create a new layer to display borders with the same colormap as the original labels
    borders_layer = np.zeros_like(labels)
    borders_layer[borders] = labels[borders]
    
    # Add the borders layer to Napari
    viewer.add_labels(borders_layer, name='Borders View', color={i: c for i, c in enumerate(viewer.layers[label_name].colormap.colors)})

# Initialize the Napari viewer
viewer = napari.Viewer()

label_name = '3D NRRD Label'

# Add the 3D data to the viewer
image_layer =  viewer.add_image(data, colormap='gray', name='3D NRRD Data')
labels_layer = viewer.add_labels(label, name=label_name)

@thread_worker
def update_borders(labels):
    print("update_borders")
    return generate_borders_view(labels)


@viewer.bind_key('b')
def regenerate_borders(viewer):
    if 'borders' in viewer.layers:
        msg = 'regenerating borders'
    else:
        msg = 'generating borders'
    viewer.status = msg
    print(msg)
    worker = update_borders(labels_layer.data)
    worker.returned.connect(update_layer)
    print('worker started')

def update_layer(borders):
    print('updating layer')
    if 'borders' in viewer.layers:
        viewer.layers['borders'].data = borders
    else:
        viewer.add_labels(borders, name='borders')
    print('worker finished')

@viewer.bind_key('q')
def decrease_brush_size(viewer):
    msg = 'decrease brush size'
    viewer.status = msg
    print(msg)
    labels_layer.brush_size = labels_layer.brush_size - 1

@viewer.bind_key('e')
def increase_brush_size(viewer):
    msg = 'increase brush size'
    viewer.status = msg
    print(msg)
    labels_layer.brush_size = labels_layer.brush_size + 1

@viewer.bind_key('s')
def toggle_show_selected_label(viewer):
    msg = 'toggle show selected label'
    viewer.status = msg
    print(msg)
    labels_layer.show_selected_label = not labels_layer.show_selected_label

@viewer.bind_key('a')
def decrease_selected_label(viewer):
    msg = 'decrease selected label'
    viewer.status = msg
    print(msg)
    labels_layer.selected_label = labels_layer.selected_label - 1

@viewer.bind_key('d')
def increase_selected_label(viewer):
    msg = 'increase selected label'
    viewer.status = msg
    print(msg)
    labels_layer.selected_label = labels_layer.selected_label + 1

# Function to capture cursor information when 'w' is pressed
def capture_cursor_info(event):
    # Get cursor position in world coordinates
    position = viewer.cursor.position

    # Convert world coordinates to data indices
    indices = tuple(int(np.round(coord)) for coord in position)

    # Get the value of the label under the cursor
    label_value = labels_layer.data[indices]

    # Print the cursor position and label value
    print(f"Cursor Position: {indices}, Label Value: {label_value}")
    labels_layer.selected_label = label_value

# Bind the function to the 'w' key press event
@viewer.bind_key('w')
def on_w_key(event):
    capture_cursor_info(event)

# Make axes visible by default
viewer.axes.visible = True

# Start the Napari event loop
napari.run()