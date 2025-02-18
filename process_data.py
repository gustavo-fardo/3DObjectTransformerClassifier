import os
import numpy as np

# Encoding of the labels into integers
label_dict = { 'bathtub': 0, 
                'bed': 1, 
                'chair': 2, 
                'desk': 3,  
                'dresser': 4,  
                'monitor': 5, 
                'night_stand': 6, 
                'sofa': 7, 
                'table': 8, 
                'toilet': 9}
# Number of vertices taken from every OFF file (number of features extracted)
num_points = 1024

# Extract every single vertex from the OFF file, endcoded as in XYZ coordinates
# Returns a (num_vertices, 3)
def read_off(file_path):
    with open(file_path, 'r') as file:
        if 'OFF' not in file.readline():
            raise ValueError("File is not in OFF format")
        n_vertices, _, _ = map(int, file.readline().split())
        vertices = [list(map(float, file.readline().split())) for _ in range(n_vertices)]
    return np.array(vertices)

# Filter or pad with zeros the point cloud to have exactly num_points vertices
# Returns a (num_points, 3)
def preprocess_point_cloud(vertices, num_points=num_points):
    if len(vertices) > num_points:
        indices = np.random.choice(len(vertices), num_points, replace=False)
        vertices = vertices[indices]
    elif len(vertices) < num_points:
        padding = np.zeros((num_points - len(vertices), 3))
        vertices = np.vstack([vertices, padding])
    return vertices

# Makes the train/test split, loading the data and the labels
# Returns the train/test point clouds and labels, and the label dictionary 
def load_off_files(data_path):
    train_point_clouds = []
    train_labels = []
    test_point_clouds = []
    test_labels = []

    for label_name, label_index in label_dict.items():
        label_path = os.path.join(data_path, label_name)
        for split in ['train', 'test']:
            split_path = os.path.join(label_path, split)
            if os.path.exists(split_path):
                for file_name in os.listdir(split_path):
                    if file_name.endswith('.off') and file_name != '.DS_Store':
                        file_path = os.path.join(split_path, file_name)
                        vertices = read_off(file_path)
                        point_cloud = preprocess_point_cloud(vertices, num_points=num_points)
                        
                        if split == 'train':
                            train_point_clouds.append(point_cloud)
                            train_labels.append(label_index)
                        elif split == 'test':
                            test_point_clouds.append(point_cloud)
                            test_labels.append(label_index)
    
    return (
        np.array(train_point_clouds), np.array(train_labels),
        np.array(test_point_clouds), np.array(test_labels),
        label_dict
    )
