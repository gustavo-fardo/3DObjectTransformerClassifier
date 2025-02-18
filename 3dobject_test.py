import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import keras
import os
import random
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Preprocess the OFF file into a point cloud
def read_off(file_path):
    with open(file_path, 'r') as file:
        if 'OFF' not in file.readline():
            raise ValueError("File is not in OFF format")
        n_vertices, _, _ = map(int, file.readline().split())
        vertices = [list(map(float, file.readline().split())) for _ in range(n_vertices)]
    return np.array(vertices)

def preprocess_point_cloud(vertices, num_points=1024):
    if len(vertices) > num_points:
        indices = np.random.choice(len(vertices), num_points, replace=False)
        vertices = vertices[indices]
    elif len(vertices) < num_points:
        padding = np.zeros((num_points - len(vertices), 3))
        vertices = np.vstack([vertices, padding])
    return vertices

# Step 2: Load only the test files
from process_data import load_off_files, label_dict

# Step 3: Define the Embedding Layer
from custom_transformer_layers import EmbeddingBlock, PositionalEncodingLayer

# Step 4: Define the Transformer Block
from custom_transformer_layers import TransformerBlock    

# Step 5: Load the saved model
def load_and_test_model(model_path, data_path):
    # Load the model
    model = tf.keras.models.load_model(model_path, custom_objects={
    'EmbeddingBlock': EmbeddingBlock,
    'PositionalEncodingLayer': PositionalEncodingLayer,
    'TransformerBlock': TransformerBlock
    })
    
    # Load only the test files
    train_point_clouds, train_labels, test_point_clouds, test_labels, label_dict = load_off_files(data_path)
    
    # Reshape for model input (batch_size, num_points, 3)
    test_point_clouds = np.expand_dims(test_point_clouds, axis=-1)
    
    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_point_clouds, test_labels)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)
    
    # Make predictions on the test set
    predictions = model.predict(test_point_clouds)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calculate precision, recall, and F1-score
    report = classification_report(test_labels, predicted_classes, target_names=label_dict.keys())
    print("Classification Report:\n", report)
    
    # Compute confusion matrix
    cm = confusion_matrix(test_labels, predicted_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_dict.keys(), yticklabels=label_dict.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # (Optional) Make predictions on the test set and visualize some samples

    # Generate 5 random indices
    random_indices = random.sample(range(len(test_point_clouds)), 5)
    
    # Print predictions for the random samples and save 2D renders
    print("Predictions for 5 random test samples:")
    for i in random_indices:
        predicted_class = np.argmax(predictions[i])
        predicted_label = list(label_dict.keys())[list(label_dict.values()).index(predicted_class)]
        true_label = list(label_dict.keys())[list(label_dict.values()).index(test_labels[i])]
        print(f"Sample {i}: Predicted Class = {predicted_label}, True Class = {true_label}")
        
        # Get the point cloud for the sample
        point_cloud = test_point_clouds[i].squeeze()
        
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
        
        # Set plot title and labels
        ax.set_title(f'Sample {i}: Predicted Class = {predicted_label}, True Class = {true_label}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set the limits for the axes to ensure the whole point cloud is visible with extra margin
        margin = 0.1  # Define a margin
        ax.set_xlim([point_cloud[:, 0].min() - margin, point_cloud[:, 0].max() + margin])
        ax.set_ylim([point_cloud[:, 1].min() - margin, point_cloud[:, 1].max() + margin])
        ax.set_zlim([point_cloud[:, 2].min() - margin, point_cloud[:, 2].max() + margin])
        
        # Save the plot as a 2D render
        plt.savefig(f'sample_{i}_render.png')
        plt.close()

# Define paths
model_path = '3Dobjectransform.keras'  # Path to the saved model
data_path = 'ModelNet10/'  # Path to the dataset

# Load the model and test on the test files
load_and_test_model(model_path, data_path)