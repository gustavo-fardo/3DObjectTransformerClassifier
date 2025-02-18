import numpy as np
from tensorflow.keras import layers, models
import os
from tensorflow.python.client import device_lib
from tensorflow.keras.utils import plot_model
import random

from process_data import read_off, preprocess_point_cloud, num_points

# Define hyperparameters -----------------------

# Network input dimensions
input_shape = (num_points, 3)  # Shape of the input layer (num_points, num_axis)
embedding_dim = 64             # Expands the feature space from num_axis to embedding_dim (num_points, embedding_dim)

# Transformer block size (total number of heads will be num_heads * num_layers)
num_layers = 2                 # Number of transformer blocks
num_heads = 4                  # Number of attention heads

# Feed forward network size
ff_dim = 128                   # Hidden layer size in feed forward network inside transformer

# Training parameters
epochs = 20
batch_size = 128

# ----------------------------------------------

# List all available devices
devices = device_lib.list_local_devices()
for device in devices:
    print(device)

# Step 1: Preprocess the OFF file into a point cloud
from process_data import read_off, preprocess_point_cloud, num_points

# Step 2: Prepare data
from process_data import load_off_files

# Step 3: Define the Embedding Layer
from custom_transformer_layers import EmbeddingBlock

# Step 4: Define the Transformer Block
from custom_transformer_layers import TransformerBlock    

# Step 5: Build the full model
def build_transformer_model(input_shape, num_classes, embedding_dim=64, num_heads=4, ff_dim=128, num_layers=2):
    inputs = layers.Input(shape=input_shape)

    # Embedding block
    embedding_layer = EmbeddingBlock(embedding_dim=embedding_dim)
    x = embedding_layer(inputs)

    # Transformer blocks
    transformer_block = TransformerBlock(embedding_dim, num_heads, ff_dim)
    for _ in range(num_layers):
        x = transformer_block(x)

    # Global pooling and output
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model


# Step 6 - Train and evaluate the model

data_path = 'ModelNet10/'
# Load the data
train_point_clouds, train_labels, test_point_clouds, test_labels, label_dict = load_off_files(data_path)

# Reshape for model input
train_point_clouds = np.expand_dims(train_point_clouds, axis=-1)
test_point_clouds = np.expand_dims(test_point_clouds, axis=-1)

# Build and compile the model
model = build_transformer_model(input_shape=input_shape, num_classes=len(label_dict))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display the model architecture
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.summary()

# Train the model
history = model.fit(train_point_clouds, train_labels, epochs=epochs, batch_size=batch_size)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_point_clouds, test_labels)

# Print test results
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Make predictions on the test set
predictions = model.predict(test_point_clouds)

# Generate 5 random indices
random_indices = random.sample(range(len(test_point_clouds)), 5)

# Print predictions for the random samples
print("Predictions for 5 random test samples:")
for i in random_indices:
    print(f"Sample {i}: Predicted Class = {np.argmax(predictions[i])}, True Class = {test_labels[i]}")

model.save('3Dobjectransform.keras')  # SavedModel format (recommended)