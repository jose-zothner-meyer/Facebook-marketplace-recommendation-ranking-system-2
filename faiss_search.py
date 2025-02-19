"""
faiss_search.py

This script loads pre-computed image embeddings and creates a FAISS index for efficient similarity search.
It then provides an example of using the FAISS index to retrieve images similar to a given query image.
Additionally, it defines a helper function, extract_embedding(), to load an image, process it using a feature extractor,
and output its embedding.
"""

import json  # Module for reading/writing JSON files.
import numpy as np  # For working with numerical arrays.
import faiss  # Facebook AI Similarity Search library for efficient similarity search.
from PIL import Image  # Library for image processing.
import torch  # PyTorch library for tensor operations.
import torchvision.transforms as transforms  # For image transformation pipelines.

# Import the pre-trained model architecture to extract features.
from resNet import FineTunedResNet
# Import FileHandler utility for saving objects to disk.
from file_handler import FileHandler

# ---------------------------------------------------------------------------------------------------
# Load the saved dictionary containing image embeddings.
with open('data/output/image_embeddings.json', 'r') as f:
    # Load the JSON file into a Python dictionary.
    embeddings_dict = json.load(f)

# Extract image IDs (keys of the dictionary).
image_ids = list(embeddings_dict.keys())
# Save the list of image ids as a pickle file using the FileHandler.
FileHandler.pickle_obj(image_ids, "data/output/image_ids.pkl")
# Convert the embeddings from the dictionary values into a numpy array with dtype float32.
embeddings = np.array(list(embeddings_dict.values()), dtype=np.float32)

# ---------------------------------------------------------------------------------------------------
# Create a FAISS index using L2 distance (Euclidean distance).
dimension = embeddings.shape[1]  # Get the dimension of each embedding vector.
index = faiss.IndexFlatL2(dimension)  # Create an index for vectors of the specified dimension using L2 metric.
index.add(embeddings)  # Add the embeddings to the index for search.
# Save the constructed index to disk as a pickle file.
FileHandler.pickle_obj(index, "data/output/faiss_index.pkl")

# ---------------------------------------------------------------------------------------------------
# Example usage: Query the index with the first embedding from the loaded embeddings.
D, I = index.search(embeddings[:1], k=5)
# Extract the similar image ids corresponding to the indices returned from the search.
similar_images = [image_ids[idx] for idx in I[0]]
print("Existing image example")
print("Similar images (from stored embeddings) to the first image:", similar_images)

# ---------------------------------------------------------------------------------------------------
def extract_embedding(image_path):
    """
    Loads an image, applies transforms, and extracts a feature embedding using a pre-trained model.

    Args:
        image_path (str): Path to the query image file.

    Returns:
        numpy.ndarray: Flattened feature embedding as a numpy array.
    """
    # Define a transformation pipeline that should match the one used during training.
    transform = transforms.Compose([
        transforms.Resize(256),         # Resize the image to ensure consistency.
        transforms.CenterCrop(224),     # Center crop the image to 224x224 pixels.
        transforms.ToTensor(),          # Convert the image into a PyTorch tensor.
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean.
                             std=[0.229, 0.224, 0.225])   # Normalize with ImageNet standard deviation.
    ])
    
    # Open the image file and convert it to RGB format.
    image = Image.open(image_path).convert("RGB")
    # Apply the defined transformations and add a batch dimension.
    image_tensor = transform(image).unsqueeze(0)  # Shape becomes [1, C, H, W].

    # Set the device to GPU if available, otherwise CPU.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Define the number of classes for the model.
    num_classes = 13  # Update this if the number of classes is different.
    # Specify the path to the pre-trained model weights.
    saved_weights = 'data/model_evaluation/model_20250211-192554/weights/epoch_5.pth'  # Update if needed.
    
    # Initialize the model architecture.
    model_training = FineTunedResNet(num_classes)
    # Load the model weights from the saved file, mapping to the selected device.
    model_training.load_state_dict(torch.load(saved_weights, map_location=device))
    # Move the model to the appropriate device.
    model_training.to(device)
    
    # Convert the model into a feature extractor by removing its classification head.
    # Note: Adjust the slicing based on your model's structure.
    model_extractor = torch.nn.Sequential(*list(model_training.combined_model.children())[:-1])
    model_extractor.to(device)
    model_extractor.eval()  # Set the model to evaluation mode to disable dropout, etc.
    
    # Extract the embedding without calculating gradients.
    with torch.no_grad():
        # Pass the image tensor through the feature extractor.
        embedding = model_extractor(image_tensor.to(device))
        # Flatten the embedding, move it to the CPU, and convert to a numpy array.
        embedding = embedding.flatten().cpu().numpy()
    
    # Return the computed embedding.
    return embedding

# ---------------------------------------------------------------------------------------------------
# Provide the path to your query image.
query_image_path = 'images/0a40eb4e-b033-4a98-9374-e1d4ad2943ac.jpg'  # Update this path as needed.

# Extract the embedding for the query image using the defined function.
query_embedding = extract_embedding(query_image_path)
# Reshape embedding to (1, dimension) for FAISS search.
query_embedding = query_embedding.reshape(1, dimension)

# Search the FAISS index using the extracted query embedding.
D, I = index.search(query_embedding, k=5)
# Retrieve the similar image ids based on search results.
similar_images = [image_ids[idx] for idx in I[0]]
print("Raw image example")
print("Similar images for the query image:", similar_images)