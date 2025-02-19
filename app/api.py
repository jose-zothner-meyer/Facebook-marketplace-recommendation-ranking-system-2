"""
api.py

This file implements a FastAPI application that provides endpoints for image similarity search.
It loads a pre-trained and modified ResNet model for feature extraction, applies image transformations,
and uses a pre-built FAISS index for efficient nearest neighbor search. The API accepts an image
upload, extracts its embedding, and returns the IDs of similar images.
"""

import pickle                            # For loading and saving serialized Python objects.
import uvicorn                           # ASGI server to run the FastAPI app.
from fastapi import FastAPI              # FastAPI framework for building the API.
from fastapi.responses import JSONResponse  # For returning JSON responses.
from pydantic import BaseModel           # For data validation and serialization.
from PIL import Image                    # For image processing.
from fastapi import File, UploadFile, Form# For file uploads and form data handling.
import torch                             # PyTorch for tensor operations.
import torch.nn as nn                    # Neural network modules from PyTorch.
import json                              # For reading and writing JSON files.
from torchvision import models           # To load pre-trained models.
from torchvision.models import ResNet50_Weights  # Predefined weights for ResNet-50.
from torchvision import transforms       # For image transformation pipelines.

##############################################################
# Import your image processing script here                   #
# This script is purely to apply transforms onto raw images!!  #
##############################################################
import image_processor  # Module that contains the ImageProcessor class to apply transformations.

##############################################################################
# Define a modified ResNet model for feature extraction. In this API, we re-   #
# implement a simplified version directly, instead of importing resNet.py,     #
# for the sake of self-containment in the API.                                #
##############################################################################
class FineTunedResNet(nn.Module):
    """
    Modified ResNet-50 model for transfer learning.

    Key changes:
      - Directly replaces the original fully connected (fc) layer with a sequential block
        via self.new_layers that applies a ReLU activation followed by a linear mapping.
      - Combines the base ResNet-50 model and the new classification head into a single model.
      - Freezes all layers except for those in 'layer4' and 'fc', enabling fine-tuning.
    """
    def __init__(self, num_classes: int) -> None:
        """
        Initializes the FineTunedResNet model.

        Args:
            num_classes (int): Number of output classes for the final layer.
        """
        super(FineTunedResNet, self).__init__()
        
        # Load the pre-trained ResNet-50 model with ImageNet weights.
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze all layers to avoid updating their weights during training.
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze only layers 'layer4' and 'fc' to allow fine-tuning.
        for name, layer in self.model.named_children():
            if name in ['layer4', 'fc']:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Replace the original fc layer with a new block:
        # Apply a ReLU activation and then a linear layer from 1000 to num_classes outputs.
        self.new_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1000, num_classes)  # Maps 1000 features to the desired output classes.
        )
        
        # Combine the pre-trained model and the new layers into a single sequential model.
        self.combined_model = nn.Sequential(
            self.model,
            self.new_layers
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor representing one or more images.

        Returns:
            torch.Tensor: Output tensor from the combined model.
        """
        return self.combined_model(x)

##############################################################################
# Model and Feature Extractor Initialization
##############################################################################
try:
    # Instantiate the model with 13 classes.
    model = FineTunedResNet(13)
    saved_weights = 'epoch_5.pth'  # Path to pre-trained weights (update if needed)
    # Load the saved state dictionary mapping and load it onto CPU.
    model.load_state_dict(torch.load(saved_weights, map_location="cpu"))
    # Modify the model for feature extraction by removing the final classification layer.
    # This creates a new sequential model with all layers except the last one.
    model = torch.nn.Sequential(*list(model.combined_model.children())[:-1])
    # "pass" is used here as a placeholder.
    pass
except Exception as e:
    raise OSError("Error in creating Feature Extractor") from e

##############################################################################
# FAISS Index and Embeddings Loading
##############################################################################
try:
    # Load the FAISS index from a pickle file.
    faiss_index = pickle.load(open("faiss_index.pkl", "rb"))
    # Load image embeddings from a JSON file.
    image_embeddings = json.load(open("image_embeddings.json", "rb"))
    # Load the image ids mapping from a pickle file.
    image_ids = pickle.load(open("image_ids.pkl", "rb"))
    pass  # Placeholder.
except Exception as e:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location") from e

##############################################################################
# Define a transformer function to create the image transformation pipeline.
##############################################################################
def transformer(self):
    """
    Creates a transformation pipeline for processing input images.

    This pipeline performs the following steps:
      1. Resizes the image to 256x256 pixels.
      2. Converts the image to a PyTorch tensor.
      3. Normalizes the tensor using ImageNet mean and standard deviation.

    Returns:
        torchvision.transforms.Compose: A composition of transformations.
    """
    pipeline_transformer = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the image.
        transforms.ToTensor(),           # Convert to tensor.
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # Normalize using ImageNet parameters.
    ])
    return pipeline_transformer

##############################################################################
# FastAPI Application Initialization and Endpoints
##############################################################################
app = FastAPI()  # Initialize the FastAPI app.
print("Starting server")  # Log server start.

@app.get('/healthcheck')
def healthcheck():
    """
    Healthcheck endpoint to verify the API server is running.

    Returns:
        dict: A simple message indicating the server is up.
    """
    msg = "API is up and running!"
    return {"message": msg}

@app.post('/predict/similar_images')
def predict_combined(image: UploadFile = File(...)):
    """
    Endpoint to predict similar images given an uploaded image.

    Steps:
      1. Reads the uploaded image and converts it to RGB.
      2. Applies the transformation pipeline from the image_processor module.
      3. Extracts the image embedding using the pre-trained feature extractor.
      4. Uses the FAISS index to retrieve images similar to the input image.

    Args:
        image (UploadFile): The uploaded image file.

    Returns:
        JSONResponse: A JSON response containing similar image IDs.
    """
    # Step 1: Load the raw image using PIL and convert to RGB.
    pil_image = Image.open(image.file).convert("RGB")

    # Step 2: Use the ImageProcessor's transformer to apply necessary image transformations.
    img_proc = image_processor.ImageProcessor()  # Instantiate the ImageProcessor class.
    transformer_pipeline = img_proc.transformer()   # Retrieve the transformation pipeline.
    processed_image = transformer_pipeline(pil_image) # Apply the transformation.
    
    # Ensure the processed image tensor has a batch dimension.
    if processed_image.dim() == 3:
        processed_image = processed_image.unsqueeze(0)

    # Step 3: Extract image embedding using the pre-configured feature extractor model.
    with torch.no_grad():
        embedding_tensor = model(processed_image)  # Forward pass through the feature extractor.
        # Flatten the embedding, detach from computation graph, move to CPU,
        # and reshape it to (1, embedding_dimension) for FAISS search.
        embedding = embedding_tensor.flatten().detach().cpu().numpy().reshape(1, -1)

    # Step 4: Use the FAISS index to search for similar images.
    D, I = faiss_index.search(embedding, k=5)  # Search using k=5 nearest neighbors.
    similar_images = [image_ids[idx] for idx in I[0]]  # Retrieve corresponding image IDs.

    # Return similar image IDs as JSON response.
    return JSONResponse(content={
        "similar_index": similar_images
    })

##############################################################################
# Run the FastAPI Application using Uvicorn
##############################################################################
if __name__ == '__main__':
    # Run the Uvicorn server specifying the application instance (app) from this file.
    uvicorn.run("api:app", host="0.0.0.0", port=8080)