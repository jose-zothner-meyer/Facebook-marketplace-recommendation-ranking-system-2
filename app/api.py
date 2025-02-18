import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
import json
from pydantic import BaseModel
from torchvision import models
from torchvision.models import ResNet50_Weights
from torchvision import transforms
##############################################################
# TODO                                                       #
# Import your image processing script here                   #
import image_processor
# This script is purely to apply transforms onto raw images!!
##############################################################

class FineTunedResNet(nn.Module):
    """
    Modified ResNet-50 model for transfer learning.
    
    Key changes:
      - Directly replaces the original fc layer with a sequential block (via self.new_layers) instead of creating a separate combined model.
      - This avoids applying AdaptiveAvgPool2d twice.
      - Only layer4 and fc layers (inside the original model) are unfrozen.
    """
    def __init__(self, num_classes: int) -> None:
        super(FineTunedResNet, self).__init__()
        
        # Load pre-trained ResNet-50 using the new weights syntax.
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze all layers initially.
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze parameters only in layer4 and the fc layer.
        for name, layer in self.model.named_children():
            if name in ['layer4', 'fc']:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Replace the fc layer with a sequential block.
        # According to the teacher’s design, we first apply a ReLU activation,
        # then use a linear layer to map the 1000 features to the desired number of classes.
        self.new_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1000, num_classes)  # Final layer outputs class scores.
        )
        # Combine the base model and the new classification head.
        self.combined_model = nn.Sequential(
            self.model,
            self.new_layers
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using the modified ResNet-50 model."""
        return self.combined_model(x)

try:
    model = FineTunedResNet(13)
    saved_weights = 'epoch_5.pth'  # update if needed
    model.load_state_dict(torch.load(saved_weights, map_location="cpu"))
    model = torch.nn.Sequential(*list(model.combined_model.children())[:-1])
    pass
except:
    raise OSError("Error in creating Feature Extractor")

try:
    faiss_index = pickle.load(open("faiss_index.pkl", "rb"))
    image_embeddings = json.load(open("image_embeddings.json", "rb"))
    image_ids = pickle.load(open("image_ids.pkl", "rb"))
    pass
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")

def transformer(self):
        """
        The function `transform_pipeline` creates a transformation pipeline for image processing in
        PyTorch.
        
        Returns:
          The `transform_pipeline` method returns a composition of transformations that includes
        resizing the image to (256, 256) pixels, converting it to a PyTorch tensor, and normalizing the
        image using the specified mean and standard deviation values.
        """
        pipeline_transformer = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        return pipeline_transformer


app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}
  
@app.post('/predict/similar_images')
def predict_combined(image: UploadFile = File(...)):
    # Step 1: Load the raw image
    pil_image = Image.open(image.file).convert("RGB")

    # Step 2: Apply necessary transforms using your image_processor module
    # Instantiate ImageProcessor and use its transformer method to get the pipeline.
    img_proc = image_processor.ImageProcessor()
    transformer = img_proc.transformer()
    processed_image = transformer(pil_image)
    # Ensure the image tensor has a batch dimension
    if processed_image.dim() == 3:
        processed_image = processed_image.unsqueeze(0)

    # Step 3: Extract embeddings using the FeatureExtractor (model is already setup to act as one)
    with torch.no_grad():
        embedding_tensor = model(processed_image)
        # Flatten and convert to numpy array; reformat as 2D (batch, dimension)
        embedding = embedding_tensor.flatten().detach().cpu().numpy().reshape(1, -1)

    # Step 4: Use the FAISS index to get similar images
    D, I = faiss_index.search(embedding, k=5)
    similar_images = [image_ids[idx] for idx in I[0]]

    return JSONResponse(content={
        "similar_index": similar_images
    })
    
    
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)