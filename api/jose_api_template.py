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
from pydantic import BaseModel
from resNet import FineTunedResNet
from torchvision import models
from torchvision.models import ResNet50_Weights
##############################################################
# TODO                                                       #
# Import your image processing script here                   #
import image_processor
# This script is purely to apply transforms onto raw images!!
##############################################################

try:
    model = FineTunedResNet(13)
    saved_weights = 'data/model_evaluation/model_20250211-192554/weights/epoch_5.pth'  # update if needed
    model.load_state_dict(torch.load(saved_weights, map_location="cpu"))
    model = torch.nn.Sequential(*list(model.combined_model.children())[:-1])
    pass
except:
    raise OSError("Error in creating Feature Extractor")

try:
    faiss_index = pickle.load(open("data/output/faiss_index.pkl", "rb"))
    image_embeddings = pickle.load(open("data/output/image_embeddings.pkl", "rb"))
    image_ids = pickle.load(open("data/output/image_ids.pkl", "rb"))
    pass
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")


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