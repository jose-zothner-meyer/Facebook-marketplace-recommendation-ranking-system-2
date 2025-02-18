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
##################################################################
# TODO                                                           #
# 1. Load faiss index.pkl
# 2. Load image embedding ids.pkl OR json file
##################################################################
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
    pil_image = Image.open(image.file)
    
    #####################################################################
    # TODO                                                              #
    # 1. Take a raw image
    # 2. apply necessary transforms onto it
    # 3. Put transformed image through the FeatureExtractor. That will get you the embeddings.
    # 4. Use these embeddings with your faiss code to get the similar images          
    #####################################################################

    return JSONResponse(content={
    "similar_index": "", # Return the index of similar images here
        })
    
    
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)