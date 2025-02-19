"""
pipeline.py

This file contains the integrated training and feature extraction pipeline.
It combines data processing, model training (with additional accuracy metrics), model conversion,
and (optionally) image embedding extraction into one function: run_pipeline().

Key changes:
  - The training loop saves the model weights (and metrics) at the end of every epoch.
  - A timestamped folder is created under data/model_evaluation with a subfolder "weights".
  - After training, the classification model is converted to a feature extraction model by
    replacing its classification head with a new fully connected layer with 1000 neurons.
  - The final model weights are saved to data/final_model/image_model.pt.
"""

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from data_loader import ProductsDataset
from file_handler import FileHandler
from resNet import FineTunedResNet


import os
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim


# Import the teacher's model and our dataset class.


class FaceBookSimSearchModel:


    def __init__(self):
        self.label_encoder = FileHandler.load_pickle("data/output/", "image_encoder.pkl")
        self.label_decoder = FileHandler.load_pickle("data/output/", "image_decoder.pkl")
        self.model = FineTunedResNet(len(self.label_encoder))
        self.device = self._create_device()
        self.writer = self.create_writer()
        self.optimizer = self.create_optimizer()
        self.criterion = self.create_criterion()
        # NOTE not sure where this is to be used left it in anyways. I think you do this in your image cleaning?
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
            ])


    def run_pipeline(self):
        """
        Executes the full training and feature extraction pipeline.
        
        Steps:
        a) Data Processing & Setup:
            - Reads the training CSV (with columns "Image" and "labels").
            - Creates encoder/decoder mappings.
            - Splits the dataset (using stratification on "labels") into training, validation, and test sets.
            - Writes each split to temporary CSV files.
        b) Model Training:
            - Instantiates the teacher’s FineTunedResNet model.
            - Trains the model using a standard training loop while computing and logging additional metrics (accuracy).
            - At the end of each epoch, saves model weights and appends training/validation metrics to a text file.
        c) Model Conversion for Feature Extraction:
            - Converts the classification model into a feature extraction model by replacing the classification head
            with a new fully connected layer that outputs a 1000-dimensional vector.
            - Saves the converted model to 'data/final_model/image_model.pt'.
        d) (Optional) Embedding Extraction:
            - Once you need to extract image embeddings, you can use the feature extraction model.
        """

    def _create_device(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        return device
        
    def create_data_loaders(
        self, 
        train_split: float = 0.4, 
        validation_split: float = 0.5,   
    ):
        dataframe = pd.read_csv("data/training_data.csv", dtype={'labels': int})
        test_split = 1 - (train_split + validation_split) 
        # Split the DataFrame using stratification on "labels"
        df_training, df_temp = train_test_split(dataframe, test_size=train_split, stratify=dataframe['labels'])
        df_validation, df_test = train_test_split(df_temp, test_size=validation_split, stratify=df_temp['labels'])

        # Create a folder to store temporary CSV files
        temp_csv_dir = 'data/temp_csv'
        os.makedirs(temp_csv_dir, exist_ok=True)

        # Write temporary CSV files (ImageDataset expects a CSV file path)
        temp_train_csv = os.path.join(temp_csv_dir, 'temp_train.csv')
        temp_val_csv = os.path.join(temp_csv_dir, 'temp_val.csv')
        temp_test_csv = os.path.join(temp_csv_dir, 'temp_test.csv')
        df_training.to_csv(temp_train_csv, index=False)
        df_validation.to_csv(temp_val_csv, index=False)
        df_test.to_csv(temp_test_csv, index=False)

        # Create datasets from the temporary CSV files.
        train_dataset = ProductsDataset(temp_train_csv, 'cleaned_images/')
        validation_dataset = ProductsDataset(temp_val_csv, 'cleaned_images/')
        test_dataset = ProductsDataset(temp_test_csv, 'cleaned_images/')

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        return train_dataloader, validation_dataloader, test_dataloader

        # ---------------------------
        # b) Model Training (Task 5)
        # ---------------------------
    def create_device(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Device is set to {device}")
        self.model.to(device)
        return device

    def create_optimizer(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)
        return optimizer

    def create_writer(self):
        self.writer = SummaryWriter('resource/tensorboard')
        return self.writer
    
    def create_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def train_model(self, dataset, epochs=1):
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            total_loss = 0.0
            train_correct = 0
            train_total = 0
            for i, (images, labels, img_name) in enumerate(dataset):
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                total_loss += loss.item()

                # Calculate training accuracy.
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)

                if i % 10 == 9:
                    self.writer.add_scalar('training loss', running_loss / 10, epoch * len(dataset) + i)
                    running_loss = 0.0
                

                avg_train_loss = total_loss / len(dataset)
                train_accuracy = train_correct / train_total
                print(f'Epoch [{epoch + 1}/{epochs}], Validation Loss: {avg_train_loss:.4f}, '
                f'Validation Accuracy: {train_accuracy:.4f}')
                self.writer.add_scalar('avg training loss', avg_train_loss, epoch)
                self.writer.add_scalar('train accuracy', train_accuracy, epoch)
            
        self.writer.flush()
        self._save_model_weights("data/output/training/weights", f'epoch_{epoch + 1}.pth')

    def eval_model(self, dataset, epochs=4):
        # Validation phase: compute loss and accuracy.
        for epoch in range(epochs):
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels, img_name in dataset:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
            avg_val_loss = val_loss / len(dataset)
            val_accuracy = val_correct / val_total
            self.writer.add_scalar('validation loss', avg_val_loss, epoch)
            self.writer.add_scalar('validation accuracy', val_accuracy, epoch)
            self.writer.flush()
            self._save_model_weights("data\\eval\\weights", f'epoch_{epoch + 1}.pth')
            print(f'Epoch [{epoch + 1}/{epochs}], Validation Loss: {avg_val_loss:.4f}, '
                f'Validation Accuracy: {val_accuracy:.4f}')
        self._save_model_weights("data/output/training/weights", f'epoch_{epoch + 1}.pth')
        self._generate_embeddings(dataset)

    def close_writer(self):
        self.writer.close()
        print("Closing the summary writer")

    def _save_model_weights(self, folder, filename):
        # Save model weights for the epoch.
        os.makedirs(folder, exist_ok=True)
        weights_path = os.path.join(folder, filename)
        torch.save(self.model.state_dict(), weights_path)

    def _generate_embeddings(self, dataset):
        model_extractor = nn.Sequential(*list(self.model.combined_model.children())[:-1])
        image_embeddings = {}
        with torch.no_grad():  # Disable gradient computation for inference
            for idx, (image, label, img_name) in enumerate(dataset):
                image = image.to(self.device)  # Move image tensor to the correct device

                # Extract feature embedding using the modified model
                embedding = model_extractor(image)
                embedding = embedding.flatten().detach().cpu().numpy()  # Convert tensor to NumPy array

                # Store the embedding using the image filename as the key
                image_embeddings[str(img_name)] = embedding.tolist()
        FileHandler.save_json(image_embeddings, "data/output", "image_embeddings.json")
        FileHandler.pickle_obj(image_embeddings, "data/output", "image_embeddings.pkl")
        print("Image embeddings successfully saved")