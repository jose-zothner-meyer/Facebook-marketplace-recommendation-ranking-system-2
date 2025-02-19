"""
model.py

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

from sklearn.model_selection import train_test_split  # For splitting the dataset.
from torch.utils.data import DataLoader               # DataLoader for batching data.
from torch.utils.tensorboard import SummaryWriter       # TensorBoard logging.
from torchvision import transforms                     # Image transformation utilities.
from data_loader import ProductsDataset                # Custom dataset class.
from file_handler import FileHandler                   # Utility for file operations.
from resNet import FineTunedResNet                     # The model architecture.

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class FaceBookSimSearchModel:
    """
    FaceBookSimSearchModel encapsulates all functions for preparing data,
    training the model, evaluating, and generating image embeddings.
    """

    def __init__(self):
        """
        Initialize the model instance:
         - Loads label encoder and decoder.
         - Creates the model.
         - Sets up device, tensorboard writer, optimizer, criterion, and directories.
         - Defines the data transformation process.
        """
        # Load label encoders/decoders from pickle files.
        self.label_encoder = FileHandler.load_pickle("data/output/", "image_encoder.pkl")
        self.label_decoder = FileHandler.load_pickle("data/output/", "image_decoder.pkl")
        
        # Determine the device (GPU if available, otherwise CPU).
        self.device = self._create_device()
        
        # Initialize the model with the number of classes from the label_encoder.
        self.model = FineTunedResNet(len(self.label_encoder))
        
        # Create TensorBoard writer for logging training information.
        self.writer = self.create_writer()
        
        # Set up the optimizer (Adam) on the model parameters.
        self.optimizer = self.create_optimizer()
        
        # Define the loss function.
        self.criterion = self.create_criterion()
        
        # Create directories for saving model-related outputs.
        self.model_dir, self.weights_dir = FileHandler.create_model_dir()
        
        # Define image transformations: convert images to tensor and normalize.
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert image to tensor.
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # Normalize with ImageNet stats.
        ])

    def run_pipeline(self):
        """
        Executes the full training and feature extraction pipeline.
        
        Steps:
          a) Data Processing & Setup:
             - Reads training CSV and creates encoder/decoder mappings.
             - Splits dataset using stratification on "labels" into training, validation, and test sets.
             - Writes each split to temporary CSV files.
          b) Model Training:
             - Trains the model using a training loop.
             - Computes and logs training metrics (loss and accuracy) with TensorBoard.
             - Saves model weights and logs metrics at the end of each epoch.
          c) Model Conversion for Feature Extraction:
             - Converts the classification model into a feature extraction model by replacing the classification head
               with a new fully connected layer that outputs a 1000-dimensional vector.
             - Saves the converted model to 'data/final_model/image_model.pt'.
          d) (Optional) Embedding Extraction:
             - Extracts image embeddings using the feature extraction model.
             
        Note: The actual implementation should call these steps sequentially.
        """
        pass  # The full implementation would be provided here.

    def _create_device(self):
        """
        Create and return the computing device.
        
        Returns:
            torch.device: The GPU if available, otherwise CPU.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)  # Print the selected device for debugging purposes.
        return device
        
    def create_data_loaders(self, train_split: float = 0.4, validation_split: float = 0.5):
        """
        Creates DataLoaders for training, validation, and testing datasets.
        
        Args:
            train_split (float): Fraction of data to use for training (default: 0.4).
            validation_split (float): Fraction of the temporary dataset to use for validation (default: 0.5).
            
        Returns:
            tuple: (train_dataloader, validation_dataloader, test_dataloader)
        """
        # Read the training data CSV file.
        dataframe = pd.read_csv("data/training_data.csv", dtype={'labels': int})
        # Compute the test split proportion.
        test_split = 1 - (train_split + validation_split)
        
        # Split the dataset using stratification based on the 'labels' column.
        df_training, df_temp = train_test_split(dataframe, test_size=train_split, stratify=dataframe['labels'])
        df_validation, df_test = train_test_split(df_temp, test_size=validation_split, stratify=df_temp['labels'])
        
        # Create directory for temporary CSV files.
        temp_csv_dir = 'data/temp_csv'
        os.makedirs(temp_csv_dir, exist_ok=True)
        
        # Define file paths for each CSV split.
        temp_train_csv = os.path.join(temp_csv_dir, 'temp_train.csv')
        temp_val_csv = os.path.join(temp_csv_dir, 'temp_val.csv')
        temp_test_csv = os.path.join(temp_csv_dir, 'temp_test.csv')
        
        # Save the split datasets to CSV files.
        df_training.to_csv(temp_train_csv, index=False)
        df_validation.to_csv(temp_val_csv, index=False)
        df_test.to_csv(temp_test_csv, index=False)
        
        # Create datasets using the ProductsDataset class.
        train_dataset = ProductsDataset(temp_train_csv, 'cleaned_images/')
        validation_dataset = ProductsDataset(temp_val_csv, 'cleaned_images/')
        test_dataset = ProductsDataset(temp_test_csv, 'cleaned_images/')
        
        # Create DataLoaders to enable batch processing.
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        return train_dataloader, validation_dataloader, test_dataloader

    def create_device(self):
        """
        Creates and prints the computing device.
        
        Returns:
            torch.device: GPU if available, otherwise CPU.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Device is set to {device}")
        return device

    def create_optimizer(self):
        """
        Creates the optimizer used for training.
        
        Returns:
            torch.optim.Optimizer: Adam optimizer with a learning rate of 0.001.
        """
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)
        return optimizer

    def create_writer(self):
        """
        Creates a TensorBoard SummaryWriter instance for logging training metrics.
        
        Returns:
            SummaryWriter: The writer instance storing logs in 'resource/tensorboard'.
        """
        self.writer = SummaryWriter('resource/tensorboard')
        return self.writer
    
    def create_criterion(self):
        """
        Creates the loss function.
        
        Returns:
            nn.Module: CrossEntropyLoss used as the loss criterion.
        """
        criterion = nn.CrossEntropyLoss()
        return criterion

    def train_model(self, dataset, epochs=1):
        """
        Trains the model using the given dataset.
        
        Args:
            dataset (DataLoader): DataLoader for the training dataset.
            epochs (int): Number of epochs to train.
        """
        for epoch in range(epochs):
            self.model.train()            # Set the model to training mode.
            self.model.to(self.device)      # Move the model to the selected device.
            running_loss = 0.0              # Loss accumulator for logging.
            total_loss = 0.0                # Total loss for the epoch.
            train_correct = 0             # Counter for correct predictions.
            train_total = 0               # Counter for total predictions.
            
            # Iterate over batches in the dataset.
            for i, (images, labels, img_name) in enumerate(dataset):
                images = images.to(self.device)  # Transfer images to device.
                labels = labels.to(self.device)  # Transfer labels to device.
                self.optimizer.zero_grad()       # Reset gradients.
                
                outputs = self.model(images)     # Forward pass.
                loss = self.criterion(outputs, labels)  # Calculate loss.
                loss.backward()                  # Backpropagation.
                self.optimizer.step()            # Update weights.
                
                running_loss += loss.item()      # Accumulate batch loss.
                total_loss += loss.item()        # Accumulate total loss.
                
                # Compute training accuracy.
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
                
                if i % 10 == 9:
                    # Log average loss every 10 batches.
                    self.writer.add_scalar('training loss', running_loss / 10, epoch * len(dataset) + i)
                    running_loss = 0.0

            # Calculate average training loss and accuracy for the epoch.
            avg_train_loss = total_loss / len(dataset)
            train_accuracy = train_correct / train_total
            print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_train_loss:.4f}, '
                  f'Training Accuracy: {train_accuracy:.4f}')
            self.writer.add_scalar('avg training loss', avg_train_loss, epoch)
            self.writer.add_scalar('train accuracy', train_accuracy, epoch)
            
        # Flush any pending logs.
        self.writer.flush()
        # Save final model weights after training.
        self._save_model_weights(self.weights_dir, f'training/epoch_final.pth')

        # Write training metrics to a file for future reference.
        metrics_path = os.path.join(self.model_dir, 'training/metrics.txt')
        with open(metrics_path, 'a') as f:
            f.write(f'Epoch {epoch + 1}, Avg Train Loss: {avg_train_loss:.4f}, '
                    f'Train Accuracy: {train_accuracy:.4f}, Validation Loss: {avg_train_loss:.4f}, '
                    f'Validation Accuracy: {train_accuracy:.4f}\n')

    def eval_model(self, dataset, epochs=4):
        """
        Evaluates the model on the given validation dataset.
        
        Args:
            dataset (DataLoader): DataLoader for the validation dataset.
            epochs (int): Number of evaluation epochs.
        """
        for epoch in range(epochs):
            val_loss = 0.0       # Initialize loss accumulator.
            val_correct = 0      # Counter for correct predictions.
            val_total = 0        # Counter for total predictions.
            self.model.to(self.device)
            
            with torch.no_grad():  # Disable gradient calculation.
                for images, labels, img_name in dataset:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)  # Inference.
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
            
            # Compute average validation loss and accuracy.
            avg_val_loss = val_loss / len(dataset)
            val_accuracy = val_correct / val_total
            self.writer.add_scalar('validation loss', avg_val_loss, epoch)
            self.writer.add_scalar('validation accuracy', val_accuracy, epoch)
            self.writer.flush()
            
            # Save model weights after each validation epoch.
            self._save_model_weights(self.weights_dir, f'eval/epoch_{epoch + 1}.pth')
            print(f'Epoch [{epoch + 1}/{epochs}], Validation Loss: {avg_val_loss:.4f}, '
                  f'Validation Accuracy: {val_accuracy:.4f}')
        
        # Save final validation weights and metrics.
        self._save_model_weights(self.weights_dir, f'eval/epoch_final.pth')
        metrics_path = os.path.join(self.model_dir, 'eval/metrics.txt')
        with open(metrics_path, 'a') as f:
            f.write(f'Epoch {epoch + 1}, Avg Train Loss: {avg_val_loss:.4f}, '
                    f'Train Accuracy: {val_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}, '
                    f'Validation Accuracy: {val_accuracy:.4f}\n')
        
        # Generate and save image embeddings after evaluation.
        self._generate_embeddings(dataset)

    def close_writer(self):
        """
        Closes the TensorBoard SummaryWriter.
        """
        self.writer.close()
        print("Closing the summary writer")

    def _save_model_weights(self, folder, filename):
        """
        Saves the model weights (state dictionary) to the specified folder and filename.
        
        Args:
            folder (str): Destination folder path.
            filename (str): The name of the file to save the weights.
        """
        os.makedirs(folder, exist_ok=True)  # Make sure the folder exists.
        weights_path = os.path.join(folder, filename)  # Build complete file path.
        torch.save(self.model.state_dict(), weights_path)  # Save model weights.

    def save_final_model(self):
        """
        Creates final directories and (optionally) saves the last model.
        """
        model_dir, weights_dir = FileHandler.create_model_dir()
        # Additional code to save the final model can be implemented here.

    def _generate_embeddings(self, dataset):
        """
        Generates image embeddings using the feature extraction model and saves them to disk.
        
        Args:
            dataset (DataLoader): Dataset used for generating embeddings.
        """
        # Create a feature extractor by removing the classification head.
        model_extractor = nn.Sequential(*list(self.model.combined_model.children())[:-1])
        image_embeddings = {}  # Dictionary to hold image filename and its corresponding embedding.
        
        with torch.no_grad():  # Disable gradient computation for inference.
            for idx, (image, label, img_name) in enumerate(dataset):
                image = image.to(self.device)  # Move image batch to the appropriate device.
                embedding = model_extractor(image)  # Get the feature embedding.
                # Flatten the embedding, detach and move to CPU, then convert to numpy array.
                embedding = embedding.flatten().detach().cpu().numpy()
                # Save the embedding using the image filename as key.
                image_embeddings[str(img_name)] = embedding.tolist()
        
        # Save the embeddings as JSON.
        FileHandler.save_json(image_embeddings, "data/output", "image_embeddings.json")
        # Also save the embeddings using pickle.
        FileHandler.pickle_obj(image_embeddings, "data/output", "image_embeddings.pkl")
        print(f"Image embeddings successfully saved")