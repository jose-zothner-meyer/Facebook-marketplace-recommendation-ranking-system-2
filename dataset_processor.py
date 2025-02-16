import pandas as pd
from typing import Dict, List
from file_handler import FileHandler
import os

class DatasetProcesser:
    """
    Processes a cleaned products dataset to assign numeric labels and integrate images data.
    
    Steps include:
      - Loading product and images CSV files.
      - Extracting the root category.
      - Creating encoder/decoder mappings.
      - Assigning numeric labels.
      - Merging images data.
      - Preparing and saving the final training dataset.
    """
    
    def __init__(
            self, 
            products_file: str = "cleaned_products.csv",
            images_file: str = "Images.csv", 
            output_file: str = "training_dataset.csv",
        ):
        """
        Initialize the ProductLabeler with file paths.
        
        Args:
            products_file (str): Path to the cleaned products CSV.
            images_file (str): Path to the images CSV.
            output_file (str): Path to save the processed training CSV.
        """
        self.products_file = products_file
        self.images_file = images_file
        self.output_file = output_file
        self.df_pdt = None       # DataFrame for products.
        self.df_images = None    # DataFrame for images.
        self.encoder = None      # Mapping: category string -> numeric label.
        self.decoder = None      # Mapping: numeric label -> category string.


    def _load_data(self) -> None:
        """Load product and images CSV files."""
        print("Loading data...")
        self.df_pdt = pd.read_csv(self.products_file, lineterminator='\n')
        print(self.df_pdt["category"])
        self.df_images = pd.read_csv(self.images_file, lineterminator='\n')
        print(f"Data loaded from {self.products_file} and {self.images_file}.")

    def _extract_root_category(self) -> None:
        """
        Extract the root category from the 'category' column.
        The root category is obtained by splitting the category string at '/'.
        """
        print("Extracting root categories...")
        self.df_pdt['root_category'] = self.df_pdt['category'].apply(lambda cat: cat.split("/")[0].strip())
        print("Root categories extracted.")

    def _create_encoder_decoder(self) -> None:
        """
        Create encoder and decoder mappings for root categories.
        """
        print("Creating encoder and decoder...")
        unique_categories: List[str] = self.df_pdt['root_category'].unique().tolist()
        self.encoder = {cat: idx for idx, cat in enumerate(unique_categories)}
        self.decoder = {idx: cat for cat, idx in self.encoder.items()}
        os.makedirs("data/output", exist_ok=True)
        FileHandler.pickle_obj(self.encoder, "data/output/image_encoder.pkl")
        FileHandler.pickle_obj(self.encoder, "data/output/image_decoder.pkl")
        print("Encoder and decoder created.")

    def _assign_labels(self) -> None:
        """
        Assign numeric labels to products based on the root category.
        """
        print("Assigning labels...")
        self.df_pdt['labels'] = self.df_pdt['root_category'].map(self.encoder)
        print("Labels assigned.")

    def _merge_images(self) -> None:
        """
        Merge images data into the products DataFrame.
        
        The 'id' column in images is renamed to 'Image' and merged on product ID.
        """
        print("Merging images data...")
        self.df_images.rename(columns={'id': 'Image'}, inplace=True)
        self.df_pdt = self.df_pdt.merge(self.df_images, left_on='id', right_on='Image', how='left', suffixes=('', '_img'))
        print("Images data merged.")

    def _get_X_y(self) -> None:
        """
        Prepare the final training dataset by selecting the image identifier and labels.
        """
        print("Preparing training data...")
        # Here, choose a column from images data as the image identifier.
        middle_column = self.df_images.columns[len(self.df_images.columns) // 2]
        self.df_pdt['Image'] = self.df_images[middle_column]
        self.df_pdt = self.df_pdt[['Image', 'labels']]
        print("Training data prepared.")

    def _save_data(self) -> None:
        """Save the processed training DataFrame to CSV."""
        print(f"Saving labeled data to {self.output_file}...")
        self.df_pdt.to_csv(self.output_file, index=False)
        print(f"Labeled data saved to {self.output_file}.")

    def process(self) -> None:
        """
        Execute the full processing pipeline.
        """
        self._load_data()
        self._extract_root_category()
        self._create_encoder_decoder()
        self._assign_labels()
        self._merge_images()
        self._get_X_y()
        self._save_data()