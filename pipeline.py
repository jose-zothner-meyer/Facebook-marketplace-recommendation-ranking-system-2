from data_cleaner import DataCleaner                  # Import the DataCleaner for cleaning the raw dataset.
from data_loader import ProductsDataset                 # Import the ProductsDataset class for loading dataset.
from dataset_processor import DatasetProcesser          # Import the DatasetProcesser for further dataset processing.
from image_processor import ImageProcessor              # Import the ImageProcessor for image transformations and processing.
from model import FaceBookSimSearchModel                # Import the FaceBookSimSearchModel for training and evaluation.
from file_handler import FileHandler                    # Import FileHandler for file operations such as saving CSV files.

class Pipeline:
    """
    Pipeline orchestrates the different stages of the project workflow including
    data cleaning, dataset processing, image processing, model training, and evaluation.
    
    Depending on the provided flags, it will sequentially:
      - Clean and process the raw dataset.
      - Process images (e.g., cleaning, resizing).
      - Train the model.
      - Evaluate the model.
    """
    
    def __init__(
        self, 
        process_data: bool = False,
        process_imgs: bool = False,
        train_model: bool = False,
        eval_model: bool = False
    ):
        """
        Initializes the Pipeline instance and triggers the corresponding processes based on flags.

        Args:
            process_data (bool): If True, performs raw data cleaning and processing.
            process_imgs (bool): If True, processes images using the image processing pipeline.
            train_model (bool): If True, trains the model on the processed data.
            eval_model (bool): If True, evaluates the model on the validation/test set.
        """
        # Create an instance of DataCleaner for the Products.csv file.
        self.data_cleaner = DataCleaner("Products.csv")
        
        # If the process_data flag is enabled, clean and process the dataset.
        if process_data:
            self.clean_dataset()
            
        # If the process_imgs flag is enabled, process images from the dataset.
        if process_imgs:
            self.process_imgs()
            
        # If the train_model flag is enabled, instantiate the model, generate training data,
        # train the model and finally close the TensorBoard writer.
        if train_model:
            self.model = FaceBookSimSearchModel()
            # Create data loaders and retrieve the training dataset loader (index 0 corresponds to train set).
            dataset = self.model.create_data_loaders()[0]
            self.model.train_model(dataset)
            self.model.close_writer()
            
        # If the eval_model flag is enabled, instantiate the model, generate validation data,
        # evaluate the model and finally close the TensorBoard writer.
        if eval_model:
            self.model = FaceBookSimSearchModel()
            # Create data loaders and retrieve the validation dataset loader (index 1 corresponds to validation set).
            dataset = self.model.create_data_loaders()[1]
            self.model.eval_model(dataset)
            self.model.close_writer()
        
    def clean_dataset(self):
        """
        Cleans and processes the raw dataset by:
          1. Dropping unnecessary columns.
          2. Casting data types for selected columns.
          3. Dropping rows with missing values.
          4. Stripping unwanted currency characters from data.
          5. Saving the cleaned DataFrame as a CSV file.
          6. Further processing the dataset using the DatasetProcesser.
          7. Saving the final training data to a CSV file.
        """
        # Drop columns not needed in the analysis.
        self.data_cleaner.drop_column()
        # Cast selected columns to the appropriate data types.
        self.data_cleaner.cast_data_types(['id', 'product_name', 'category', 'product_description', 'location'])
        # Remove rows with missing (NA) values.
        self.data_cleaner.drop_na()
        # Remove any currency characters from the dataset.
        self.data_cleaner.strip_currency_char()
        # Save the cleaned DataFrame to a CSV file.
        FileHandler.df_to_csv(self.data_cleaner.df, "cleaned_products.csv")
        
        # Initialize a DatasetProcesser to process the dataset further.
        processor = DatasetProcesser()
        # Execute the processing (e.g., splitting, formatting columns) on the dataset.
        processor.process()
        # Save the processed dataset as training data to a CSV file.
        FileHandler.df_to_csv(processor.df_pdt, "data/training_data.csv")
        
    def process_imgs(self):
        """
        Processes all images in the designated image folder.

        This method instantiates the ImageProcessor, which applies image transformations
        (such as resizing and padding), and saves the cleaned images to a specified folder.
        """
        # Instantiate the ImageProcessor.
        processor = ImageProcessor()
        # Process the images using the defined image processing pipeline.
        processor.process_image_data()


