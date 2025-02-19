from time import time, strftime  # Import time functions (time and strftime) for timestamp related operations.
import pickle                   # Import pickle for object serialization.
import os                       # Import os for file and directory operations.
import json                     # Import json for saving objects in JSON format.
import pandas as pd             # Import pandas for DataFrame operations.
from datetime import datetime   # Import datetime to get the current date and time.


class FileHandler:
    """
    A utility class for handling file operations including saving/loading pickled objects,
    saving data in JSON format, and converting DataFrames to CSV files.
    """

    @staticmethod
    def create_model_dir(base_dir='data/model_evaluation'):
        """
        Create a timestamped directory (with a subfolder for weights) to store model outputs.
        
        Args:
            base_dir (str): Base directory path where model evaluation folders will be created.
        
        Returns:
            tuple: A tuple containing the path to the model directory and the weights subdirectory.
        """
        # Generate a timestamp string formatted as YYYYMMDD-HHMMSS.
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        # Concatenate the base directory with a unique folder name using the timestamp.
        model_dir = os.path.join(base_dir, f'model_{timestamp}')
        # Create a subdirectory for storing model weights.
        weights_dir = os.path.join(model_dir, 'weights')
        # Create the weights directory, including any intermediate directories if they don't exist.
        os.makedirs(weights_dir, exist_ok=True)
        # Return the paths of the created directories.
        return model_dir, weights_dir

    @staticmethod
    def pickle_obj(obj, fp, mode: str = "wb"):
        """
        Save an object to a file using pickle serialization.
        
        Args:
            obj: The object to pickle (serialize).
            fp (str): The file path where the object will be saved.
            mode (str): The file mode (default is "wb" for writing in binary mode).
        """
        # Open the file at the given file path in the specified mode.
        with open(fp, mode) as f:
            # Dump the object into the file using pickle.
            pickle.dump(obj, f)
        # Inform the user that the object was successfully pickled.
        print(f"The obj {obj} was pickled and saved to {fp}")

    @staticmethod
    def load_pickle(filepath, filename, mode: str = "rb"):
        """
        Load a pickled object from a file.
        
        Args:
            filepath (str): The directory path where the file is located.
            filename (str): The name of the file to load.
            mode (str): The file mode (default is "rb" for reading in binary mode).
        
        Returns:
            The unpickled Python object.
        """
        # Open the combined file path (directory + filename) in the specified mode.
        with open(filepath + filename, mode) as f:
            # Load the object from the file.
            pkl_obj = pickle.load(f)
        # Inform the user that the object was successfully unpickled.
        print(f"The obj {filename} was unpickled")
        # Return the loaded object.
        return pkl_obj

    @staticmethod
    def save_json(obj, filepath, filename, mode: str = "w"):
        """
        Save an object as a JSON file.
        
        Args:
            obj: The object to be saved (should be JSON serializable).
            filepath (str): The directory where the file should be saved.
            filename (str): The name of the JSON file.
            mode (str): File mode (default is "w" for writing).
        """
        # Ensure the directory exists; if not, create it.
        os.makedirs(filepath, exist_ok=True)
        # Open the file in the specified mode and save the object in JSON format.
        with open(filepath + "/" + filename, mode) as f:
            json.dump(obj, f)
        # Inform the user that the object was successfully saved as JSON.
        print(f"The obj {obj} was pickled and saved to {filepath}")

    @staticmethod
    def df_to_csv(df: pd.DataFrame, fp):
        """
        Save a pandas DataFrame to a CSV file.
        
        Args:
            df (pd.DataFrame): The DataFrame to be saved.
            fp (str): The file path where the CSV file will be saved.
        """
        # Save the DataFrame to CSV using the provided file path.
        df.to_csv(fp)
        # Inform the user that the DataFrame was successfully saved.
        print(f"Saving DataFrame to filepath {fp}")