from time import time
import pickle
import os
import json
import pandas as pd


class FileHandler:

    @staticmethod
    def create_model_dir(self, base_dir='data/model_evaluation'):
        """
        Create a timestamped directory (with a subfolder for weights) to store model outputs.
        """
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        model_dir = os.path.join(base_dir, f'model_{timestamp}')
        weights_dir = os.path.join(model_dir, 'weights')
        os.makedirs(weights_dir, exist_ok=True)
        return model_dir, weights_dir

    @staticmethod
    def pickle_obj(obj, fp, mode: str = "wb"):
        with open(fp, mode) as f:
            pickle.dump(obj, f)
        print(f"The obj {obj} was pickled and saved to {fp}")

    @staticmethod
    def load_pickle(filepath, filename, mode: str = "rb"):
        with open(filepath + filename, mode) as f:
            pkl_obj = pickle.load(f)
        print(f"The obj {filename} was unpickled")
        return pkl_obj

    @staticmethod
    def save_json(obj, filepath, filename, mode: str = "w"):
        os.makedirs(filepath, exist_ok=True)
        with open(filepath + filename, mode) as f:
            json.dump(obj, f)
        print(f"The obj {obj} was pickled and saved to {filepath}")

    @staticmethod
    def df_to_csv(df: pd.DataFrame, fp):
        df.to_csv(fp)
        print(f"Saving DataFrame to filepath {fp}")