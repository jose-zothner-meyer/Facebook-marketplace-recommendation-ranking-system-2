from data_cleaner import DataCleaner
from data_loader import ProductsDataset
from dataset_processor import DatasetProcesser
from image_processor import ImageProcessor
from data_loader import ProductsDataset
from model import FaceBookSimSearchModel
from file_handler import FileHandler


class Pipeline:
    
    def __init__(
        self, 
        process_data: bool = False,
        process_imgs: bool = False,
        train_model: bool = False,
        eval_model: bool = False
    ):
        self.data_cleaner = DataCleaner("Products.csv")
        if process_data:
            self.clean_dataset()        
        if process_imgs:
            self.process_imgs()
        if train_model:
            self.model = FaceBookSimSearchModel()
            dataset = self.model.create_data_loaders()[0]
            self.model.train_model(dataset)
            self.model.close_writer()
        if eval_model:
            self.model = FaceBookSimSearchModel()
            dataset = self.model.create_data_loaders()[1]
            self.model.eval_model(dataset)
            self.model.close_writer()
        
    def clean_dataset(self):
        self.data_cleaner.drop_column()
        self.data_cleaner.cast_data_types(['id', 'product_name', 'category', 'product_description', 'location'])
        self.data_cleaner.drop_na()
        self.data_cleaner.strip_currency_char()
        FileHandler.df_to_csv(self.data_cleaner.df, "cleaned_products.csv")
        processor = DatasetProcesser()
        processor.process()
        FileHandler.df_to_csv(processor.df_pdt, "data/training_data.csv")
        

    def process_imgs(self):
        processor = ImageProcessor()
        processor.process_image_data()


