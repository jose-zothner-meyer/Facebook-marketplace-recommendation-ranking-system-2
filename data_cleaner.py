import pandas as pd


class DataCleaner:

    def __init__(self, raw_data_fp: str, line_terminator: str = "\n"):
        self.line_terminator = line_terminator
        self.df = self.load_dataframe(data_fp=raw_data_fp)
        """
        Perform data cleaning on the specified CSV file.
        
        Steps:
          1. Read CSV using the specified line terminator.
          2. Drop the 'Unnamed: 0' column if it exists.
          3. Drop rows with missing values.
          4. Convert selected columns to string type.
          5. Clean the 'price' column: remove '£' symbols and commas, convert to numeric.
          6. Drop rows where 'price' conversion fails (NaN).
          7. Return the cleaned DataFrame.
        
        :param input_file: Path to the CSV file to clean.
        :param line_terminator: Line terminator used in the CSV.
        :return: A cleaned pandas DataFrame.
        """
        # 1. Read the CSV file.
    def load_dataframe(self, data_fp):
        """
        The function `load_dataframe` reads a CSV file into a pandas DataFrame using a specified line
        terminator.
        
        Args:
          data_fp: This parameter should be a string representing the file path 
          where the CSV file is located on your system.
        
        Returns:
          A pandas DataFrame is being returned.
        """
        df = pd.read_csv(data_fp, lineterminator=self.line_terminator)
        return df
    
    def drop_column(self, column: str = "Unamed: 0"):
        """
        The function `drop_column` removes a specified column from a DataFrame if it exists.
        
        Args:
          column (str): parameter  which specifies the name of 
          the column to be dropped from the DataFrame.
        """
        # 2. Drop the 'Unnamed: 0' column if it exists.
        if column in self.df.columns:
            self.df.drop(column, axis=1, inplace=True)

    def drop_na(self):
        """
        Drops rows with missing values from a DataFrame.
        """
        # 3. Drop rows with missing values.
        self.df.dropna(inplace=True)

    def cast_data_types(self, column_list: list, data_type: str = "string"):
        """
        The function `cast_data_types` converts selected columns in a DataFrame to a specified data
        type.
        
        Args:
          column_list (list): a list of column names that you want to
            convert to a specific data type.
          data_type (str): The data type to which the selected columns should be converted. 
        """
        # 4. Convert selected columns to string.
        # ['id', 'product_name', 'category', 'product_description', 'location']
        for col in column_list:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(data_type)

    def strip_currency_char(self, price_column: str = "price", sign: str = "£"):
        """
        The function `clean_prices` removes special characters and converts the 'price' column to
        numeric format in a DataFrame.
        
        Args:
            price_column (str): The name of the column in the DataFrame 
            that contains the prices to be cleaned. 
        """
        # 5. Clean the 'price' column.
        if price_column in self.df.columns:
            self.df['price'] = self.df['price'].str.replace(sign, '', regex=False)  # Remove '£'.
            self.df['price'] = self.df['price'].str.replace(',', '', regex=False)  # Remove commas.
            self.df['price'] = self.df['price'].astype(str)  # Ensure it's a string.
            self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')  # Convert to numeric.
            # 6. Drop rows where price conversion failed.
            self.df.dropna(subset=['price'], inplace=True)
