import pandas as pd
import os
import logging

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_dataset(self):
        if os.path.exists(self.file_path):
            try:
                df = pd.read_csv(self.file_path)
                logging.info(f"Data loaded successfully. Shape: {df.shape}")
                return df
            except Exception as e:
                logging.error(f"Error loading the file: {e}")
                return pd.DataFrame()
        else:
            logging.error(f"File not found: {self.file_path}")
            return pd.DataFrame()

# Bu yerda to'g'ridan-to'g'ri ishga tushirilganda bajariladigan qism
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    file_path = r"C:\Users\Rasulbek907\Desktop\Project_MP\Data\Preprosessed\Preprosessed.csv"
    loader = DataLoader(file_path)
    data = loader.load_dataset()

    print(data.head())  # Ma'lumotlarning birinchi 5 qatorini ko'rish



import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class DataPreProcessing:
    def __init__(self, full_df: pd.DataFrame):
        self.df = full_df.copy()  # keep original safe
        self.encoders = {}        # store encoders for each column

    def handle_missing_values(self):
        """Drop rows/columns with too many NaNs and fill remaining ones."""
        logging.info("Handling missing values...")
        threshold = 0.5 * len(self.df)
        self.df = self.df.dropna(axis=1, thresh=threshold)
        self.df = self.df.fillna(0)
        return self.df

    def remove_duplicates(self):
        """Remove duplicate rows."""
        logging.info("Removing duplicate rows...")
        before = self.df.shape[0]
        self.df = self.df.drop_duplicates()
        after = self.df.shape[0]
        logging.info(f"Removed {before - after} duplicate rows.")
        return self.df

    def encode_categorical(self):
        """Apply sklearn LabelEncoder to categorical columns."""
        logging.info("Encoding categorical variables with LabelEncoder...")
        cat_cols = self.df.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.encoders[col] = le
            logging.info(f"Encided columns {col}")
        return self.df
    
    def scale_numeric(self):
        """Scale numeric columns between 0 and 1 using MinMaxScaler."""
        logging.info("Scaling numeric columns with MinMaxScaler...")
        num_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        if len(num_cols) > 0:
            self.scaler = MinMaxScaler()
            self.df[num_cols] = self.scaler.fit_transform(self.df[num_cols])
            logging.info(f"Scaled columns: {list(num_cols)}")
        else:
            logging.warning("No numeric columns found to scale.")
        return self.df

    def get_processed_data(self):
        """Return the final processed dataframe."""
        return self.df





