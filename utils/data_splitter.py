import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataSplitter:
    def __init__(self, output_dir="data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(self, file_path):
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Dataset cargado exitosamente. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error al cargar {file_path}: {str(e)}")
            raise

    def split_dataset(self, df, train_size=0.7, random_state=1022):
        test_size = 1.0 - train_size
        logger.info(f"Iniciando partición del dataset. Shape original: {df.shape}")
        logger.info(f"Proporción: {train_size:.1%} entrenamiento, {test_size:.1%} prueba")

        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state
        )

        logger.info(f"Particiones creadas - Train: {train_df.shape}, Test: {test_df.shape}")
        return train_df, test_df

    def save_datasets(self, train_df, test_df, dataset_name):
        train_file = self.output_dir / f"train_{dataset_name}.csv"
        test_file = self.output_dir / f"test_{dataset_name}.csv"

        try:
            train_df.to_csv(train_file, index=False)
            test_df.to_csv(test_file, index=False)
        except Exception as e:
            logger.error(f"Error al guardar datasets: {str(e)}")
            raise

    def process_dataset(self, file_path, dataset_name, train_size=0.7, random_state=1022):
        # Cargar dataset
        df = self.load_dataset(file_path)
        # Dividir dataset
        train_df, test_df = self.split_dataset(df, train_size=train_size, random_state=random_state)
        # Guardar datasets
        self.save_datasets(train_df, test_df, dataset_name)
        return train_df, test_df
