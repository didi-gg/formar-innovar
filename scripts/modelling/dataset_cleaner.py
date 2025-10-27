import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.base_script import BaseScript
from utils.eda_analysis_base import EDAAnalysisBase


class DatasetCleaner(BaseScript):
    def __init__(self):
        super().__init__()

        # Rutas de los archivos de entrada
        self.input_paths = {
            'moodle': 'data/processed/full_short_dataset_moodle.csv',
            'no_moodle': 'data/processed/full_short_dataset_no_moodle.csv'
        }

        # Rutas de los archivos de salida
        self.output_paths = {
            'moodle': 'data/processed/full_short_dataset_moodle_clean.csv',
            'no_moodle': 'data/processed/full_short_dataset_no_moodle_clean.csv'
        }

        # Obtener características a excluir de EDAAnalysisBase
        self.excluded_features = set(EDAAnalysisBase.EXCLUDED_FEATURES)
        self.excluded_features.update(EDAAnalysisBase.EXCLUDED_FEATURES_NULL_VARIANCE)
        # Agregar variables objetivo (son cadenas, no listas)
        self.excluded_features.add(EDAAnalysisBase.TARGET_CATEGORICAL)

    def validate_input_files(self):
        missing_files = []

        for dataset_type, path in self.input_paths.items():
            if not os.path.exists(path):
                missing_files.append(path)

        if missing_files:
            raise FileNotFoundError(f"Archivos de entrada faltantes: {missing_files}")
        self.logger.info("Todos los archivos de entrada encontrados correctamente")

    def load_and_clean_dataset(self, input_path: str) -> pd.DataFrame:
        self.logger.info(f"Cargando dataset desde: {input_path}")

        # Cargar dataset
        df = pd.read_csv(input_path)
        original_shape = df.shape
        self.logger.info(f"Dataset original: {original_shape[0]} filas, {original_shape[1]} columnas")

        # Identificar columnas a eliminar que existen en el dataset
        columns_to_drop = []
        for col in self.excluded_features:
            if col in df.columns:
                columns_to_drop.append(col)

        # Eliminar columnas excluidas
        if columns_to_drop:
            df_clean = df.drop(columns=columns_to_drop)
            self.logger.info(f"Eliminadas {len(columns_to_drop)} columnas: {sorted(columns_to_drop)}")
        else:
            df_clean = df.copy()
            self.logger.info("No se encontraron columnas a eliminar")

        final_shape = df_clean.shape
        self.logger.info(f"Dataset limpio: {final_shape[0]} filas, {final_shape[1]} columnas")
        self.logger.info(f"Reducción de columnas: {original_shape[1] - final_shape[1]} columnas eliminadas")

        return df_clean

    def process_dataset(self, dataset_type: str):
        input_path = self.input_paths[dataset_type]
        output_path = self.output_paths[dataset_type]

        # Cargar y limpiar dataset
        df_clean = self.load_and_clean_dataset(input_path)

        # Guardar dataset limpio
        self.save_to_csv(df_clean, output_path)

    def run_cleaning(self):
        try:
            # Validar archivos de entrada
            self.validate_input_files()

            # Procesar cada dataset
            for dataset_type in ['moodle', 'no_moodle']:
                self.process_dataset(dataset_type)

        except Exception as e:
            self.logger.error(f"Error durante la limpieza: {str(e)}")
            raise

def main():
    cleaner = DatasetCleaner()

    try:
        # Ejecutar limpieza
        cleaner.run_cleaning()

    except Exception as e:
        cleaner.logger.error(f"Error en el proceso principal: {str(e)}")
        raise e
    finally:
        cleaner.close()


if __name__ == "__main__":
    main()
