import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.data_splitter import DataSplitter
from utils.base_script import BaseScript

class DataSplitRunner(BaseScript):
    TRAIN_SIZE = 0.7
    RANDOM_STATE = 1022

    def __init__(self):
        super().__init__()
        self.data_dir = project_root / "data" / "processed"
        self.output_dir = project_root / "data" / "processed"

    def run(self):
        self.logger.info("=== SCRIPT DE PARTICIÓN DE DATASETS ===")

        try:
            # Inicializar el splitter
            splitter = DataSplitter(output_dir=str(self.output_dir))

            # Procesar dataset con Moodle
            train_moodle, test_moodle = splitter.process_dataset(
                file_path=str(self.data_dir / "full_short_dataset_moodle_clean.csv"),
                dataset_name="moodle",
                train_size=self.TRAIN_SIZE,
                random_state=self.RANDOM_STATE
            )

            # Procesar dataset sin Moodle
            train_no_moodle, test_no_moodle = splitter.process_dataset(
                file_path=str(self.data_dir / "full_short_dataset_no_moodle_clean.csv"),
                dataset_name="no_moodle",
                train_size=self.TRAIN_SIZE,
                random_state=self.RANDOM_STATE
            )

            # Resumen final
            self.logger.info("=== PROCESO COMPLETADO EXITOSAMENTE ===")
            self.logger.info(f"Dataset Moodle - Train: {train_moodle.shape}, Test: {test_moodle.shape}")
            self.logger.info(f"Dataset No-Moodle - Train: {train_no_moodle.shape}, Test: {test_no_moodle.shape}")

        except Exception as e:
            self.logger.error(f"Error durante la ejecución: {str(e)}")
            raise e

def main():
    runner = DataSplitRunner()
    return runner.run()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
