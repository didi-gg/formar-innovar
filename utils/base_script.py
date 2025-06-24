import duckdb
import logging
import pandas as pd


class BaseScript:
    def __init__(self):
        self._setup_logging()
        self.con = duckdb.connect()
        self.logger = logging.getLogger(self.__class__.__name__)

    def _setup_logging(self):
        root_logger = logging.getLogger()
        if not root_logger.hasHandlers():
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[logging.StreamHandler()],
            )

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, "con") and self.con:
            self.con.close()
            self.con = None

    def save_to_csv(self, df: pd.DataFrame, file_path: str):
        df.to_csv(file_path, index=False, encoding="utf-8-sig", quoting=1, date_format="%Y-%m-%d %H:%M:%S")
        self.logger.info(f"Archivo CSV guardado exitosamente en: {file_path}")
