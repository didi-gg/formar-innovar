import duckdb
import pandas as pd
import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class MoodleCourseProcessor:
    """
    Procesador de cursos de Moodle y Edukrea. Carga datos de cursos por estudiante,
    filtra cursos únicos y guarda los resultados en archivos CSV.
    """

    def __init__(self):
        self.connection = duckdb.connect()
        self.logger = logging.getLogger(__name__)

    def __del__(self):
        self.close_connection()

    def close_connection(self):
        if hasattr(self, "connection") and self.connection:
            self.connection.close()
            self.connection = None

    def _extract_unique_courses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrae combinaciones únicas de cursos de un DataFrame.

        Args:
            df (pd.DataFrame): DataFrame con columnas relacionadas a cursos.

        Returns:
            pd.DataFrame: DataFrame con cursos únicos.
        """
        unique_cols = ["year", "id_grado", "course_id", "course_name", "sede", "id_asignatura"]
        return df[unique_cols].drop_duplicates()

    def save_to_csv(self, df: pd.DataFrame, file_path: str):
        """
        Guarda un DataFrame como archivo CSV.

        Args:
            df (pd.DataFrame): Datos a guardar.
            file_path (str): Ruta del archivo de salida.
        """
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        self.logger.info(f"Archivo guardado exitosamente en: {file_path}")

    def process_course_data(self):
        """
        Carga los datos de cursos desde archivos CSV intermedios,
        extrae cursos únicos y los guarda en nuevos archivos CSV.
        """
        moodle_input_path = "data/interim/moodle/student_moodle_courses.csv"
        edukrea_input_path = "data/interim/moodle/student_edukrea_courses.csv"

        moodle_df = pd.read_csv(moodle_input_path)
        edukrea_df = pd.read_csv(edukrea_input_path)

        unique_moodle_courses = self._extract_unique_courses(moodle_df)
        unique_edukrea_courses = self._extract_unique_courses(edukrea_df)

        self.save_to_csv(unique_moodle_courses, "data/interim/moodle/courses_unique_moodle.csv")
        self.save_to_csv(unique_edukrea_courses, "data/interim/moodle/courses_unique_edukrea.csv")
        self.logger.info("Procesamiento de cursos completado.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    processor = MoodleCourseProcessor()
    processor.process_course_data()
