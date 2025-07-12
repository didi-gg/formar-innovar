import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.base_script import BaseScript


class UniqueCoursesProcessor(BaseScript):
    """
    Procesador de cursos de Moodle y Edukrea. Carga datos de cursos por estudiante,
    filtra cursos únicos y guarda los resultados en archivos CSV.
    """

    def _extract_unique_courses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrae combinaciones únicas de cursos de un DataFrame.

        Args:
            df (pd.DataFrame): DataFrame con columnas relacionadas a cursos.

        Returns:
            pd.DataFrame: DataFrame con cursos únicos.
        """
        unique_cols = ["year", "id_grado", "course_id", "course_name", "sede", "id_asignatura", "platform"]
        return df[unique_cols].drop_duplicates()

    def process_course_data(self):
        """
        Carga los datos de cursos desde archivos CSV intermedios,
        extrae cursos únicos y los guarda en un solo archivo CSV.
        """
        student_courses_input_path = "data/interim/moodle/student_courses.csv"
        student_courses = pd.read_csv(student_courses_input_path)

        # Extraer cursos únicos de todo el dataset manteniendo la columna platform
        unique_courses = self._extract_unique_courses(student_courses)
        unique_courses["course_name"] = unique_courses["course_name"].apply(self._clean_text_field)

        self.save_to_csv(unique_courses, "data/interim/moodle/unique_courses.csv")
        self.logger.info(f"Procesamiento de cursos completado. Total cursos únicos: {len(unique_courses)}")
        self.logger.info(f"Cursos por plataforma: {unique_courses['platform'].value_counts().to_dict()}")


if __name__ == "__main__":
    processor = UniqueCoursesProcessor()
    processor.process_course_data()
    processor.logger.info("Unique courses processed successfully.")
    processor.close()
