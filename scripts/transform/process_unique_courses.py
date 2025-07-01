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
        unique_cols = ["year", "id_grado", "course_id", "course_name", "sede", "id_asignatura"]
        return df[unique_cols].drop_duplicates()

    def process_course_data(self):
        """
        Carga los datos de cursos desde archivos CSV intermedios,
        extrae cursos únicos y los guarda en nuevos archivos CSV.
        """
        student_courses_input_path = "data/interim/moodle/student_courses.csv"
        student_courses = pd.read_csv(student_courses_input_path)

        moodle_df = student_courses[student_courses['platform'] == 'Moodle']
        edukrea_df = student_courses[student_courses['platform'] == 'Edukrea']

        unique_moodle_courses = self._extract_unique_courses(moodle_df)
        unique_edukrea_courses = self._extract_unique_courses(edukrea_df)

        self.save_to_csv(unique_moodle_courses, "data/interim/moodle/unique_courses_moodle.csv")
        self.save_to_csv(unique_edukrea_courses, "data/interim/moodle/unique_courses_edukrea.csv")
        self.logger.info("Procesamiento de cursos completado.")


if __name__ == "__main__":
    processor = UniqueCoursesProcessor()
    processor.process_course_data()
    processor.logger.info("Unique courses processed successfully.")
    processor.close()
