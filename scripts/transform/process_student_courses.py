"""
Este script crea un archivo CSV con la relación entre los estudiantes y los cursos en los que están inscritos,
incluyendo el nombre del curso e identificador de asignatura (id_asignatura), para el año 2025.
"""

import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.moodle_path_resolver import MoodlePathResolver
from utils.base_script import BaseScript


class StudentMoodleCoursesProcessor(BaseScript):
    def load_student_courses(self, year, enrollments_file, enrol_file, courses_file, students_file, mapping_file, excluded_courses=()):
        sql = f"""
            SELECT DISTINCT
                ue.userid AS moodle_user_id,
                {year} AS year,
                s.id_grado AS id_grado,
                e.courseid AS course_id,
                c.fullname AS course_name,
                s.documento_identificación AS documento_identificación,
                s.sede AS sede,
                map.id_asignatura AS id_asignatura
            FROM '{enrollments_file}' ue
            JOIN '{enrol_file}' e ON ue.enrolid = e.id
            JOIN '{courses_file}' c ON e.courseid = c.id
            JOIN '{students_file}' s ON ue.userid = s.moodle_user_id
            JOIN read_csv_auto('{mapping_file}') AS map ON e.courseid = map.course_id
            WHERE s.year = {year}
                AND e.courseid NOT IN {excluded_courses}
                AND s.id_grado = map.id_grado
                AND c.visible = 1
        """
        try:
            return self.con.execute(sql).df()
        except Exception as e:
            self.logger.error(f"Error cargando datos de estudiantes para el año {year}: {str(e)}")
            raise

    def load_student_courses_edukrea(self, year, enrollments_file, enrol_file, courses_file, students_file, mapping_file, excluded_courses=()):
        base_sql = f"""
            SELECT DISTINCT
                ue.userid AS moodle_user_id,
                {year} AS year,
                s.id_grado AS id_grado,
                e.courseid AS course_id,
                c.fullname AS course_name,
                s.documento_identificación AS documento_identificación,
                s.sede AS sede,
                map.id_asignatura AS id_asignatura
            FROM '{enrollments_file}' ue
            JOIN '{enrol_file}' e ON ue.enrolid = e.id
            JOIN '{courses_file}' c ON e.courseid = c.id
            JOIN '{students_file}' s ON ue.userid = s.edukrea_user_id
            JOIN '{mapping_file}' AS map ON e.courseid = map.course_id
            WHERE s.year = {year}
                AND c.visible = 1
        """

        # Agregar condición solo si hay cursos excluidos
        if excluded_courses:
            courses_str = ", ".join(str(cid) for cid in excluded_courses)
            base_sql += f" AND e.courseid NOT IN ({courses_str})"

        try:
            return self.con.execute(base_sql).df()
        except Exception as e:
            self.logger.error(f"Error cargando datos de estudiantes para el año {year}: {str(e)}")
            raise

    def process_student_courses(self):
        students_file = "data/interim/estudiantes/enrollments.csv"
        mapping_file = "data/interim/moodle/course_mapping_moodle.csv"

        excluded_courses = (
            # Institucionales y prueba
            549,
            550,
            332,
            40,
            # Inteligencia Emocional
            154,
            386,
            411,
            515,
            155,
            390,
            394,
            156,
            157,
            398,
            158,
            402,
            159,
            416,
            160,
            418,
            213,
            502,
            409,
            565,
        )

        # Process 2024
        year = 2024
        enrollments_file, enrol_file, courses_file = MoodlePathResolver.get_paths(year, "user_enrolments", "enrol", "course")
        df_2024 = self.load_student_courses(year, enrollments_file, enrol_file, courses_file, students_file, mapping_file, excluded_courses)

        # Process 2025
        year = 2025
        enrollments_file, enrol_file, courses_file = MoodlePathResolver.get_paths(year, "user_enrolments", "enrol", "course")
        df_2025 = self.load_student_courses(year, enrollments_file, enrol_file, courses_file, students_file, mapping_file, excluded_courses)

        df_combined = pd.concat([df_2024, df_2025], ignore_index=True)
        # Guardar como CSV
        output_file = "data/interim/moodle/student_courses_moodle.csv"
        self.save_to_csv(df_combined, output_file)

        # Process Edukrea
        mapping_file = "data/interim/moodle/course_mapping_edukrea.csv"
        enrollments_file, enrol_file, courses_file = MoodlePathResolver.get_paths("Edukrea", "user_enrolments", "enrol", "course")
        df_edukrea = self.load_student_courses_edukrea(2025, enrollments_file, enrol_file, courses_file, students_file, mapping_file, ())

        # Guardar como CSV
        output_file = "data/interim/moodle/student_courses_edukrea.csv"
        self.save_to_csv(df_edukrea, output_file)


if __name__ == "__main__":
    processor = StudentMoodleCoursesProcessor()
    processor.process_student_courses()
    processor.logger.info("Student courses processed successfully.")
    processor.close()
