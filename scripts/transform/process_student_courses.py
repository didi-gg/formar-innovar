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
    def load_student_courses(self, year, enrollments_file, enrol_file, courses_file, students_file, mapping_file, platform = 'moodle', excluded_courses=()):

        if platform == 'edukrea':
            user_id_col = 'edukrea_user_id'
        else:
            user_id_col = 'moodle_user_id'

        sql = f"""
            SELECT DISTINCT
                '{platform}' AS platform,
                s.{user_id_col} AS moodle_user_id,
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
            JOIN '{students_file}' s ON ue.userid = s.{user_id_col}
            JOIN '{mapping_file}' AS map ON e.courseid = map.course_id AND map.platform = '{platform}'
            WHERE s.year = {year}
                AND s.id_grado = map.id_grado
                AND c.visible = 1
        """

        # Agregar condición solo si hay cursos excluidos
        if excluded_courses:
            courses_str = ", ".join(str(cid) for cid in excluded_courses)
            sql += f" AND e.courseid NOT IN ({courses_str})"

        try:
            return self.con.execute(sql).df()
        except Exception as e:
            self.logger.error(f"Error cargando datos de estudiantes para el año {year}: {str(e)}")
            raise

    def process_student_courses(self):
        students_file = "data/interim/estudiantes/enrollments.csv"
        mapping_file = "data/interim/moodle/course_mapping.csv"

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
        df_2024 = self.load_student_courses(year, enrollments_file, enrol_file, courses_file, students_file, mapping_file, platform = 'moodle', excluded_courses=excluded_courses)

        # Process 2025
        year = 2025
        enrollments_file, enrol_file, courses_file = MoodlePathResolver.get_paths(year, "user_enrolments", "enrol", "course")
        df_2025 = self.load_student_courses(year, enrollments_file, enrol_file, courses_file, students_file, mapping_file, platform = 'moodle', excluded_courses=excluded_courses)

        # Process Edukrea
        enrollments_file, enrol_file, courses_file = MoodlePathResolver.get_paths("Edukrea", "user_enrolments", "enrol", "course")
        df_edukrea = self.load_student_courses(2025, enrollments_file, enrol_file, courses_file, students_file, mapping_file, platform = 'edukrea', excluded_courses=())

        df_combined = pd.concat([df_2024, df_2025, df_edukrea], ignore_index=True)

        # Guardar como CSV
        output_file = "data/interim/moodle/student_courses.csv"
        self.save_to_csv(df_combined, output_file)


if __name__ == "__main__":
    processor = StudentMoodleCoursesProcessor()
    processor.process_student_courses()
    processor.logger.info("Student courses processed successfully.")
    processor.close()
