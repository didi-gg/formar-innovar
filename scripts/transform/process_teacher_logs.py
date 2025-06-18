import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.moodle_path_resolver import MoodlePathResolver
from utils.base_script import BaseScript


class TeacherLoginProcessor(BaseScript):
    def _get_teacher_ids(self, year, course_file, unique_courses_file, context_file, role_assignments_file, role_file, user_file):
        sql_docentes = f"""
            SELECT DISTINCT u.id AS userid
            FROM '{course_file}' c
            JOIN '{unique_courses_file}' uc ON c.id = uc.course_id
            JOIN '{context_file}' ctx ON ctx.instanceid = c.id AND ctx.contextlevel = 50
            JOIN '{role_assignments_file}' ra ON ra.contextid = ctx.id
            JOIN '{role_file}' r ON r.id = ra.roleid
            JOIN '{user_file}' u ON u.id = ra.userid
            WHERE r.shortname = 'editingteacher'  -- o 'teacher' si también quieres no-editing
            AND uc.year = {year}
            AND NOT (u.firstname = 'Provisional' AND u.lastname = 'Girardot')
            """
        try:
            docentes_df = self.con.execute(sql_docentes).df()
            return tuple(docentes_df["userid"].tolist())
        except Exception as e:
            self.logger.error(f"Error cargando datos de actividad para el año {year}: {str(e)}")
            raise

    def _get_log(self, year, docente_ids, logs_parquet, courses_file):
        docente_ids_sql = str(docente_ids)

        sql_logs_docentes = f"""
            SELECT 
                '{year}' AS year,
                logs.id,
                logs.eventname,
                logs.component,
                logs.action,
                logs.target,
                logs.objectid,
                logs.contextinstanceid,
                logs.userid,
                logs.courseid,
                logs.timecreated,
                logs.origin,
                logs.ip
            FROM '{logs_parquet}' AS logs
            INNER JOIN '{courses_file}' AS courses ON courses.course_id = logs.courseid
            WHERE
                EXTRACT(YEAR FROM to_timestamp(logs.timecreated)) = {year}
                AND courses.year = {year}
                AND logs.userid IN {docente_ids_sql}
            ORDER BY logs.timecreated
        """
        try:
            return self.con.execute(sql_logs_docentes).df()
        except Exception as e:
            self.logger.error(f"Error al cargar los logs: {str(e)}")
            raise

    def process_teacher_logs(self):
        unique_courses_file = "data/interim/moodle/unique_courses_moodle.csv"

        # Get logs for 2024
        year = 2024
        course_file, context_file, role_assignments_file, role_file, user_file, logs_parquet = MoodlePathResolver.get_paths(
            year, "course", "context", "role_assignments", "role", "user", "logstore_standard_log"
        )
        teacher_ids_2024 = self._get_teacher_ids(year, course_file, unique_courses_file, context_file, role_assignments_file, role_file, user_file)
        log_2024 = self._get_log(year, teacher_ids_2024, logs_parquet, unique_courses_file)

        # Get logs for 2025
        year = 2025
        course_file, context_file, role_assignments_file, role_file, user_file, logs_parquet = MoodlePathResolver.get_paths(
            year, "course", "context", "role_assignments", "role", "user", "logstore_standard_log"
        )
        teacher_ids_2025 = self._get_teacher_ids(year, course_file, unique_courses_file, context_file, role_assignments_file, role_file, user_file)
        log_2025 = self._get_log(year, teacher_ids_2025, logs_parquet, unique_courses_file)

        # Get logs Edukrea
        year = 2025
        course_file, context_file, role_assignments_file, role_file, user_file, logs_parquet = MoodlePathResolver.get_paths(
            "Edukrea", "course", "context", "role_assignments", "role", "user", "logstore_standard_log"
        )
        unique_courses_file = "data/interim/moodle/unique_courses_edukrea.csv"
        teacher_ids_edukrea = self._get_teacher_ids(year, course_file, unique_courses_file, context_file, role_assignments_file, role_file, user_file)
        logs_edukrea = self._get_log(year, teacher_ids_edukrea, logs_parquet, unique_courses_file)

        # Concatenate 2024 y 2025 logs
        logs_2024_2025 = pd.concat([log_2024, log_2025], ignore_index=True)

        # Save as csv
        output_file = "data/interim/moodle/teacher_logs_moodle.csv"
        self.save_to_csv(logs_2024_2025, output_file)

        # Save Edukrea logs
        output_file_edukrea = "data/interim/moodle/teacher_logs_edukrea.csv"
        self.save_to_csv(logs_edukrea, output_file_edukrea)


if __name__ == "__main__":
    processor = TeacherLoginProcessor()
    processor.process_teacher_logs()
    processor.logger.info("Teacher logs processed successfully.")
    processor.close()
