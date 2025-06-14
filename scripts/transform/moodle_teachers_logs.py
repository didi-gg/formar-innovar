import duckdb
import pandas as pd
import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class MoodleTeachersLogs:
    def __init__(self):
        self.con = duckdb.connect()
        self.logger = logging.getLogger(__name__)

    def __del__(self):
        if hasattr(self, "con") and self.con:
            self.con.close()

    def close(self):
        if hasattr(self, "con") and self.con:
            self.con.close()
            self.con = None

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
        # Convertimos a texto SQL
        docente_ids_sql = str(docente_ids)

        sql_logs_docentes = f"""
            SELECT 
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
        unique_courses_file = "data/interim/moodle/courses_unique_moodle.csv"

        # Get logs for 2024
        year = 2024
        course_file = f"data/raw/moodle/{year}/Course/mdlvf_course.parquet"
        context_file = f"data/raw/moodle/{year}/System/mdlvf_context.parquet"
        role_assignments_file = f"data/raw/moodle/{year}/Users/mdlvf_role_assignments.parquet"
        role_file = f"data/raw/moodle/{year}/Users/mdlvf_role.parquet"
        user_file = f"data/raw/moodle/{year}/Users/mdlvf_user.parquet"
        logs_parquet = f"data/raw/moodle/{year}/Log/mdlvf_logstore_standard_log.parquet"

        teacher_ids_2024 = self._get_teacher_ids(year, course_file, unique_courses_file, context_file, role_assignments_file, role_file, user_file)
        log_2024 = self._get_log(year, teacher_ids_2024, logs_parquet, unique_courses_file)

        # Get logs for 2025
        year = 2025
        course_file = f"data/raw/moodle/{year}/Course/mdlvf_course.parquet"
        context_file = f"data/raw/moodle/{year}/System/mdlvf_context.parquet"
        role_assignments_file = f"data/raw/moodle/{year}/Users/mdlvf_role_assignments.parquet"
        role_file = f"data/raw/moodle/{year}/Users/mdlvf_role.parquet"
        user_file = f"data/raw/moodle/{year}/Users/mdlvf_user.parquet"
        logs_parquet = f"data/raw/moodle/{year}/Log/mdlvf_logstore_standard_log.parquet"

        teacher_ids_2025 = self._get_teacher_ids(year, course_file, unique_courses_file, context_file, role_assignments_file, role_file, user_file)
        log_2025 = self._get_log(year, teacher_ids_2025, logs_parquet, unique_courses_file)

        # Get logs Edukrea
        year = 2025
        unique_courses_file = "data/interim/moodle/courses_unique_edukrea.csv"
        course_file = "data/raw/moodle/Edukrea/Courses/mdl_course.parquet"
        context_file = "data/raw/moodle/Edukrea/Access and Roles/mdl_context.parquet"
        role_assignments_file = "data/raw/moodle/Edukrea/Assignments and Grades/mdl_role_assignments.parquet"
        role_file = "data/raw/moodle/Edukrea/Access and Roles/mdl_role.parquet"
        user_file = "data/raw/moodle/Edukrea/Users/mdl_user.parquet"
        logs_parquet = "data/raw/moodle/Edukrea/Logs and Events/mdl_logstore_standard_log.parquet"

        teacher_ids_edukrea = self._get_teacher_ids(year, course_file, unique_courses_file, context_file, role_assignments_file, role_file, user_file)
        logs_edukrea = self._get_log(year, teacher_ids_edukrea, logs_parquet, unique_courses_file)

        # Concatenate 2024 y 2025 logs
        logs_2024_2025 = pd.concat([log_2024, log_2025], ignore_index=True)

        # Save as csv
        output_file = "data/interim/moodle/teachers_logs_moodle.csv"
        logs_2024_2025.to_csv(output_file, index=False, encoding="utf-8-sig", quoting=1)

        # Save Edukrea logs
        output_file_edukrea = "data/interim/moodle/teachers_logs_edukrea.csv"
        logs_edukrea.to_csv(output_file_edukrea, index=False, encoding="utf-8-sig", quoting=1)


if __name__ == "__main__":
    processor = MoodleTeachersLogs()
    processor.process_teacher_logs()
