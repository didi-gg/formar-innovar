import duckdb
import pandas as pd
import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class MoodleStudentLogs:
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

    def _get_log(self, year, logs_parquet, student_courses_file):
        sql_logs_estudiantes = f"""
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
            INNER JOIN '{student_courses_file}' AS student_courses ON student_courses.course_id = logs.courseid AND student_courses.moodle_user_id = logs.userid
            WHERE
                EXTRACT(YEAR FROM to_timestamp(logs.timecreated)) = {year}
                AND student_courses.year = {year}
            ORDER BY logs.timecreated
        """
        try:
            return self.con.execute(sql_logs_estudiantes).df()
        except Exception as e:
            self.logger.error(f"Error al cargar los logs: {str(e)}")
            raise

    def process_student_logs(self):
        student_courses_file = "data/interim/moodle/student_moodle_courses.csv"

        # Get logs for 2024
        year = 2024
        logs_parquet = f"data/raw/moodle/{year}/Log/mdlvf_logstore_standard_log.parquet"
        log_2024 = self._get_log(year, logs_parquet, student_courses_file)

        # Get logs for 2025
        year = 2025
        logs_parquet = f"data/raw/moodle/{year}/Log/mdlvf_logstore_standard_log.parquet"
        log_2025 = self._get_log(year, logs_parquet, student_courses_file)

        # Get logs Edukrea
        year = 2025
        logs_parquet = "data/raw/moodle/Edukrea/Logs and Events/mdl_logstore_standard_log.parquet"
        student_courses_file = "data/interim/moodle/student_edukrea_courses.csv"
        logs_edukrea = self._get_log(year, logs_parquet, student_courses_file)

        # Concatenate 2024 y 2025 logs
        logs_2024_2025 = pd.concat([log_2024, log_2025], ignore_index=True)

        # Save as csv
        output_file = "data/interim/moodle/students_logs_moodle.csv"
        logs_2024_2025.to_csv(output_file, index=False, encoding="utf-8-sig", quoting=1)

        # Save Edukrea logs
        output_file_edukrea = "data/interim/moodle/students_logs_edukrea.csv"
        logs_edukrea.to_csv(output_file_edukrea, index=False, encoding="utf-8-sig", quoting=1)


if __name__ == "__main__":
    processor = MoodleStudentLogs()
    processor.process_student_logs()
