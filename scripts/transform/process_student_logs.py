import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.moodle_path_resolver import MoodlePathResolver
from utils.base_script import BaseScript


class StudentLoginProcessor(BaseScript):
    def _get_log(self, year, logs_parquet, student_courses_file):
        sql_logs_estudiantes = f"""
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
        student_courses_file = "data/interim/moodle/student_courses_moodle.csv"
        logs_table = "logstore_standard_log"

        # Get logs for 2024
        year = 2024
        logs_parquet = MoodlePathResolver.get_paths(year, logs_table)[0]
        log_2024 = self._get_log(year, logs_parquet, student_courses_file)

        # Get logs for 2025
        year = 2025
        logs_parquet = MoodlePathResolver.get_paths(year, logs_table)[0]
        log_2025 = self._get_log(year, logs_parquet, student_courses_file)

        # Get logs Edukrea
        year = 2025
        logs_parquet = MoodlePathResolver.get_paths("Edukrea", logs_table)[0]
        student_courses_file = "data/interim/moodle/student_courses_edukrea.csv"
        logs_edukrea = self._get_log(year, logs_parquet, student_courses_file)

        # Concatenate 2024 y 2025 logs
        logs_2024_2025 = pd.concat([log_2024, log_2025], ignore_index=True)

        # Save as csv
        output_file = "data/interim/moodle/student_logs_moodle.csv"
        self.save_to_csv(logs_2024_2025, output_file)

        # Save Edukrea logs
        output_file_edukrea = "data/interim/moodle/student_logs_edukrea.csv"
        self.save_to_csv(logs_edukrea, output_file_edukrea)


if __name__ == "__main__":
    processor = StudentLoginProcessor()
    processor.process_student_logs()
    processor.logger.info("Student logs processed successfully.")
    processor.close()
