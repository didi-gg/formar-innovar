import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.moodle_path_resolver import MoodlePathResolver
from utils.base_script import BaseScript


class StudentLoginProcessor(BaseScript):
    def _get_log(self, year, logs_parquet, student_courses_file, platform='moodle'):
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
                student_courses.documento_identificación,
                logs.courseid,
                logs.timecreated,
                logs.origin,
                logs.ip,
                '{platform}' AS platform
            FROM '{logs_parquet}' AS logs
            INNER JOIN '{student_courses_file}' AS student_courses ON student_courses.course_id = logs.courseid AND student_courses.moodle_user_id = logs.userid AND student_courses.platform = '{platform}'
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
        student_courses_file = "data/interim/moodle/student_courses.csv"
        logs_table = "logstore_standard_log"

        # Get logs for 2024
        year = 2024
        logs_parquet = MoodlePathResolver.get_paths(year, logs_table)[0]
        log_2024 = self._get_log(year, logs_parquet, student_courses_file, platform='moodle')

        # Add platform column and create moodle_user_id for moodle 2024
        log_2024['moodle_user_id'] = log_2024['userid']

        # Get logs for 2025
        year = 2025
        logs_parquet = MoodlePathResolver.get_paths(year, logs_table)[0]
        log_2025 = self._get_log(year, logs_parquet, student_courses_file, platform='moodle')
        
        # Add platform column and create moodle_user_id for moodle 2025
        log_2025['moodle_user_id'] = log_2025['userid']

        # Get logs Edukrea
        year = 2025
        logs_parquet = MoodlePathResolver.get_paths("Edukrea", logs_table)[0]
        logs_edukrea = self._get_log(year, logs_parquet, student_courses_file, platform='edukrea')
        
        # Add platform column and create edukrea_user_id for edukrea
        logs_edukrea['edukrea_user_id'] = logs_edukrea['userid']

        # Concatenate all logs (2024, 2025, and Edukrea)
        all_logs = pd.concat([log_2024, log_2025, logs_edukrea], ignore_index=True)
        
        # Convert user IDs to integers after concatenation
        if 'moodle_user_id' in all_logs.columns:
            all_logs['moodle_user_id'] = all_logs['moodle_user_id'].fillna(0).astype('Int64')
        if 'edukrea_user_id' in all_logs.columns:
            all_logs['edukrea_user_id'] = all_logs['edukrea_user_id'].fillna(0).astype('Int64')

        # Save combined logs as csv
        output_file = "data/interim/moodle/student_logs.csv"
        self.save_to_csv(all_logs, output_file)

if __name__ == "__main__":
    processor = StudentLoginProcessor()
    processor.process_student_logs()
    processor.logger.info("Student logs processed successfully.")
    processor.close()
