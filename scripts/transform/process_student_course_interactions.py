import pandas as pd
import os
import sys
import re
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.moodle_path_resolver import MoodlePathResolver
from utils.base_script import BaseScript
from utils.academic_period_utils import AcademicPeriodUtils

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class StudentCourseInteractions(BaseScript):
    @staticmethod
    def _process_course_interactions(df):
        # Asegurarse de que las columnas relevantes están en formato numérico
        df["num_views"] = pd.to_numeric(df["num_views"], errors="coerce").fillna(0)
        df["num_interactions"] = pd.to_numeric(df["num_interactions"], errors="coerce").fillna(0)

        # Agrupar por estudiante y curso
        student_course_interactions = (
            df.groupby(["moodle_user_id", "documento_identificación", "course_id", "period", "year", "sede"])
            .agg(
                total_views=("num_views", "sum"),
                total_interactions=("num_interactions", "sum"),
                num_modules_viewed=("has_viewed", "sum"),
                num_modules_interacted=("has_participated", "sum"),
                total_modules=("course_module_id", "nunique"),
            )
            .reset_index()
        )

        # También puedes agregar proporciones si deseas
        student_course_interactions["percent_modules_viewed"] = (
            student_course_interactions["num_modules_viewed"] / student_course_interactions["total_modules"]
        ).round(2)

        student_course_interactions["percent_modules_interacted"] = (
            student_course_interactions["num_modules_interacted"] / student_course_interactions["total_modules"]
        ).round(2)

        return student_course_interactions

    def process_all_course_interactions(self):
        df = pd.read_csv("data/interim/moodle/student_modules_moodle.csv")
        course_interactions_moodle = StudentCourseInteractions._process_course_interactions(df)

        df = pd.read_csv("data/interim/moodle/student_modules_edukrea.csv")
        course_interactions_edukrea = StudentCourseInteractions._process_course_interactions(df)

        self.save_to_csv(course_interactions_moodle, "data/interim/moodle/student_course_interactions_moodle.csv")
        self.save_to_csv(course_interactions_edukrea, "data/interim/moodle/student_course_interactions_edukrea.csv")
        self.logger.info("Proceso de logins de Moodle completado.")


if __name__ == "__main__":
    processor = StudentCourseInteractions()
    processor.process_all_course_interactions()
    processor.close()
    processor.logger.info("Moodle HVP processing completed.")
