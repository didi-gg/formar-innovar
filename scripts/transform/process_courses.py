import pandas as pd
import os
import sys
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.moodle_path_resolver import MoodlePathResolver
from utils.academic_period_utils import AcademicPeriodUtils
from utils.base_script import BaseScript


class CoursesProcessor(BaseScript):
    @staticmethod
    def _categorize_moodle_modules(df, module_col="module_type"):
        categories = {
            "is_evaluation": ["assign", "quiz", "workshop", "lesson", "choice", "feedback", "lti"],
            "is_collaboration": ["forum", "chat", "glossary", "workshop"],
            "is_content": ["resource", "page", "book", "folder", "url", "label", "bootstrapelements", "hvp", "lesson", "lti"],
        }
        for cat_name, module_list in categories.items():
            df[cat_name] = df[module_col].isin(module_list).astype(int)

        return df

    @staticmethod
    def _calculate_module_metrics(df):
        df["was_updated"] = (df["teacher_updated_before_start"] | df["teacher_updated_during_week_planned"]).astype(int)

        agg = (
            df.groupby(["course_id", "year", "period"])
            .agg(
                num_modules=("course_module_id", "count"),
                num_modules_updated=("was_updated", "sum"),
                num_teacher_views_before_planned_start_date=("teacher_accessed_before_start", "sum"),
                teacher_total_updates=("teacher_total_updates", "sum"),
                teacher_total_views=("teacher_total_views", "sum"),
                student_total_views=("student_total_views", "sum"),
                student_total_interactions=("student_total_interactions", "sum"),
                min_days_since_creation=("days_since_creation", "min"),
                max_days_since_creation=("days_since_creation", "max"),
                avg_days_since_creation=("days_since_creation", "mean"),
                median_days_since_creation=("days_since_creation", "median"),
                avg_days_since_last_update=("days_since_last_update", "mean"),
                median_days_since_last_update=("days_since_last_update", "median"),
            )
            .reset_index()
        )

        # Redondear promedios a 2 decimales
        cols_to_round = [
            "avg_days_since_creation",
            "avg_days_since_last_update",
        ]
        agg[cols_to_round] = agg[cols_to_round].round(2)

        return agg

    @staticmethod
    def _adding_percentages(moodle_summary_df):
        moodle_summary_df["percent_evaluation"] = (moodle_summary_df["count_evaluation"] / moodle_summary_df["num_modules"]).fillna(0).round(2)
        moodle_summary_df["percent_collaboration"] = (moodle_summary_df["count_collaboration"] / moodle_summary_df["num_modules"]).fillna(0).round(2)
        moodle_summary_df["percent_content"] = (moodle_summary_df["count_content"] / moodle_summary_df["num_modules"]).fillna(0).round(2)
        moodle_summary_df["percent_in_english"] = (moodle_summary_df["count_in_english"] / moodle_summary_df["num_modules"]).fillna(0).round(2)
        moodle_summary_df["percent_interactive"] = (moodle_summary_df["count_interactive"] / moodle_summary_df["num_modules"]).fillna(0).round(2)
        moodle_summary_df["percent_updated"] = (moodle_summary_df["num_modules_updated"] / moodle_summary_df["num_modules"]).fillna(0).round(2)

        return moodle_summary_df

    @staticmethod
    def _group_moodle_modules(df):
        # Group by the required columns and aggregate counts
        moodle_summary_df = (
            df.groupby(["sede", "id_grado", "id_asignatura", "asignatura_name", "course_id", "course_name", "period", "year", "total_students"])
            .agg(
                {
                    "is_evaluation": "sum",
                    "is_collaboration": "sum",
                    "is_content": "sum",
                    "is_in_english": "sum",
                    "is_interactive": "sum",
                }
            )
            .reset_index()
        )

        # Rename columns for better readability
        moodle_summary_df = moodle_summary_df.rename(
            columns={
                "is_evaluation": "count_evaluation",
                "is_collaboration": "count_collaboration",
                "is_content": "count_content",
                "is_in_english": "count_in_english",
                "is_interactive": "count_interactive",
            }
        )

        # Display the resulting DataFrame
        return moodle_summary_df

    @staticmethod
    def process_df(df):
        modules_df = CoursesProcessor._categorize_moodle_modules(df)
        moodle_summary_df = CoursesProcessor._group_moodle_modules(modules_df)
        temporal_metrics_df = CoursesProcessor._calculate_module_metrics(modules_df)

        moodle_summary_df = moodle_summary_df.merge(temporal_metrics_df, on=["course_id", "year", "period"], how="left")
        moodle_summary_df = CoursesProcessor._adding_percentages(moodle_summary_df)
        return moodle_summary_df

    def process_course_data(self):
        modules_moodle_df = pd.read_csv("data/interim/moodle/modules_featured_moodle.csv")
        moodle_courses = CoursesProcessor.process_df(modules_moodle_df)

        modules_edukrea_df = pd.read_csv("data/interim/moodle/modules_featured_edukrea.csv")
        edukrea_courses = CoursesProcessor.process_df(modules_edukrea_df)

        moodle_courses.to_csv("data/interim/moodle/courses_moodle.csv", index=False, encoding="utf-8-sig", quoting=1)
        edukrea_courses.to_csv("data/interim/moodle/courses_edukrea.csv", index=False, encoding="utf-8-sig", quoting=1)
        self.logger.info("Courses data processed and saved successfully.")


if __name__ == "__main__":
    processor = CoursesProcessor()
    processor.process_course_data()
    processor.logger.info("Modules processed successfully.")
    processor.close()
