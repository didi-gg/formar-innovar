import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.base_script import BaseScript


class StudentModulesProcessor(BaseScript):
    @staticmethod
    def _classify_event(eventname):
        eventname = str(eventname).lower()
        if "view" in eventname:
            return "view"
        else:
            return "interaction"

    @staticmethod
    def _merge_modules_students(students_df, modules_df):
        students_sel = students_df[["moodle_user_id", "documento_identificación", "year", "course_id", "sede"]]
        return students_sel.merge(modules_df, on=["course_id", "year", "sede"], how="inner")

    @staticmethod
    def _get_student_logs_summary(logs_df):
        student_logs_summary = (
            logs_df.groupby(["userid", "contextinstanceid", "year"])
            .agg(
                num_views=("event_type", lambda x: (x == "view").sum()),
                num_interactions=("event_type", lambda x: (x == "interaction").sum()),
                first_view=("timecreated", "min"),
                last_view=("timecreated", "max"),
            )
            .reset_index()
            .rename(columns={"userid": "moodle_user_id", "contextinstanceid": "course_module_id"})
        )
        return student_logs_summary

    @staticmethod
    def _calculate_metrics(df):
        df["first_view"] = pd.to_datetime(df["first_view"], errors="coerce")
        df["last_view"] = pd.to_datetime(df["last_view"], errors="coerce")
        df["planned_start_date"] = pd.to_datetime(df["planned_start_date"], errors="coerce")
        df["planned_end_date"] = pd.to_datetime(df["planned_end_date"], errors="coerce")

        # Cálculo de métricas
        df["has_viewed"] = (df["num_views"].fillna(0) > 0).astype(int)
        df["has_participated"] = (df["num_interactions"].fillna(0) > 0).astype(int)
        df["days_before_start"] = (df["first_view"] - df["planned_start_date"]).dt.days
        df["days_after_end"] = (df["last_view"] - df["planned_end_date"]).dt.days
        df["was_on_time"] = ((df["first_view"] >= df["planned_start_date"]) & (df["first_view"] <= df["planned_end_date"])).astype(int)

        return df

    def _process_df(modules_df, students_df, logs_df):
        modules_df = modules_df[
            [
                "year",
                "course_id",
                "course_module_id",
                "sede",
                "id_grado",
                "id_asignatura",
                "asignatura_name",
                "course_name",
                "section_id",
                "section_name",
                "module_type_id",
                "instance",
                "module_creation_date",
                "module_type",
                "module_name",
                "week",
                "period",
                "is_interactive",
                "is_in_english",
                "planned_start_date",
                "planned_end_date",
            ]
        ]

        df_base = StudentModulesProcessor._merge_modules_students(students_df, modules_df)
        logs_df["event_type"] = logs_df["eventname"].apply(StudentModulesProcessor._classify_event)
        student_logs_summary = StudentModulesProcessor._get_student_logs_summary(logs_df)
        df_full = df_base.merge(student_logs_summary, on=["moodle_user_id", "course_module_id", "year"], how="left")
        df_full = StudentModulesProcessor._calculate_metrics(df_full)

        return df_full

    def process_course_data(self):
        modules_df = pd.read_csv("data/interim/moodle/modules_featured_moodle.csv")
        students_df = pd.read_csv("data/interim/moodle/student_courses_moodle.csv")
        logs_df = pd.read_csv("data/interim/moodle/student_logs_moodle.csv")

        student_modules_moodle = StudentModulesProcessor._process_df(modules_df, students_df, logs_df)

        modules_df = pd.read_csv("data/interim/moodle/modules_featured_edukrea.csv")
        students_df = pd.read_csv("data/interim/moodle/student_courses_edukrea.csv")
        logs_df = pd.read_csv("data/interim/moodle/student_logs_edukrea.csv")

        student_modules_edukrea = StudentModulesProcessor._process_df(modules_df, students_df, logs_df)

        self.save_to_csv(student_modules_moodle, "data/interim/moodle/student_modules_moodle.csv")
        self.save_to_csv(student_modules_edukrea, "data/interim/moodle/student_modules_edukrea.csv")

        self.logger.info("Courses data processed and saved successfully.")


if __name__ == "__main__":
    processor = StudentModulesProcessor()
    processor.process_course_data()
    processor.logger.info("Student Modules processed successfully.")
    processor.close()
