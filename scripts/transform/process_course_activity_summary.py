import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.moodle_path_resolver import MoodlePathResolver
from utils.base_script import BaseScript
from utils.academic_period_utils import AcademicPeriodUtils


class MoodleCourseActivityProcessor(BaseScript):
    """
    Procesa eventos de interacción con cursos y recursos en Moodle.
    """

    def __init__(self):
        self.period_utils = AcademicPeriodUtils()
        super().__init__()

    def _load_course_activity_data(self, logs_parquet, student_courses, year=2024, platform='moodle'):
        try:
            # Eventos principales
            events_of_interest = {
                "\\core\\event\\course_viewed": "course_viewed",
                "\\mod_hvp\\event\\course_module_viewed": "hvp_module_viewed",
                "\\mod_assign\\event\\course_module_viewed": "assign_module_viewed",
                "\\mod_assign\\event\\submission_status_viewed": "submission_status_viewed",
                "\\mod_hvp\\event\\attempt_submitted": "hvp_attempt_submitted",
                "\\mod_forum\\event\\course_module_viewed": "forum_module_viewed",
                "\\mod_url\\event\\course_module_viewed": "url_module_viewed",
                "\\mod_resource\\event\\course_module_viewed": "resource_module_viewed",
                "\\mod_forum\\event\\post_created": "forum_post_created",
                "\\mod_quiz\\event\\course_module_viewed": "quiz_module_viewed",
                "\\mod_forum\\event\\discussion_subscription_created": "forum_discussion_subscription_created",
                "\\mod_quiz\\event\\attempt_viewed": "quiz_attempt_viewed",
                "\\mod_forum\\event\\assessable_uploaded": "forum_assessable_uploaded",
                "\\mod_quiz\\event\\attempt_summary_viewed": "quiz_attempt_summary_viewed",
                "\\mod_quiz\\event\\attempt_started": "quiz_attempt_started",
                "\\mod_quiz\\event\\attempt_reviewed": "quiz_attempt_reviewed",
                "\\mod_quiz\\event\\attempt_submitted": "quiz_attempt_submitted",
                "\\mod_assign\\event\\submission_form_viewed": "assign_submission_form_viewed",
                "\\mod_assign\\event\\assessable_submitted": "assign_assessable_submitted",
                "\\assignsubmission_file\\event\\assessable_uploaded": "file_assessable_uploaded",
                "\\assignsubmission_file\\event\\submission_created": "file_submission_created",
                "\\mod_forum\\event\\discussion_viewed": "forum_discussion_viewed",
                "\\mod_forum\\event\\post_updated": "forum_post_updated",
                "\\mod_forum\\event\\post_deleted": "forum_post_deleted",
                "\\mod_forum\\event\\subscription_created": "forum_subscription_created",
                "\\assignsubmission_onlinetext\\event\\assessable_uploaded": "onlinetext_assessable_uploaded",
                "\\mod_forum\\event\\discussion_deleted": "forum_discussion_deleted",
                "\\assignsubmission_onlinetext\\event\\submission_created": "onlinetext_submission_created",
                "\\mod_assign\\event\\feedback_viewed": "assign_feedback_viewed",
                "\\mod_forum\\event\\discussion_created": "forum_discussion_created",
                "\\mod_lti\\event\\course_module_viewed": "lti_module_viewed",
                "\\mod_quiz\\event\\attempt_preview_started": "quiz_attempt_preview_started",
                "\\mod_page\\event\\course_module_viewed": "page_module_viewed",
                "\\assignsubmission_comments\\event\\comment_created": "submission_comment_created",
                "\\mod_feedback\\event\\course_module_viewed": "feedback_module_viewed",
                "\\mod_chat\\event\\course_module_viewed": "chat_module_viewed",
                "\\mod_forum\\event\\course_searched": "forum_course_searched",
                "\\mod_forum\\event\\discussion_subscription_deleted": "forum_discussion_subscription_deleted",
                "\\mod_assign\\event\\remove_submission_form_viewed": "assign_remove_submission_form_viewed",
                "\\mod_choice\\event\\course_module_viewed": "choice_module_viewed",
                "\\mod_feedback\\event\\response_submitted": "feedback_response_submitted",
                "\\mod_choice\\event\\answer_created": "choice_answer_created",
            }

            where_clause = " OR ".join([f"eventname = '{e}'" for e in events_of_interest.keys()])

            sql = f"""
                SELECT 
                    {year} AS year,
                    logs.userid,
                    logs.timecreated,
                    logs.eventname,
                    logs.courseid
                FROM '{logs_parquet}' AS logs
                WHERE ({where_clause})
                AND EXTRACT(YEAR FROM to_timestamp(timecreated)) = {year}
            """

            df = self.con.execute(sql).df()

            # Procesar timestamps
            df["timecreated"] = pd.to_datetime(df["timecreated"], unit="s", utc=True).dt.tz_convert("America/Bogota")
            df["period"] = df["timecreated"].apply(self.period_utils.determine_period_from_date)

            # Mapear tipos de actividad
            df["activity_type"] = df["eventname"].map(events_of_interest)

            # Agregar columna platform
            df["platform"] = platform

            # Cargar inscripciones de estudiantes
            df_courses = self.con.execute(f"""
                SELECT moodle_user_id, course_id, id_asignatura, year, course_name, documento_identificación
                FROM '{student_courses}'
                WHERE platform = '{platform}'
            """).df()

            # Unir para quedarnos solo con cursos inscritos
            df_merged = df.merge(
                df_courses, left_on=["userid", "courseid"], right_on=["moodle_user_id", "course_id"], how="inner", suffixes=("", "_student")
            )

            return df_merged

        except Exception as e:
            self.logger.error(f"Error cargando datos de actividad para el año {year}: {str(e)}")
            raise

    def process_course_activity(self):
        logs_table = "logstore_standard_log"
        logs_parquet_2024 = MoodlePathResolver.get_paths(2024, logs_table)[0]
        logs_parquet_2025 = MoodlePathResolver.get_paths(2025, logs_table)[0]
        logs_parquet_edukrea = MoodlePathResolver.get_paths("Edukrea", logs_table)[0]

        student_courses = "data/interim/moodle/student_courses.csv"

        # Cargar datos de 2024 y 2025
        data_2024 = self._load_course_activity_data(logs_parquet_2024, student_courses, year=2024, platform='moodle')
        data_2025 = self._load_course_activity_data(logs_parquet_2025, student_courses, year=2025, platform='moodle')
        data_edukrea = self._load_course_activity_data(logs_parquet_edukrea, student_courses, year=2025, platform='edukrea')

        # Combinar todos los datos
        combined_data = pd.concat([data_2024, data_2025, data_edukrea])

        # Pivotear para contar eventos por tipo de actividad
        df_summary = combined_data.pivot_table(
            index=["userid", "documento_identificación", "courseid", "period", "id_asignatura", "year", "course_name", "platform"],
            columns="activity_type",
            values="timecreated",
            aggfunc="count",
            fill_value=0,
        ).reset_index()

        # Asegurar que todas las columnas existan
        expected_columns = [
            # Acceso a contenidos
            "course_viewed",
            "hvp_module_viewed",
            "assign_module_viewed",
            "submission_status_viewed",
            "forum_module_viewed",
            "url_module_viewed",
            "resource_module_viewed",
            "lti_module_viewed",
            "page_module_viewed",
            "chat_module_viewed",
            "feedback_module_viewed",
            "choice_module_viewed",
            # Evaluaciones y entregas
            "hvp_attempt_submitted",
            "quiz_module_viewed",
            "quiz_attempt_viewed",
            "quiz_attempt_summary_viewed",
            "quiz_attempt_started",
            "quiz_attempt_reviewed",
            "quiz_attempt_submitted",
            "assign_submission_form_viewed",
            "assign_assessable_submitted",
            "file_assessable_uploaded",
            "file_submission_created",
            "onlinetext_assessable_uploaded",
            "onlinetext_submission_created",
            "assign_remove_submission_form_viewed",
            "assign_feedback_viewed",
            "submission_comment_created",
            "feedback_response_submitted",
            "choice_answer_created",
            # Participación social en foros
            "forum_post_created",
            "forum_post_updated",
            "forum_post_deleted",
            "forum_discussion_viewed",
            "forum_discussion_created",
            "forum_assessable_uploaded",
            "forum_subscription_created",
            "forum_discussion_subscription_created",
            "forum_discussion_deleted",
            "forum_discussion_subscription_deleted",
            "forum_course_searched",
        ]
        for col in expected_columns:
            if col not in df_summary.columns:
                df_summary[col] = 0

        # Guardar resultados
        output_path = "data/interim/moodle/course_activity_summary.csv"
        self.save_to_csv(df_summary, output_path)

        return df_summary


if __name__ == "__main__":
    processor = MoodleCourseActivityProcessor()
    processor.process_course_activity()
    processor.logger.info("Moodle course activity processed successfully.")
    processor.close()
