import pandas as pd
import pymysql
import os

import dotenv

dotenv.load_dotenv()

# Conexión a la base de datos
MOODLE_DB_HOST = os.getenv("MOODLE_DB_HOST")
MOODLE_DB_USER = os.getenv("MOODLE_DB_USER")
MOODLE_DB_PASS = os.getenv("MOODLE_DB_PASS")
MOODLE_DB_NAME = os.getenv("MOODLE_DB_NAME")

db_config = {"host": MOODLE_DB_HOST, "user": MOODLE_DB_USER, "password": MOODLE_DB_PASS, "database": MOODLE_DB_NAME}

try:
    conn = pymysql.connect(**db_config)
    print("Conexión exitosa a la base de datos.")
except pymysql.MySQLError as e:
    print(f"Error conectándose a la base de datos: {e}")
    exit()


# Consulta SQL
query = """
    SELECT 
        table_name AS table_name,
        table_rows AS row_count
    FROM 
        information_schema.tables
    WHERE 
        table_schema = 'cgalyttm_mood463'
    ORDER BY 
        row_count DESC;
"""

groups = {
    "Assignments": [
        "assign",
        "assign_grades",
        "assign_overrides",
        "assign_plugin_config",
        "assign_submission",
        "assign_user_flags",
        "assign_user_mapping",
        "assignfeedback_comments",
        "assignfeedback_file",
        "assignsubmission_file",
        "assignsubmission_onlinetext",
        "assignment",
        "assignment_submissions",
        "assignment_upgrade",
    ],
    "Chat": ["chat", "chat_messages", "chat_messages_current", "chat_users"],
    "Enrollment": ["enrol", "enrol_lti_tools", "enrol_paypal", "user_enrolments"],
    "LTI": ["lti", "lti_submission", "lti_tool_settings", "lti_types_config", "ltiservice_gradebookservices"],
    "Survey": ["survey", "survey_analysis", "survey_answers"],
    "Analytics": [
        "analytics_models",
        "analytics_models_log",
        "analytics_predict_samples",
        "analytics_predictions",
        "analytics_train_samples",
        "analytics_used_analysables",
        "analytics_used_files",
    ],
    "Choice": ["choice", "choice_answers", "choice_options"],
    "Feedback": ["feedback", "feedback_completed", "feedback_completedtmp", "feedback_item", "feedback_sitecourse_map"],
    "Page": ["page"],
    "Users": [
        "user",
        "user_devices",
        "user_enrolments",
        "user_info_data",
        "user_lastaccess",
        "user_password_history",
        "user_password_resets",
        "user_preferences",
        "user_private_key",
    ],
    "Badges": ["badge", "badge_alignment", "badge_criteria", "badge_endorsement", "badge_issued", "badge_manual_award", "badge_related"],
    "Course": [
        "course",
        "course_completion_aggr_methd",
        "course_completion_crit_compl",
        "course_completion_criteria",
        "course_completion_defaults",
        "course_completions",
        "course_format_options",
        "course_modules",
        "course_modules_viewed",
        "course_published",
        "course_sections",
    ],
    "Forum": [
        "forum",
        "forum_digests",
        "forum_discussion_subs",
        "forum_discussions",
        "forum_grades",
        "forum_read",
        "forum_subscriptions",
        "forum_track_prefs",
    ],
    "Quiz": [
        "quiz",
        "quiz_attempts",
        "quiz_feedback",
        "quiz_grade_items",
        "quiz_grades",
        "quiz_overrides",
        "quiz_sections",
        "quiz_slots",
        "quizaccess_seb_quizsettings",
    ],
    "Workshop": [
        "workshop",
        "workshop_aggregations",
        "workshop_submissions",
        "workshopallocation_scheduled",
        "workshopeval_best_settings",
        "workshopform_accumulative",
        "workshopform_comments",
        "workshopform_numerrors",
        "workshopform_numerrors_map",
        "workshopform_rubric",
        "workshopform_rubric_config",
    ],
    "Book": ["book"],
    "Data": ["data", "data_fields", "data_records"],
    "Lesson": ["lesson", "lesson_answers", "lesson_attempts", "lesson_branch", "lesson_grades", "lesson_overrides", "lesson_pages", "lesson_timer"],
    "Scorm": ["scorm", "scorm_aicc_session", "scorm_attempt", "scorm_scoes"],
    "Wiki": ["wiki"],
}


# Función para asignar grupo
def assign_group(table_name, groups):
    for group, tables in groups.items():
        if table_name in tables:
            return group
    return ""


try:
    # Conectar a la base de datos
    conn = pymysql.connect(**db_config)
    results = []

    # Obtener la lista de tablas
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'cgalyttm_mood463';
        """)
        tables = cursor.fetchall()

        # Iterar sobre las tablas y contar filas
        for (table_name,) in tables:
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                group = assign_group(table_name.replace("mdlvf_", ""), groups)
                results.append({"table_name": table_name, "row_count": row_count, "group": group})
                print(f"Tabla: {table_name}, Número de filas: {row_count}, Grupo: {group}")

    # Crear un DataFrame con los resultados
    df = pd.DataFrame(results)
    print("\nConteo de filas por tabla completado:")
    print(df)

    # Guardar los resultados en un archivo CSV
    df.to_csv("scripts_aux/tmp/conteo_filas_tablas.csv", index=False)
    print("Resultados guardados en 'scripts_aux/tmp/conteo_filas_tablas.csv'.")

except pymysql.MySQLError as e:
    print(f"Error conectándose a la base de datos o ejecutando consultas: {e}")
finally:
    # Cerrar la conexión
    conn.close()
    print("Conexión cerrada.")
