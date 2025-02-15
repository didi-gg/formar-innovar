"""
Este script extrae datos de una base de datos MySQL de Moodle y los convierte en archivos Parquet organizados por categorías.
"""

import pymysql
import pyarrow as pa
import pyarrow.parquet as pq
import os
import dotenv

# Carga de variables de entorno
dotenv.load_dotenv()

MOODLE_DB_HOST = os.getenv("MOODLE_DB_HOST")
MOODLE_DB_USER = os.getenv("MOODLE_DB_USER")
MOODLE_DB_PASS = os.getenv("MOODLE_DB_PASS")
MOODLE_DB_NAME = os.getenv("MOODLE_DB_NAME")

# Configuración de la conexión a MySQL
db_config = {
    "host": MOODLE_DB_HOST,
    "user": MOODLE_DB_USER,
    "password": MOODLE_DB_PASS,
    "database": MOODLE_DB_NAME,
}

# Diccionario que organiza tablas de Moodle en categorías
tables_by_group = {
    "Analytics": [
        "mdlvf_analytics_indicator_calc",
        "mdlvf_analytics_models",
        "mdlvf_analytics_predict_samples",
        "mdlvf_analytics_prediction_actions",
        "mdlvf_analytics_predictions",
        "mdlvf_analytics_used_analysables",
        "mdlvf_analytics_used_files",
    ],
    "Assignments": [
        "mdlvf_assign",
        "mdlvf_assign_grades",
        "mdlvf_assign_plugin_config",
        "mdlvf_assign_submission",
        "mdlvf_assign_user_flags",
        "mdlvf_assign_user_mapping",
        "mdlvf_assignfeedback_comments",
        "mdlvf_assignfeedback_editpdf_annot",
        "mdlvf_assignfeedback_editpdf_cmnt",
        "mdlvf_assignfeedback_editpdf_queue",
        "mdlvf_assignfeedback_editpdf_quick",
        "mdlvf_assignfeedback_editpdf_rot",
        "mdlvf_assignsubmission_file",
        "mdlvf_assignsubmission_onlinetext",
    ],
    "Backup": [
        "mdlvf_backup_controllers",
        "mdlvf_backup_logs",
    ],
    "Badges": [
        "mdlvf_badge_external_backpack",
    ],
    "Blocks": [
        "mdlvf_block",
        "mdlvf_block_instances",
        "mdlvf_block_positions",
        "mdlvf_block_recent_activity",
        "mdlvf_block_recentlyaccesseditems",
    ],
    "Book": [
        "mdlvf_book",
        "mdlvf_book_chapters",
    ],
    "Chat": [
        "mdlvf_chat",
        "mdlvf_chat_messages",
    ],
    "Choice": [
        "mdlvf_choice",
        "mdlvf_choice_answers",
        "mdlvf_choice_options",
    ],
    "Comments": [
        "mdlvf_comments",
    ],
    "Content": [
        "mdlvf_bootstrapelements",
        "mdlvf_files",
        "mdlvf_folder",
        "mdlvf_glossary",
        "mdlvf_glossary_formats",
        "mdlvf_label",
        "mdlvf_url",
    ],
    "Course": [
        "mdlvf_course",
        "mdlvf_course_categories",
        "mdlvf_course_completion_defaults",
        "mdlvf_course_completions",
        "mdlvf_course_format_options",
        "mdlvf_course_modules",
        "mdlvf_course_modules_completion",
        "mdlvf_course_sections",
        "mdlvf_event",
        "mdlvf_grade_categories",
        "mdlvf_grade_categories_history",
        "mdlvf_grade_grades",
        "mdlvf_grade_grades_history",
        "mdlvf_grade_items",
        "mdlvf_grade_items_history",
        "mdlvf_grade_settings",
        "mdlvf_resource",
        "mdlvf_tool_dataprivacy_request",
        "mdlvf_tool_recyclebin_course",
        "mdlvf_tool_usertours_steps",
        "mdlvf_tool_usertours_tours",
    ],
    "Course Formats": [
        "mdlvf_format_grid_icon",
        "mdlvf_format_grid_summary",
        "mdlvf_modules",
    ],
    "Enrollment": [
        "mdlvf_enrol",
        "mdlvf_user_enrolments",
    ],
    "External Services": [
        "mdlvf_external_functions",
        "mdlvf_external_services",
        "mdlvf_external_services_functions",
        "mdlvf_external_services_users",
        "mdlvf_external_tokens",
        "mdlvf_repository",
        "mdlvf_repository_instances",
    ],
    "Feedback": [
        "mdlvf_feedback",
        "mdlvf_feedback_completed",
        "mdlvf_feedback_item",
        "mdlvf_feedback_value",
    ],
    "Filters": [
        "mdlvf_filter_active",
    ],
    "Forum": [
        "mdlvf_forum",
        "mdlvf_forum_discussion_subs",
        "mdlvf_forum_discussions",
        "mdlvf_forum_grades",
        "mdlvf_forum_posts",
        "mdlvf_forum_subscriptions",
    ],
    "Grading": [
        "mdlvf_grading_areas",
        "mdlvf_grading_definitions",
        "mdlvf_grading_instances",
        "mdlvf_gradingform_rubric_criteria",
        "mdlvf_gradingform_rubric_fillings",
        "mdlvf_gradingform_rubric_levels",
        "mdlvf_rating",
        "mdlvf_scale",
    ],
    "h5": [
        "mdlvf_h5p_libraries",
        "mdlvf_h5p_library_dependencies",
        "mdlvf_hvp",
        "mdlvf_hvp_content_hub_cache",
        "mdlvf_hvp_content_user_data",
        "mdlvf_hvp_contents_libraries",
        "mdlvf_hvp_counters",
        "mdlvf_hvp_events",
        "mdlvf_hvp_libraries",
        "mdlvf_hvp_libraries_cachedassets",
        "mdlvf_hvp_libraries_hub_cache",
        "mdlvf_hvp_libraries_languages",
        "mdlvf_hvp_libraries_libraries",
        "mdlvf_hvp_tmpfiles",
        "mdlvf_hvp_xapi_results",
    ],
    "Lesson": [
        "mdlvf_lesson",
    ],
    "License": [
        "mdlvf_license",
    ],
    "Log": [
        "mdlvf_log_display",
        "mdlvf_logstore_standard_log",
        "mdlvf_upgrade_log",
    ],
    "LTI": [
        "mdlvf_lti",
        "mdlvf_lti_submission",
    ],
    "Messages": [
        "mdlvf_message",
        "mdlvf_message_contact_requests",
        "mdlvf_message_contacts",
        "mdlvf_message_conversation_actions",
        "mdlvf_message_conversation_members",
        "mdlvf_message_conversations",
        "mdlvf_message_popup",
        "mdlvf_message_popup_notifications",
        "mdlvf_message_processors",
        "mdlvf_message_providers",
        "mdlvf_message_read",
        "mdlvf_message_user_actions",
        "mdlvf_message_users_blocked",
        "mdlvf_messageinbound_handlers",
        "mdlvf_messages",
    ],
    "Mnet": [
        "mdlvf_mnet_application",
        "mdlvf_mnet_host",
        "mdlvf_mnet_remote_rpc",
        "mdlvf_mnet_remote_service2rpc",
        "mdlvf_mnet_rpc",
        "mdlvf_mnet_service",
        "mdlvf_mnet_service2rpc",
    ],
    "Notifications": [
        "mdlvf_notifications",
        "mdlvf_registration_hubs",
    ],
    "Page": [
        "mdlvf_page",
    ],
    "Questions": [
        "mdlvf_qtype_ddimageortext",
        "mdlvf_qtype_ddimageortext_drags",
        "mdlvf_qtype_ddimageortext_drops",
        "mdlvf_qtype_ddmarker",
        "mdlvf_qtype_ddmarker_drags",
        "mdlvf_qtype_essay_options",
        "mdlvf_qtype_match_options",
        "mdlvf_qtype_match_subquestions",
        "mdlvf_qtype_multichoice_options",
        "mdlvf_qtype_shortanswer_options",
        "mdlvf_question",
        "mdlvf_question_answers",
        "mdlvf_question_attempt_step_data",
        "mdlvf_question_attempt_steps",
        "mdlvf_question_attempts",
        "mdlvf_question_categories",
        "mdlvf_question_ddwtos",
        "mdlvf_question_gapselect",
        "mdlvf_question_hints",
        "mdlvf_question_numerical",
        "mdlvf_question_numerical_options",
        "mdlvf_question_truefalse",
        "mdlvf_question_usages",
    ],
    "Quiz": [
        "mdlvf_quiz",
        "mdlvf_quiz_attempts",
        "mdlvf_quiz_feedback",
        "mdlvf_quiz_grades",
        "mdlvf_quiz_reports",
        "mdlvf_quiz_sections",
        "mdlvf_quiz_slots",
    ],
    "Survey": [
        "mdlvf_survey",
        "mdlvf_survey_questions",
    ],
    "System": [
        "mdlvf_cache_flags",
        "mdlvf_capabilities",
        "mdlvf_config",
        "mdlvf_config_log",
        "mdlvf_config_plugins",
        "mdlvf_context",
    ],
    "Tags": [
        "mdlvf_tag",
        "mdlvf_tag_area",
        "mdlvf_tag_coll",
        "mdlvf_tag_instance",
    ],
    "Task": [
        "mdlvf_task_adhoc",
        "mdlvf_task_log",
        "mdlvf_task_scheduled",
    ],
    "Users": [
        "mdlvf_cohort",
        "mdlvf_cohort_members",
        "mdlvf_favourite",
        "mdlvf_groups",
        "mdlvf_my_pages",
        "mdlvf_role",
        "mdlvf_role_allow_assign",
        "mdlvf_role_allow_override",
        "mdlvf_role_allow_switch",
        "mdlvf_role_allow_view",
        "mdlvf_role_assignments",
        "mdlvf_role_capabilities",
        "mdlvf_role_context_levels",
        "mdlvf_sessions",
        "mdlvf_user",
        "mdlvf_user_devices",
        "mdlvf_user_info_category",
        "mdlvf_user_info_data",
        "mdlvf_user_info_field",
        "mdlvf_user_lastaccess",
        "mdlvf_user_preferences",
        "mdlvf_user_private_key",
    ],
    "Workshop": [
        "mdlvf_workshop",
    ],
}

# Carpeta base de salida
output_base_dir = "data/processed/parquets"

# Crear la carpeta base si no existe
os.makedirs(output_base_dir, exist_ok=True)

try:
    conn = pymysql.connect(**db_config)
    print("Conexión exitosa a la base de datos.")

    with conn.cursor() as cursor:
        for group, tables in tables_by_group.items():
            # Crear directorio para el grupo
            group_dir = os.path.join(output_base_dir, group)
            os.makedirs(group_dir, exist_ok=True)

            for table in tables:
                print(f"Cargando tabla: {table}")

                # Leer la tabla
                cursor.execute(f"SELECT * FROM {table}")
                rows = cursor.fetchall()
                columns = [col[0] for col in cursor.description]

                # Convertir a PyArrow Table
                arrow_table = pa.Table.from_pydict({col: [row[i] for row in rows] for i, col in enumerate(columns)})

                # Ruta del archivo Parquet
                output_file = os.path.join(group_dir, f"{table}.parquet")

                # Exportar la tabla a Parquet
                print(f"Exportando tabla {table} a {output_file}...")
                pq.write_table(arrow_table, output_file, compression="snappy")
                print(f"Tabla {table} exportada exitosamente.")

except pymysql.MySQLError as e:
    print(f"Error conectando a la base de datos: {e}")

finally:
    if conn:
        conn.close()
        print("Conexión cerrada.")
