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

MOODLE_EDUKREA_DB_HOST = os.getenv("MOODLE_EDUKREA_DB_HOST")
MOODLE_EDUKREA_DB_USER = os.getenv("MOODLE_EDUKREA_DB_USER")
MOODLE_EDUKREA_DB_PASS = os.getenv("MOODLE_EDUKREA_DB_PASS")
MOODLE_EDUKREA_DB_NAME = os.getenv("MOODLE_EDUKREA_DB_NAME")

# Configuración de la conexión a MySQL
db_config = {
    "host": MOODLE_EDUKREA_DB_HOST,
    "user": MOODLE_EDUKREA_DB_USER,
    "password": MOODLE_EDUKREA_DB_PASS,
    "database": MOODLE_EDUKREA_DB_NAME,
}

# Diccionario que organiza tablas de Moodle en categorías
tables_by_group = {
    "Logs and Events": ["mdl_logstore_standard_log", "mdl_task_log", "mdl_config_log", "mdl_block_xp_log", "mdl_log_display", "mdl_event"],
    "Interactive Content": [
        "mdl_hvp_contents_libraries",
        "mdl_hvp_xapi_results",
        "mdl_hvp_libraries_cachedassets",
        "mdl_hvp",
        "mdl_hvp_libraries_languages",
        "mdl_hvp_events",
        "mdl_hvp_tmpfiles",
        "mdl_h5p_library_dependencies",
        "mdl_hvp_libraries_libraries",
        "mdl_hvp_counters",
        "mdl_h5p_libraries",
        "mdl_hvp_libraries",
        "mdl_h5p_contents_libraries",
        "mdl_hvp_libraries_hub_cache",
        "mdl_hvp_content_user_data",
        "mdl_h5p_libraries_cachedassets",
        "mdl_geogebra_attempts",
        "mdl_h5p",
        "mdl_hvp_content_hub_cache",
        "mdl_geogebra",
    ],
    "Assignments and Grades": [
        "mdl_grade_items_history",
        "mdl_grade_grades_history",
        "mdl_grade_grades",
        "mdl_grade_items",
        "mdl_upgrade_log",
        "mdl_role_assignments",
        "mdl_grade_categories_history",
        "mdl_assign_plugin_config",
        "mdl_assign_submission",
        "mdl_grade_categories",
        "mdl_assign",
        "mdl_assign_user_mapping",
        "mdl_role_allow_assign",
        "mdl_assignsubmission_file",
        "mdl_assign_grades",
        "mdl_assignfeedback_editpdf_rot",
    ],
    "Files and Resources": ["mdl_files", "mdl_contentbank_content", "mdl_resource", "mdl_files_reference"],
    "Other": [
        "innodb_index_stats",
        "mdl_block_recentlyaccesseditems",
        "mdl_tiny_autosave",
        "mdl_backup_controllers",
        "mdl_block_recent_activity",
        "mdl_external_functions",
        "help_topic",
        "mdl_grading_areas",
        "mdl_block_instances",
        "mdl_block_xp_filters",
        "innodb_table_stats",
        "mdl_external_services_functions",
        "mdl_enrol",
        "mdl_question_answers",
        "mdl_question",
        "mdl_question_versions",
        "mdl_block_xp",
        "mdl_task_scheduled",
        "mdl_question_bank_entries",
        "help_relation",
        "mdl_sessions",
        "mdl_cache_flags",
        "mdl_qtype_multichoice_options",
        "mdl_qtype_match_subquestions",
        "mdl_question_truefalse",
        "help_category",
        "mdl_question_categories",
        "mdl_block",
        "help_keyword",
        "mdl_qtype_essay_options",
        "mdl_modules",
        "mdl_favourite",
        "mdl_qtype_match_options",
        "mdl_license",
        "mdl_editor_atto_autosave",
        "mdl_filter_active",
        "mdl_block_positions",
        "global_priv",
        "db",
        "mdl_external_services",
        "proxies_priv",
        "mdl_comments",
        "mdl_badge_external_backpack",
        "mdl_format_menutopic",
        "mdl_registration_hubs",
    ],
    "Access and Roles": [
        "mdl_context",
        "mdl_role_capabilities",
        "mdl_capabilities",
        "mdl_role_allow_view",
        "mdl_mnet_remote_rpc",
        "mdl_mnet_remote_service2rpc",
        "mdl_mnet_service2rpc",
        "mdl_mnet_rpc",
        "mdl_role_allow_override",
        "mdl_role_context_levels",
        "mdl_role_allow_switch",
        "mdl_role",
        "mdl_mnet_service",
        "mdl_mnet_application",
        "mdl_mnet_host",
    ],
    "Courses": [
        "mdl_course_modules_completion",
        "mdl_course_modules",
        "mdl_course_modules_viewed",
        "mdl_course_completions",
        "mdl_course_sections",
        "mdl_tool_recyclebin_course",
        "mdl_course_format_options",
        "mdl_course",
        "mdl_course_completion_criteria",
        "mdl_course_categories",
        "mdl_course_completion_aggr_methd",
        "mdl_course_completion_crit_compl",
    ],
    "Analytics": [
        "mdl_analytics_predict_samples",
        "mdl_analytics_indicator_calc",
        "mdl_analytics_predictions",
        "mdl_analytics_used_analysables",
        "mdl_analytics_models",
        "mdl_analytics_prediction_actions",
    ],
    "Messaging": [
        "mdl_notifications",
        "mdl_message_popup_notifications",
        "mdl_message_providers",
        "mdl_message_conversation_members",
        "mdl_message_conversations",
        "mdl_chat_messages",
        "mdl_chat",
        "mdl_message_processors",
        "mdl_messageinbound_handlers",
    ],
    "Users": [
        "mdl_user_preferences",
        "mdl_user_enrolments",
        "mdl_user_lastaccess",
        "mdl_user",
        "mdl_cohort_members",
        "mdl_user_private_key",
        "mdl_cohort",
        "mdl_tool_usertours_steps",
        "mdl_tool_usertours_tours",
    ],
    "Configuration": ["mdl_config_plugins", "mdl_config", "mdl_adminpresets_plug", "mdl_block_xp_config", "mdl_adminpresets_it", "mdl_adminpresets"],
    "Forums": ["mdl_forum_posts", "mdl_forum_discussion_subs", "mdl_forum", "mdl_forum_discussions", "mdl_forum_subscriptions"],
    "Basic Resources": ["mdl_label", "mdl_my_pages", "mdl_lesson_pages", "mdl_page", "mdl_url"],
    "Activities": [
        "mdl_survey_questions",
        "mdl_lesson_branch",
        "mdl_scorm_scoes_value",
        "mdl_lesson_answers",
        "mdl_scorm_attempt",
        "mdl_glossary_formats",
        "mdl_survey",
        "mdl_scorm_element",
        "mdl_scorm_scoes_data",
        "mdl_scorm_scoes",
        "mdl_lesson",
        "mdl_scorm",
    ],
    "Tools and Reports": [
        "mdl_tool_brickfield_checks",
        "mdl_reportbuilder_report",
        "mdl_tool_htmlbootstrapeditor_tpl",
        "mdl_tool_dataprivacy_request",
    ],
    "Tags": ["mdl_tag_area", "mdl_tag_instance", "mdl_tag", "mdl_tag_coll"],
    "Repositories": ["mdl_repository_instances", "mdl_repository"],
    "Quizzes": ["mdl_quiz_reports"],
    "Scales": ["mdl_scale"],
}

# Carpeta base de salida
output_base_dir = "data/processed/Edukrea"

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
