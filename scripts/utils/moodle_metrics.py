# Importar librerías
import duckdb
import pandas as pd


class MoodleMetrics:
    def __init__(self):
        """
        Constructor de la clase MoodleMetrics.
        """
        self.logs_file = "metabase-project/data/parquets/Log/mdlvf_logstore_standard_log.parquet"
        self.activities_file = "metabase-project/data/parquets/Generated/student_course_activities.parquet"
        self.con = duckdb.connect()

    def calculate_common_metrics(self, activity_type):
        """
        Calcula métricas comunes para una actividad específica.

        Parámetros:
        activity_type (str): Tipo de actividad a analizar (ej. 'assign', 'quiz', 'forum').

        Retorna:
        pd.DataFrame: DataFrame con métricas comunes calculadas.
        """
        sql = f"""
            SELECT
                a.userid,
                a.course_id,
                a.section_id,
                a.module_id,
                a.instance,
                '{activity_type}' AS activity_type,
                COUNT(log.id) AS total_interactions,
                MIN(log.timecreated) AS first_interaction,
                MAX(log.timecreated) AS last_interaction,
                MAX(log.timecreated) - MIN(log.timecreated) AS total_time_spent
            FROM '{self.activities_file}' a
            JOIN '{self.logs_file}' log
                ON a.module_id = log.contextinstanceid
                AND a.userid = log.userid
            WHERE a.activity_type = '{activity_type}'
            GROUP BY
                a.userid, a.course_id, a.section_id, a.module_id, a.instance
        """
        return self.con.execute(sql).df()
