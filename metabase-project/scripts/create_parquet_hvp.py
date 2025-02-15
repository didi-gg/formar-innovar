import duckdb
import pandas as pd
import os
import sys

# Agregar el directorio raíz al path para importar MoodleMetrics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.moodle_metrics import MoodleMetrics

# Archivos Parquet de entrada
logs_file = "metabase-project/data/parquets/Log/mdlvf_logstore_standard_log.parquet"
hvp_file = "metabase-project/data/parquets/h5/mdlvf_hvp.parquet"
activities_file = "metabase-project/data/parquets/Generated/student_course_activities.parquet"
output_file = "metabase-project/data/parquets/Generated/metrics_hvp.parquet"

# Conexión a DuckDB
con = duckdb.connect()


# Métricas específicas para HVP
def calculate_hvp_metrics_sql():
    sql = f"""
        SELECT
            a.userid,
            a.course_id,
            a.section_id,
            a.activity_id,
            a.instance,
            'hvp' AS activity_type,
            COUNT(log.id) AS total_interactions, -- Total de interacciones registradas
            CASE 
                WHEN COUNT(log.id) > 0 THEN 'Started'
                ELSE 'Not Started'
            END AS hvp_status, -- Estado del HVP
            MIN(log.timecreated) AS first_interaction, -- Primera interacción
            MAX(log.timecreated) AS last_interaction,  -- Última interacción
            MAX(log.timecreated) - MIN(log.timecreated) AS total_time_spent -- Tiempo total
        FROM '{activities_file}' a
        LEFT JOIN '{hvp_file}' hvp
            ON a.instance = hvp.id
        LEFT JOIN '{logs_file}' log
            ON hvp.id = log.contextinstanceid
            AND a.userid = log.userid
        WHERE a.activity_type = 'hvp'
        GROUP BY
            a.userid, a.course_id, a.section_id, a.activity_id, a.instance
    """
    return con.execute(sql).df()


# Generar y combinar métricas para HVP
def generate_hvp_metrics():
    # Métricas comunes para HVP
    metrics_generator = MoodleMetrics()
    hvp_common_metrics = metrics_generator.calculate_common_metrics("hvp")

    # Métricas específicas para HVP
    hvp_specific_metrics = calculate_hvp_metrics_sql()

    # Combinar ambas métricas
    hvp_metrics = pd.merge(
        hvp_common_metrics,
        hvp_specific_metrics,
        on=["userid", "course_id", "section_id", "activity_id", "instance", "activity_type"],
        how="left",
        suffixes=("_common", "_specific"),  # Evitar duplicados
    )

    # Guardar como archivo Parquet (opcional)
    hvp_metrics.to_parquet(output_file, index=False)

    print("Métricas de HVP generadas y guardadas correctamente.")

# Ejecutar el script
generate_hvp_metrics()
