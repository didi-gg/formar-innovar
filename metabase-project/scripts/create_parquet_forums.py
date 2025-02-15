# Importar librerías
import duckdb
import pandas as pd
import os
import sys

# Agregar el directorio raíz al path para importar MoodleMetrics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.moodle_metrics import MoodleMetrics

# Archivos Parquet de entrada
activities_file = "metabase-project/data/parquets/Generated/student_course_activities.parquet"
forums_file = "metabase-project/data/parquets/Forum/mdlvf_forum.parquet"
forum_posts_file = "metabase-project/data/parquets/Forum/mdlvf_forum_posts.parquet"
output_file = "metabase-project/data/parquets/Generated/metrics_forums.parquet"

# Conexión a DuckDB
con = duckdb.connect()


# Métricas específicas por tipo de actividad
def calculate_forum_metrics_sql():
    """
    Calcula métricas de foros:
    - Total de publicaciones
    - Número de hilos nuevos creados
    - Número de respuestas realizadas
    """
    sql = f"""
        SELECT
            a.userid,
            a.course_id,
            a.section_id,
            a.activity_id,
            a.module_id,
            a.instance,
            'forum' AS activity_type,
            COUNT(fp.id) AS total_posts,
            COUNT(CASE WHEN fp.parent = 0 THEN 1 ELSE NULL END) AS new_threads,
            COUNT(CASE WHEN fp.parent > 0 THEN 1 ELSE NULL END) AS replies
        FROM '{activities_file}' a
        LEFT JOIN '{forums_file}' f
            ON a.instance = f.id
        LEFT JOIN 'metabase-project/data/parquets/Forum/mdlvf_forum_discussions.parquet' fd
            ON f.id = fd.forum
        LEFT JOIN '{forum_posts_file}' fp
            ON fd.id = fp.discussion
        AND a.userid = fp.userid
        WHERE a.activity_type = 'forum'
        GROUP BY a.userid, a.course_id, a.section_id, a.activity_id, a.module_id, a.instance
    """
    return con.execute(sql).df()


# Generar métricas para todas las actividades
def generate_all_metrics():
    # Inicializar generador de métricas comunes
    metrics_generator = MoodleMetrics()

    common_metrics = metrics_generator.calculate_common_metrics("forum")

    # Generar métricas específicas
    specific_metrics = calculate_forum_metrics_sql()

    # Combinar ambas métricas
    merged_metrics = pd.merge(
        common_metrics,
        specific_metrics,
        on=[
            "userid",
            "course_id",
            "section_id",
            "activity_id",
            "module_id",
            "instance",
        ],
        how="left",
        suffixes=("_common", "_specific"),
    )

    # Identificar y eliminar columnas duplicadas
    columns_to_drop = [col for col in merged_metrics.columns if col.endswith("_specific")]
    merged_metrics.drop(columns=columns_to_drop, inplace=True)

    # Renombrar las columnas "_common" para quitar el sufijo
    merged_metrics.columns = [col.replace("_common", "") if "_common" in col else col for col in merged_metrics.columns]

    # Guardar en Parquet
    merged_metrics.to_parquet(output_file, index=False)
    print("Métricas de foros generadas y guardadas correctamente.")


# Ejecutar el script
generate_all_metrics()
