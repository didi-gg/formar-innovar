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
resources_file = "metabase-project/data/parquets/Course/mdlvf_resource.parquet"
books_file = "metabase-project/data/parquets/Book/mdlvf_book.parquet"
urls_file = "metabase-project/data/parquets/Content/mdlvf_url.parquet"
logs_file = "metabase-project/data/parquets/Log/mdlvf_logstore_standard_log.parquet"

# Archivo de salida
output_file = "metabase-project/data/parquets/Generated/metrics_resources.parquet"

# Conexión a DuckDB
con = duckdb.connect()


def calculate_resource_metrics_sql():
    """
    Calcula métricas de recursos:
    - Total de veces que el recurso fue visto
    """
    sql = f"""
        SELECT
            a.userid,
            a.course_id,
            a.section_id,
            a.activity_id,
            a.module_id,
            a.instance,
            'resource' AS activity_type,
            COUNT(log.id) AS total_views
        FROM '{activities_file}' a
        LEFT JOIN '{resources_file}' r
            ON a.instance = r.id
        LEFT JOIN '{logs_file}' log
            ON r.id = log.contextinstanceid
            AND a.userid = log.userid
        WHERE a.activity_type = 'resource'
        GROUP BY a.userid, a.course_id, a.section_id, a.activity_id, a.module_id, a.instance
    """
    return con.execute(sql).df()


def calculate_book_metrics_sql():
    """
    Calcula métricas de libros:
    - Total de veces que el libro fue abierto
    """
    sql = f"""
        SELECT
            a.userid,
            a.course_id,
            a.section_id,
            a.activity_id,
            a.module_id,
            a.instance,
            'book' AS activity_type,
            COUNT(log.id) AS total_views
        FROM '{activities_file}' a
        LEFT JOIN '{books_file}' b
            ON a.instance = b.id
        LEFT JOIN '{logs_file}' log
            ON b.id = log.contextinstanceid
            AND a.userid = log.userid
        WHERE a.activity_type = 'book'
        GROUP BY a.userid, a.course_id, a.section_id, a.activity_id, a.module_id, a.instance
    """
    return con.execute(sql).df()


def calculate_url_metrics_sql():
    """
    Calcula métricas de URLs:
    - Total de veces que se hizo clic en el enlace
    """
    sql = f"""
        SELECT
            a.userid,
            a.course_id,
            a.section_id,
            a.activity_id,
            a.module_id,
            a.instance,
            'url' AS activity_type,
            COUNT(log.id) AS total_clicks
        FROM '{activities_file}' a
        LEFT JOIN '{urls_file}' u
            ON a.instance = u.id
        LEFT JOIN '{logs_file}' log
            ON u.id = log.contextinstanceid
            AND a.userid = log.userid
        WHERE a.activity_type = 'url'
        GROUP BY a.userid, a.course_id, a.section_id, a.activity_id, a.module_id, a.instance
    """
    return con.execute(sql).df()


# Generar métricas para todas las actividades resources
def generate_all_metrics():
    # Inicializar generador de métricas comunes
    metrics_generator = MoodleMetrics()

    all_metrics = []
    for activity_type, metric_function in {
        "resource": calculate_resource_metrics_sql,
        "book": calculate_book_metrics_sql,
        "url": calculate_url_metrics_sql,
    }.items():
        # Generar métricas comunes
        common_metrics = metrics_generator.calculate_common_metrics(activity_type)

        # Generar métricas específicas
        specific_metrics = metric_function()

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

        all_metrics.append(merged_metrics)

    # Combinar todas las métricas en un solo DataFrame
    final_metrics = pd.concat(all_metrics, ignore_index=True)

    # Guardar en Parquet
    final_metrics.to_parquet(output_file, index=False)
    print("Métricas de resources generadas y guardadas correctamente.")


# Ejecutar el script
generate_all_metrics()
