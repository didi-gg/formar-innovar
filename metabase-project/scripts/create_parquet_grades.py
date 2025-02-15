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
grades_file = "metabase-project/data/parquets/Course/mdlvf_grade_grades.parquet"

# Archivo de salida
output_file = "metabase-project/data/parquets/Generated/metrics_grades.parquet"

# Conexión a DuckDB
con = duckdb.connect()


def calculate_grades_metrics_sql():
    """
    Calcula métricas de notas:
    - Nota promedio obtenida en el curso
    """
    sql = f"""
        SELECT
            a.userid,
            a.course_id,
            a.section_id,
            a.activity_id,
            a.module_id,
            a.instance,
            'grades' AS activity_type,
            AVG(gg.finalgrade) AS average_grade
        FROM '{activities_file}' a
        LEFT JOIN '{grades_file}' gg
            ON a.userid = gg.userid
            AND a.activity_id = gg.itemid
        WHERE a.activity_type = 'grades'
        GROUP BY a.userid, a.course_id, a.section_id, a.activity_id, a.module_id, a.instance
    """
    return con.execute(sql).df()


# Generar métricas para todas las actividades
def generate_all_metrics():
    # Inicializar generador de métricas comunes
    metrics_generator = MoodleMetrics()
    common_metrics = metrics_generator.calculate_common_metrics("grades")
    specific_metrics = calculate_grades_metrics_sql()

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
    print("Métricas de notas generadas correctamente.")


# Ejecutar el script
generate_all_metrics()
