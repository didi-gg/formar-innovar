"""
Este script utiliza DuckDB y Pandas para calcular métricas de interacción de estudiantes con actividades en Moodle, con un enfoque en tareas (assignments)

1. Extrae las tareas asignadas a los estudiantes y sus respectivas entregas
2. Analiza la interacción de los estudiantes con las tareas
3. Determina el estado de entrega (On Time, Late, Not Submitted)
4. Combina métricas generales y específicas en un archivo Parquet
"""

# Importar librerías
import duckdb
import pandas as pd
import os
import sys

# Agregar el directorio raíz al path para importar MoodleMetrics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.moodle_metrics import MoodleMetrics

# Archivos Parquet de entrada
activities_file = "../../data/processed/parquets/Generated/student_course_activities.parquet"
assignments_file = "../../data/processed/parquets/Assignments/mdlvf_assign.parquet"
submissions_file = "../../data/processed/parquets/Assignments/mdlvf_assign_submission.parquet"
output_file = "../../data/processed/parquets/Generated/metrics_assignments.parquet"

# Conexión a DuckDB
con = duckdb.connect()


def calculate_assignment_metrics_sql():
    """
    Obtiene todas las tareas (assign) en las que los estudiantes han interactuado.
    Clasifica las entregas:
        "On Time" → Entregadas antes de la fecha límite.
        "Late" → Entregadas después de la fecha límite.
        "Not Submitted" → No entregadas.
    Detecta si el estudiante empezó la tarea (started_working → 1 si inició, 0 si no).
    Agrupa los datos por estudiante y actividad.
    """
    sql = f"""
        SELECT
            a.userid,
            a.course_id,
            a.section_id,
            a.module_id,
            a.instance,
            'assign' AS activity_type,
            CASE
                WHEN MAX(s.status = 'submitted' AND s.timemodified <= assign.duedate) = 1 THEN 'On Time'
                WHEN MAX(s.status IN ('submitted', 'new') AND s.timemodified > assign.duedate) = 1 THEN 'Late'
                ELSE 'Not Submitted'
            END AS submission_status,
            MAX(CASE WHEN s.status = 'new' THEN 1 ELSE 0 END) AS started_working
        FROM '{activities_file}' a
        LEFT JOIN '{assignments_file}' assign ON a.instance = assign.id
        LEFT JOIN '{submissions_file}' s
            ON assign.id = s.assignment
            AND a.userid = s.userid
        WHERE a.activity_type = 'assign'
        GROUP BY
            a.userid, a.course_id, a.section_id, a.module_id, a.instance
    """
    return con.execute(sql).df()


# Generar y combinar métricas para tareas
def generate_assignment_metrics():
    # Métricas comunes para tareas
    metrics_generator = MoodleMetrics()
    assign_common_metrics = metrics_generator.calculate_common_metrics("assign")

    # Métricas específicas para tareas
    assign_specific_metrics = calculate_assignment_metrics_sql()

    # Combinar ambas métricas con sufijos personalizados
    assign_metrics = pd.merge(
        assign_common_metrics,
        assign_specific_metrics,
        on=[
            "userid",
            "course_id",
            "section_id",
            "module_id",
            "instance",
        ],
        how="left",
        suffixes=("_common", "_specific"),
    )

    # Identificar y eliminar columnas duplicadas
    columns_to_drop = [col for col in assign_metrics.columns if col.endswith("_specific")]
    assign_metrics.drop(columns=columns_to_drop, inplace=True)

    # Renombrar las columnas "_common" para quitar el sufijo
    assign_metrics.columns = [col.replace("_common", "") if "_common" in col else col for col in assign_metrics.columns]

    # Guardar como archivo Parquet (opcional)
    assign_metrics.to_parquet(output_file, index=False)
    print("Métricas de tareas generadas correctamente.")


if __name__ == "__main__":
    generate_assignment_metrics()
