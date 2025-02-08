"""
Este script genera métricas de quizzes a partir de los archivos Parquet de Moodle.
"""

# Importar librerías
import duckdb
import pandas as pd
import os
import sys

# Agregar la ruta del proyecto para importar MoodleMetrics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from analytics.moodle_metrics import MoodleMetrics

# Archivos Parquet de entrada
logs_file = "metabase-project/data/parquets/Log/mdlvf_logstore_standard_log.parquet"
activities_file = "metabase-project/data/parquets/Generated/student_course_activities.parquet"
quizzes_file = "metabase-project/data/parquets/Quiz/mdlvf_quiz.parquet"
quiz_attempts_file = "metabase-project/data/parquets/Quiz/mdlvf_quiz_attempts.parquet"
output_file = "metabase-project/data/parquets/Generated/metrics_quizzes_filtered.parquet"

# Conexión a DuckDB
con = duckdb.connect()


def calculate_quiz_metrics_sql():
    """
    Obtiene todas las métricas de los quizzes en los que los estudiantes han interactuado.
    Calcula el total de intentos, el estado del quiz (completado, incompleto o no intentado),
    el promedio de la calificación, la fecha de la primera
    y última interacción, y el tiempo total invertido en el quiz.
    Agrupa los datos por estudiante y actividad.
    """
    sql = f"""
        SELECT
            a.userid,
            a.course_id,
            a.section_id,
            a.activity_id,
            a.instance,
            'quiz' AS activity_type,
            COUNT(qa.id) AS total_attempts,
            CASE 
                WHEN COUNT(qa.id) = 0 THEN 'Not Attempted'
                WHEN MAX(qa.state) = 'finished' THEN 'Completed'
                ELSE 'Incomplete'
            END AS quiz_status,
            AVG(qa.sumgrades) AS average_score,
            MIN(qa.timestart) AS first_attempt,
            MAX(qa.timefinish) AS last_attempt,
            MAX(qa.timefinish) - MIN(qa.timestart) AS quiz_time_spent
        FROM '{activities_file}' a
        LEFT JOIN '{quizzes_file}' quiz ON a.instance = quiz.id
        LEFT JOIN '{quiz_attempts_file}' qa ON quiz.id = qa.quiz AND a.userid = qa.userid
        WHERE a.activity_type = 'quiz'
        GROUP BY a.userid, a.course_id, a.section_id, a.activity_id, a.instance
    """
    return con.execute(sql).df()


def generate_quiz_metrics():
    """
    Genera métricas de quizzes a partir de los archivos Parquet de Moodle.
    """
    # Crear una instancia de MoodleMetrics
    metrics_generator = MoodleMetrics()
    quiz_common_metrics = metrics_generator.calculate_common_metrics("quiz")  # Calcular métricas comunes para quizzes

    # Calcular métricas específicas de quizzes
    quiz_specific_metrics = calculate_quiz_metrics_sql()

    # Unir ambas métricas en un solo DataFrame
    quiz_metrics = pd.merge(
        quiz_common_metrics,
        quiz_specific_metrics,
        on=["userid", "course_id", "section_id", "activity_id", "instance", "activity_type"],
        how="left",
        suffixes=("_common", "_specific"),
    )

    # Eliminar columnas duplicadas después del merge
    columns_to_drop = [col for col in quiz_metrics.columns if col.endswith("_specific")]
    quiz_metrics.drop(columns=columns_to_drop, inplace=True)

    # Renombrar columnas eliminando "_common"
    quiz_metrics.columns = [col.replace("_common", "") if "_common" in col else col for col in quiz_metrics.columns]

    # Guardar como archivo Parquet
    quiz_metrics.to_parquet(output_file, index=False)
    print(f"Métricas de quizzes generadas y guardadas en: {output_file}")


generate_quiz_metrics()
