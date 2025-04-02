"""
Este script crea un archivo Parquet con la información de las actividades de los cursos de los estudiantes.
"""

# Importar librerías
import duckdb
import pandas as pd

# Definir rutas de archivos Parquet
activities_file = "../../data/processed/parquets/Course/activities_section_mapping.parquet"
student_courses_file = "../../data/processed/parquets/Generated/student_courses.parquet"
modules_file = "../../data/processed/parquets/Course/mdlvf_course_modules.parquet"
module_names_file = "../../data/processed/parquets/Course Formats/mdlvf_modules.parquet"
output_file = "../../data/processed/parquets/Generated/student_course_activities.parquet"

# Conectar a DuckDB
con = duckdb.connect()


def create_parquet_student_course_activities():
    # Definir la consulta SQL
    student_course_activities_sql = f"""
        SELECT
            sc.userid,
            a.course_id,
            a.section_id,
            m.id AS module_id,
            mod.name AS activity_type,
            m.instance AS instance
        FROM '{activities_file}' a
        JOIN '{student_courses_file}' sc ON a.course_id = sc.course_id
        LEFT JOIN '{modules_file}' m ON a.module_id = m.id
        LEFT JOIN '{module_names_file}' mod ON m.module = mod.id
    """

    # Ejecutar la consulta y guardar resultados
    student_course_activities_df = con.execute(student_course_activities_sql).df()
    student_course_activities_df.to_parquet(output_file, index=False)

    print(f"Archivo Parquet generado correctamente en: {output_file}")


if __name__ == "__main__":
    create_parquet_student_course_activities()
