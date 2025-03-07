"""
Este script crea un archivo Parquet con la relación entre los estudiantes y los cursos en los que están inscritos.
"""

# Importar librerías
import duckdb
import pandas as pd

# Definir rutas de los archivos Parquet
enrolments_file = "../../data/processed/parquets/Enrollment/mdlvf_user_enrolments.parquet"
enrol_file = "../../data/processed/parquets/Enrollment/mdlvf_enrol.parquet"
students_file = "../../data/processed/parquets/Generated/students.parquet"
output_file = "../../data/processed/parquets/Generated/student_courses.parquet"

# Conectar a DuckDB
con = duckdb.connect()


def create_parquet_student_courses():
    # Definir la consulta SQL
    sql = f"""
        SELECT DISTINCT
            ue.userid,
            e.courseid AS course_id
        FROM '{enrolments_file}' ue
        JOIN '{enrol_file}' e ON ue.enrolid = e.id
        JOIN '{students_file}' s ON ue.userid = s.UserID
    """

    # Ejecutar la consulta y convertir a DataFrame
    result_df = con.execute(sql).df()

    # Guardar los resultados como un archivo Parquet
    result_df.to_parquet(output_file, index=False)
    print(f"Archivo Parquet generado correctamente en: {output_file}")


if __name__ == "__main__":
    create_parquet_student_courses()
