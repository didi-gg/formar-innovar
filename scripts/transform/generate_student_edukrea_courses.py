"""
Este script crea un archivo CSV con la relación entre los estudiantes y los cursos en los que están inscritos,
incluyendo el nombre del curso e identificador de asignatura (id_asignatura), para el año 2025.
"""

# Importar librerías
import duckdb
import pandas as pd

# Definir rutas de los archivos Parquet
enrollments_file_edukrea = "data/raw/moodle/Edukrea/Users/mdl_user_enrolments.parquet"
enrol_file_edukrea = "data/raw/moodle/Edukrea/Other/mdl_enrol.parquet"
courses_file_edukrea = "data/raw/moodle/Edukrea/Courses/mdl_course.parquet"

students_file = "data/interim/estudiantes/enrollments.csv"
mapping_file = "data/interim/moodle/course_edukrea_mapping.csv"

output_file = "data/interim/moodle/student_edukrea_courses.csv"

# Conectar a DuckDB
con = duckdb.connect()

# Cursos que se deben excluir
excluded_courses = ()


def load_student_courses(year, enrollments_file, enrol_file, courses_file):
    base_sql = f"""
        SELECT DISTINCT
            ue.userid AS moodle_user_id,
            {year} AS year,
            s.id_grado AS id_grado,
            e.courseid AS course_id,
            c.fullname AS course_name,
            s.documento_identificación AS documento_identificación,
            s.sede AS sede,
            map.id_asignatura AS id_asignatura
        FROM '{enrollments_file}' ue
        JOIN '{enrol_file}' e ON ue.enrolid = e.id
        JOIN '{courses_file}' c ON e.courseid = c.id
        JOIN '{students_file}' s ON ue.userid = s.edukrea_user_id
        JOIN read_csv_auto('{mapping_file}') AS map ON e.courseid = map.course_id
        WHERE s.year = {year}
            AND c.visible = 1
    """

    # Agregar condición solo si hay cursos excluidos
    if excluded_courses:
        courses_str = ", ".join(str(cid) for cid in excluded_courses)
        base_sql += f" AND e.courseid NOT IN ({courses_str})"

    return con.execute(base_sql).df()


def create_student_moodle_courses():
    # Procesar
    df_2025 = load_student_courses(2025, enrollments_file_edukrea, enrol_file_edukrea, courses_file_edukrea)

    # Guardar como CSV
    df_2025.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Archivo CSV generado correctamente en: {output_file}")


if __name__ == "__main__":
    create_student_moodle_courses()
