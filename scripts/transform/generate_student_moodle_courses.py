"""
Este script crea un archivo CSV con la relación entre los estudiantes y los cursos en los que están inscritos,
incluyendo el nombre del curso e identificador de asignatura (id_asignatura), para los años 2024 y 2025.

Se excluyen cursos que no son relevantes para el análisis académico:
- Cursos institucionales y de prueba: (549, 550, 332)
- Cursos de Inteligencia Emocional sin calificación: (154, 386, 411, 515, 155, 390, 394, 156, 157, 398, 158, 402, 159, 416, 160, 418, 213, 502, 409, 565)
"""

# Importar librerías
import duckdb
import pandas as pd

# Definir rutas de los archivos Parquet
enrollments_file_2024 = "data/raw/moodle/2024/Enrollment/mdlvf_user_enrolments.parquet"
enrol_file_2024 = "data/raw/moodle/2024/Enrollment/mdlvf_enrol.parquet"
courses_file_2024 = "data/raw/moodle/2024/Course/mdlvf_course.parquet"

enrollments_file_2025 = "data/raw/moodle/2025/Enrollment/mdlvf_user_enrolments.parquet"
enrol_file_2025 = "data/raw/moodle/2025/Enrollment/mdlvf_enrol.parquet"
courses_file_2025 = "data/raw/moodle/2025/Course/mdlvf_course.parquet"

students_file = "data/interim/estudiantes/enrollments.csv"

output_file = "data/interim/moodle/student_moodle_courses.csv"

# Conectar a DuckDB
con = duckdb.connect()

# Cursos que se deben excluir
excluded_courses = (
    # Institucionales y prueba
    549,
    550,
    332,
    # Inteligencia Emocional
    154,
    386,
    411,
    515,
    155,
    390,
    394,
    156,
    157,
    398,
    158,
    402,
    159,
    416,
    160,
    418,
    213,
    502,
    409,
    565,
)

# Diccionario de mapeo course_id -> id_asignatura
course_to_subject = {
    # Ciencias Naturales y Educación Ambiental
    18: 1,
    345: 1,
    29: 1,
    430: 1,
    36: 1,
    324: 1,
    43: 1,
    321: 1,
    50: 1,
    312: 1,
    131: 1,
    299: 1,
    146: 1,
    417: 1,
    207: 1,
    491: 1,
    274: 1,
    552: 1,
    # Ciencias Sociales
    19: 2,
    346: 2,
    28: 2,
    331: 2,
    35: 2,
    323: 2,
    42: 2,
    322: 2,
    49: 2,
    313: 2,
    132: 2,
    298: 2,
    145: 2,
    428: 2,
    492: 2,
    208: 2,
    269: 2,
    554: 2,
    # Matemáticas
    17: 3,
    344: 3,
    30: 3,
    333: 3,
    37: 3,
    326: 3,
    264: 3,
    320: 3,
    51: 3,
    310: 3,
    130: 3,
    300: 3,
    147: 3,
    427: 3,
    206: 3,
    493: 3,
    271: 3,
    555: 3,
    # Lengua Castellana
    20: 4,
    343: 4,
    371: 4,
    507: 4,
    27: 4,
    334: 4,
    34: 4,
    325: 4,
    41: 4,
    319: 4,
    48: 4,
    311: 4,
    133: 4,
    301: 4,
    136: 4,
    426: 4,
    211: 4,
    494: 4,
    270: 4,
    556: 4,
    # English
    21: 5,
    341: 5,
    26: 5,
    335: 5,
    33: 5,
    328: 5,
    429: 5,
    356: 5,
    308: 5,
    47: 5,
    134: 5,
    303: 5,
    143: 5,
    425: 5,
    209: 5,
    495: 5,
    359: 5,
    558: 5,
    372: 5,
    508: 5,
    # Creatividad e Innovación
    342: 6,
    174: 6,
    175: 6,
    336: 6,
    176: 6,
    327: 6,
    316: 6,
    177: 6,
    178: 6,
    309: 6,
    # Aprendizaje Basado en Proyectos
    161: 7,
    383: 7,
    162: 7,
    387: 7,
    163: 7,
    391: 7,
    164: 7,
    395: 7,
    165: 7,
    399: 7,
    # Educación Física y Deportes
    23: 9,
    340: 9,
    373: 9,
    511: 9,
    24: 9,
    337: 9,
    31: 9,
    329: 9,
    38: 9,
    315: 9,
    45: 9,
    307: 9,
    135: 9,
    304: 9,
    142: 9,
    422: 9,
    210: 9,
    498: 9,
    268: 9,
    559: 9,
    # Lectura Crítica
    168: 9,
    339: 9,
    374: 9,
    512: 9,
    338: 9,
    169: 9,
    170: 9,
    330: 9,
    187: 9,
    314: 9,
    188: 9,
    306: 9,
    189: 9,
    305: 9,
    190: 9,
    421: 9,
    215: 9,
    499: 9,
    358: 9,
    557: 9,
    # Artes / Centro de Interés Artístico
    239: 10,
    384: 10,
    413: 10,
    513: 10,
    240: 10,
    388: 10,
    241: 10,
    392: 10,
    243: 10,
    396: 10,
    244: 10,
    400: 10,
    245: 10,
    414: 10,
    246: 10,
    420: 10,
    247: 10,
    500: 10,
    407: 10,
    562: 10,
    # Tecnologías Informáticas
    378: 11,
    385: 11,
    412: 11,
    514: 11,
    379: 11,
    389: 11,
    380: 11,
    393: 11,
    381: 11,
    397: 11,
    382: 11,
    401: 11,
    415: 11,
    403: 11,
    404: 11,
    419: 11,
    501: 11,
    405: 11,
    408: 11,
    563: 11,
    # Integralidad
    530: 12,
    519: 12,
    520: 12,
    531: 12,
    521: 12,
    532: 12,
    522: 12,
    523: 12,
    534: 12,
    524: 12,
    535: 12,
    525: 12,
    536: 12,
    526: 12,
    537: 12,
    527: 12,
    564: 12,
    533: 12,
    528: 12,
    529: 12,
    # Innovación y Emprendimiento
    179: 13,
    302: 13,
    234: 13,
    423: 13,
    497: 13,
    235: 13,
    357: 13,
    560: 13,
    # Aprendizaje Basado en Investigación
    361: 14,
    204: 14,
    140: 14,
    424: 14,
    212: 14,
    496: 14,
    406: 14,
    561: 14,
    # Plan de inversión
    377: 15,
    510: 15,
    # Filosofía
    375: 16,
    505: 16,
    # Français
    376: 17,
    509: 17,
    # Trigonometría
    370: 18,
    # Ciencias Políticas y Económicas
    369: 19,
    504: 19,
    # Ciencias Naturales Integradas
    368: 20,
    503: 20,
    # Cálculo
    506: 21,
    # Educación Ambiental
    410: 22,
    # Física
    566: 23,
    567: 23,
}


def load_student_courses(year, enrollments_file, enrol_file, courses_file, students_file):
    sql = f"""
        SELECT DISTINCT
            ue.userid AS moodle_user_id,
            {year} AS year,
            s.id_grado AS id_grado,
            e.courseid AS course_id,
            c.fullname AS course_name
        FROM '{enrollments_file}' ue
        JOIN '{enrol_file}' e ON ue.enrolid = e.id
        JOIN '{courses_file}' c ON e.courseid = c.id
        JOIN '{students_file}' s ON ue.userid = s.moodle_user_id
        WHERE s.year = {year}
            AND e.courseid NOT IN {excluded_courses}
    """
    return con.execute(sql).df()


def create_student_moodle_courses():
    # Procesar 2024
    df_2024 = load_student_courses(2024, enrollments_file_2024, enrol_file_2024, courses_file_2024, students_file)

    # Procesar 2025
    df_2025 = load_student_courses(2025, enrollments_file_2025, enrol_file_2025, courses_file_2025, students_file)

    # Unir los dos años
    df_combined = pd.concat([df_2024, df_2025], ignore_index=True)

    # Mapear id_asignatura
    df_combined["id_asignatura"] = df_combined["course_id"].map(course_to_subject)
    df_combined["id_asignatura"] = df_combined["id_asignatura"].fillna(0).astype(int)

    # Guardar como CSV
    df_combined.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Archivo CSV generado correctamente en: {output_file}")


if __name__ == "__main__":
    create_student_moodle_courses()
