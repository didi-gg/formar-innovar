"""
Este script toma los datos de secciones de cursos en Moodle y extrae los IDs de actividades contenidos en la columna sequence.
Luego, divide esta columna para generar un mapeo detallado de la relación sección-actividad y lo guarda en un nuevo archivo Parquet.
"""

import duckdb
import pandas as pd

# Configuración de archivos Parquet
sections_file = "data/parquets/Course/mdlvf_course_sections.parquet"
output_file = "data/parquets/Course/activities_section_mapping.parquet"

# Conexión DuckDB
con = duckdb.connect()

# Leer los archivos Parquet
sections_df = con.execute(f"SELECT * FROM '{sections_file}'").df()

# Explode la columna 'sequence' para dividir los IDs de actividades
# Contiene los IDs de actividades de cada sección, almacenados como una cadena separada por comas (ej. "101,102,103").

# 1. Eliminar filas con 'sequence' nulo
sections_df = sections_df.dropna(subset=["sequence"])
# 2. Dividir la columna 'sequence' en una lista de IDs de actividades
sections_df["activity_id"] = sections_df["sequence"].str.split(",")
# 3. Explotar la lista de IDs de actividades en filas separadas
# Esto da como resultado una fila por cada ID de actividad en la lista
sections_exploded = sections_df.explode("activity_id")

# Asegurar que 'activity_id' sea numérico y eliminar nulos
sections_exploded["activity_id"] = pd.to_numeric(
    sections_exploded["activity_id"], errors="coerce"
)

# Eliminar filas con 'activity_id' nulo
sections_exploded = sections_exploded.dropna(subset=["activity_id"])

# Convertir 'activity_id' a entero
sections_exploded["activity_id"] = sections_exploded["activity_id"].astype(int)

# Crear una tabla simplificada para las relaciones
result_df = sections_exploded[
    [
        "course",  # ID del curso
        "id",  # ID de la sección
        "name",  # Nombre de la sección
        "activity_id",  # ID de la actividad
    ]
].rename(columns={"course": "course_id", "id": "section_id", "name": "section_name"})

# Guardar el resultado como un nuevo archivo Parquet
result_df.to_parquet(output_file, index=False)

print(f"Archivo generado correctamente en {output_file}")
