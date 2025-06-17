"""
Este script extrae datos de una base de datos MySQL de Moodle y los convierte en archivos Parquet organizados por categorías.
"""

import pymysql
import pyarrow as pa
import pyarrow.parquet as pq
import os
import dotenv
from utils import MOODLE_TABLES_GROUPED

# Carga de variables de entorno
dotenv.load_dotenv()

MOODLE_DB_HOST = os.getenv("MOODLE_DB_HOST")
MOODLE_DB_USER = os.getenv("MOODLE_DB_USER")
MOODLE_DB_PASS = os.getenv("MOODLE_DB_PASS")
MOODLE_DB_NAME = os.getenv("MOODLE_DB_NAME")

# Configuración de la conexión a MySQL
db_config = {
    "host": MOODLE_DB_HOST,
    "user": MOODLE_DB_USER,
    "password": MOODLE_DB_PASS,
    "database": MOODLE_DB_NAME,
}

# Diccionario que organiza tablas de Moodle en categorías
tables_by_group = MOODLE_TABLES_GROUPED

# Carpeta base de salida
output_base_dir = "data/interim/moodle"

# Crear la carpeta base si no existe
os.makedirs(output_base_dir, exist_ok=True)

try:
    conn = pymysql.connect(**db_config)
    print("Conexión exitosa a la base de datos.")

    with conn.cursor() as cursor:
        for group, tables in tables_by_group.items():
            # Crear directorio para el grupo
            group_dir = os.path.join(output_base_dir, group)
            os.makedirs(group_dir, exist_ok=True)

            for table in tables:
                print(f"Cargando tabla: {table}")

                # Leer la tabla
                cursor.execute(f"SELECT * FROM {table}")
                rows = cursor.fetchall()
                columns = [col[0] for col in cursor.description]

                # Convertir a PyArrow Table
                arrow_table = pa.Table.from_pydict({col: [row[i] for row in rows] for i, col in enumerate(columns)})

                # Ruta del archivo Parquet
                output_file = os.path.join(group_dir, f"{table}.parquet")

                # Exportar la tabla a Parquet
                print(f"Exportando tabla {table} a {output_file}...")
                pq.write_table(arrow_table, output_file, compression="snappy")
                print(f"Tabla {table} exportada exitosamente.")

except pymysql.MySQLError as e:
    print(f"Error conectando a la base de datos: {e}")

finally:
    if conn:
        conn.close()
        print("Conexión cerrada.")
