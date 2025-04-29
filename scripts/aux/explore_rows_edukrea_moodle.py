import pandas as pd
import pymysql
import os

import dotenv

dotenv.load_dotenv()

# Conexión a la base de datos
MOODLE_EDUKREA_DB_HOST = os.getenv("MOODLE_EDUKREA_DB_HOST")
MOODLE_EDUKREA_DB_USER = os.getenv("MOODLE_EDUKREA_DB_USER")
MOODLE_EDUKREA_DB_PASS = os.getenv("MOODLE_EDUKREA_DB_PASS")
MOODLE_EDUKREA_DB_NAME = os.getenv("MOODLE_EDUKREA_DB_NAME")

db_config = {"host": MOODLE_EDUKREA_DB_HOST, "user": MOODLE_EDUKREA_DB_USER, "password": MOODLE_EDUKREA_DB_PASS, "database": MOODLE_EDUKREA_DB_NAME}

try:
    conn = pymysql.connect(**db_config)
    print("Conexión exitosa a la base de datos.")
except pymysql.MySQLError as e:
    print(f"Error conectándose a la base de datos: {e}")
    exit()

try:
    # Conectar a la base de datos
    conn = pymysql.connect(**db_config)
    results = []

    # Obtener la lista de tablas
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'brkaiups_edukrea';
        """)
        tables = cursor.fetchall()

        # Iterar sobre las tablas y contar filas
        for (table_name,) in tables:
            with conn.cursor() as cursor:
                sql = f"SELECT COUNT(*) FROM {table_name}"
                try:
                    cursor.execute(sql)
                    row_count = cursor.fetchone()[0]
                    group = ""
                    results.append({"table_name": table_name, "row_count": row_count, "group": group})
                    print(f"Tabla: {table_name}, Número de filas: {row_count}, Grupo: {group}")
                except pymysql.MySQLError as e:
                    print(f"Error al contar filas en la tabla {table_name}: {e}")
                    continue

    # Crear un DataFrame con los resultados
    df = pd.DataFrame(results)
    print("\nConteo de filas por tabla completado:")
    print(df)

    # Guardar los resultados en un archivo CSV
    df.to_csv("scripts_aux/tmp/conteo_filas_tablas_edukrea.csv", index=False)
    print("Resultados guardados en 'scripts_aux/tmp/conteo_filas_tablas_edukrea.csv'.")

except pymysql.MySQLError as e:
    print(f"Error conectándose a la base de datos o ejecutando consultas: Table Name: {table_name} {e}")
finally:
    # Cerrar la conexión
    conn.close()
    print("Conexión cerrada.")
