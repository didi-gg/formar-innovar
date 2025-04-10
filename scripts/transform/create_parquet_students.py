"""
Este script usa DuckDB y Pandas para extraer información de los estudiantes desde archivos Parquet, combinarla con un archivo Excel,
y finalmente guardar los datos procesados en un nuevo archivo Parquet.
"""

# Importar librerías
import duckdb
import pandas as pd
import os
import sys


# Agregar el directorio raíz al path para importar HashUtility
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.hash_utility import HashUtility


def create_parquet_students():
    # Cargar datos usando DuckDB
    con = duckdb.connect()

    parquet_users_path = "../../data/processed/parquets/Users/mdlvf_user.parquet"
    parquet_user_info_path = "../../data/processed/parquets/Users/mdlvf_user_info_data.parquet"
    excel_students_path = "../../data/processed/excel/Estudiantes.xlsx"
    output_path = "../../data/processed/parquets/Generated/students.parquet"

    # Consulta SQL sobre los datos de Moodle para obtener información de los estudiantes
    sql = f"""
    SELECT 
        u.id AS UserID,
        u.idnumber AS DocumentID,
        CONCAT(u.firstname, ' ', u.lastname) AS 'Nombre Completo',
        u.city AS Sede,
        to_timestamp(u.firstaccess) AS 'Fecha Primer Acceso',
        to_timestamp(u.lastaccess) AS 'Feha Último Acceso',
        to_timestamp(u.lastlogin) AS 'Fecha Último Inicio de Sesión',
        to_timestamp(u.timecreated) AS 'Fecha Creación'
    FROM 
        '{parquet_users_path}' u
    JOIN 
        '{parquet_user_info_path}' uid 
        ON u.id = uid.userid
    WHERE 
        uid.data = 'Estudiante'
        AND u.idnumber <> ''
        AND u.deleted = 0;
    """
    result_df = con.execute(sql).df()

    # Aplicar hashing al DocumentID y eliminar la columna original
    result_df["HashedDocumentID"] = result_df["DocumentID"].apply(HashUtility.hash_stable)
    result_df.drop(columns=["DocumentID"], inplace=True)

    # Carga del archivo Excel
    excel_df = pd.read_excel(excel_students_path)

    # Aplicar hashing al DocumentID del Excel para hacer el merge correctamente
    excel_df["HashedDocumentID"] = excel_df["Documento de Identificación"].astype(str).apply(HashUtility.hash_stable)
    excel_df.drop(columns=["Documento de Identificación"], inplace=True)

    # Realizar el merge entre Moodle y Excel usando el ID hasheado
    merged_df = pd.merge(result_df, excel_df, on="HashedDocumentID", how="inner")

    # Si "Sede" aparece en ambas fuentes (Sede_x y Sede_y):
    # Se mantiene el valor no nulo con combine_first() y se elimina las columnas originales Sede_x y Sede_y
    merged_df.drop(columns=["Sede_x", "Sede_y"], errors="ignore", inplace=True)

    # Eliminación de columnas no necesarias
    merged_df.drop(columns=["Orden Grado"], errors="ignore", inplace=True)

    # Convertir la columna "Fecha de nacimiento" a datetime
    merged_df["Fecha de nacimiento"] = pd.to_datetime(merged_df["Fecha de nacimiento"], errors="coerce")
    # Guardado del DataFrame final como archivo Parquet
    merged_df.to_parquet(output_path, index=False)

    print("Proceso completado. El archivo Parquet con IDs anonimizados ha sido guardado.")
    return


if __name__ == "__main__":
    create_parquet_students()
