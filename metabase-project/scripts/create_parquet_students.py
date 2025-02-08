"""
Este script usa DuckDB y Pandas para extraer información de los estudiantes desde archivos Parquet, combinarla con un archivo Excel,
y finalmente guardar los datos procesados en un nuevo archivo Parquet.
"""

# Importar librerías
import duckdb
import pandas as pd

# Cargar datos usando DuckDB
con = duckdb.connect()

# Consulta SQL sobre los datos de Moodle para obtener información de los estudiantes
sql = """
SELECT 
    u.id AS UserID,
    u.idnumber AS DocumentID,
    CONCAT(u.firstname, ' ', u.lastname) AS 'Nombre Completo',
    u.city AS Sede,
    to_timestamp(u.firstaccess) AS 'Fecha Primer Acceso',
    to_timestamp(u.lastaccess) AS 'Feha Último Acceso',
    to_timestamp(u.lastlogin) AS 'Fecha Último Inicio de Sesión',
    to_timestamp(u.timecreated) AS 'Fecha Creación',
FROM 
    'metabase-project/data/parquets/Users/mdlvf_user.parquet' u
JOIN 
    'metabase-project/data/parquets/Users/mdlvf_user_info_data.parquet' uid 
    ON u.id = uid.userid
WHERE 
    uid.data = 'Estudiante'
    AND u.idnumber <> ''
    AND u.deleted = 0;
"""
result_df = con.execute(sql).df()

# Carga del archivo Excel
excel_df = pd.read_excel("metabase-project/data/excel/Estudiantes.xlsx")
# Se renombra la columna "Documento de Identificación" a "DocumentID" para hacer coincidir con la consulta SQL
excel_df.rename(columns={"Documento de Identificación": "DocumentID"}, inplace=True)

# Se convierten los identificadores a cadena de texto para evitar problemas al hacer el merge
result_df["DocumentID"] = result_df["DocumentID"].astype(str)
excel_df["DocumentID"] = excel_df["DocumentID"].astype(str)

# Se realiza un merge (inner join) entre los datos de Moodle (Parquet) y el archivo Excel, usando "DocumentID" como clave
merged_df = pd.merge(result_df, excel_df, on="DocumentID", how="inner")

# Si "Sede" aparece en ambas fuentes (Sede_x y Sede_y):
# Se mantiene el valor no nulo con combine_first() y se elimina las columnas originales Sede_x y Sede_y
merged_df["Sede"] = merged_df["Sede_x"].combine_first(merged_df["Sede_y"])
merged_df = merged_df.drop(columns=["Sede_x", "Sede_y"], errors="ignore")

# Eliminación de columnas no necesarias
merged_df = merged_df.drop(columns=["Orden Grado"], errors="ignore")

# Convertir la columna "Fecha de nacimiento" a datetime
merged_df["Fecha de nacimiento"] = pd.to_datetime(
    merged_df["Fecha de nacimiento"], errors="coerce"
)

# Guardado del DataFrame final como archivo Parquet
merged_df.to_parquet(
    "metabase-project/data/parquets/Generated/students.parquet", index=False
)
