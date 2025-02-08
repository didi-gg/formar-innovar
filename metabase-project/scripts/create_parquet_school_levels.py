"""
Este script usa Pandas y DuckDB para leer un archivo Excel con información sobre los grados escolares, verifica que tenga las columnas necesarias, y lo guarda en formato Parquet.
"""

#  Importar librerías
import duckdb
import pandas as pd

# Cargar datos usando DuckDB
con = duckdb.connect()

# Cargar el archivo Excel
input_file = "metabase-project/data/excel/Grados.xlsx"
output_file = "metabase-project/data/parquets/Generated/grade_order.parquet"

# Leer el archivo Excel
df = pd.read_excel(input_file)

# Verificar que las columnas necesarias existan
if not {"Grado", "Nivel", "Orden"}.issubset(df.columns):
    raise ValueError("El archivo Excel debe contener las columnas: Grado, Nivel, Orden")

# Guardar como archivo Parquet
df.to_parquet(output_file, engine="pyarrow", index=False)
