"""
Este script usa Pandas y DuckDB para leer un archivo Excel con información sobre los grados escolares, verifica que tenga las columnas necesarias, y lo guarda en formato Parquet.
"""

#  Importar librerías
import duckdb
import pandas as pd

# Cargar datos usando DuckDB
con = duckdb.connect()


def create_parquet_grades():
    # Cargar el archivo Excel
    input_file = "../../data/processed/excel/Grados.xlsx"
    output_file = "../../data/processed/parquets/Generated/academic_levels.parquet"

    # Leer el archivo Excel
    df = pd.read_excel(input_file)

    # Verificar que las columnas necesarias existan
    if not {"Grado", "Nivel", "Orden"}.issubset(df.columns):
        raise ValueError("El archivo Excel debe contener las columnas: Grado, Nivel, Orden")

    # Guardar como archivo Parquet
    df.to_parquet(output_file, engine="pyarrow", index=False)

    print(f"Archivo generado correctamente en {output_file}")


if __name__ == "__main__":
    create_parquet_grades()
