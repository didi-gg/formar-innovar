"""
Este script toma el archivo de calificaciones en csv y hace las siguientes transformaciones:

1. Transforma los períodos (I, II, III, IV) en números (1, 2, 3, 4).
2. Elimina los datos de preescolar, porque tiene columnnas vacías.
3. Aplica la función hash a la columna de DocumentID.
"""

import pandas as pd
import os
import sys


# Agregar el directorio raíz al path para importar HashUtility
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.hash_utility import HashUtility


def transform_academic_report():
    # Cargar datos
    academic_report_path = "../../data/raw/calificaciones/academic_report.csv"
    output_path = "../../data/processed/csv/academic_report_transformed.csv"

    # Leer archivo CSV
    df = pd.read_csv(academic_report_path)

    # Transformar períodos
    period_mapping = {"I": 1, "II": 2, "III": 3, "IV": 4}
    df["Periodo"] = df["Periodo"].map(period_mapping)

    # Eliminar datos de preescolar
    df = df[df["Grado"] != "Párvulos"]
    df = df[df["Grado"] != "Prejardín"]
    df = df[df["Grado"] != "Jardín"]
    df = df[df["Grado"] != "Transición"]

    # Aplicar hashing a DocumentID
    df["HashedDocumentID"] = df["Documento de identidad"].apply(HashUtility.hash_stable)
    df.drop(columns=["Documento de identidad"], inplace=True)

    # Guardar datos transformados
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    transform_academic_report()
