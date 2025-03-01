import pandas as pd

def procesar_excel(ruta_archivo, umbral_nulos=0.5):
    """
    Procesa un archivo Excel leyendo todas sus pestañas, eliminando filas con muchos valores nulos,
    y unificando los datos en un DataFrame.

    Args:
        ruta_archivo (str): Ruta al archivo Excel.
        umbral_nulos (float): Proporción máxima de valores nulos permitida por fila (0 a 1).

    Returns:
        pd.DataFrame: DataFrame unificado con los datos limpios.
    """
    # Leer todas las hojas del archivo Excel
    excel_data = pd.ExcelFile(ruta_archivo)
    hojas = excel_data.sheet_names
    dataframes = []
    
    for hoja in hojas:
        # Leer la hoja
        df = excel_data.parse(sheet_name=hoja)
        
        # Eliminar filas con valores nulos mayores al umbral
        df = df[df.isnull().mean(axis=1) < umbral_nulos]
        
        # Agregar columna para identificar la pestaña de origen
        df['Hoja'] = hoja
        
        # Añadir al conjunto de DataFrames
        dataframes.append(df)
    
    # Concatenar todas las hojas en un solo DataFrame
    df_unificado = pd.concat(dataframes, ignore_index=True)
    return df_unificado

# Ejemplo de uso:
# ruta = 'ruta_a_tu_archivo.xlsx'
# df_resultado = procesar_excel(ruta)
