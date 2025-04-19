import os
import pandas as pd
import logging
import re
import unicodedata
import numpy as np


def procesar_hoja_asistencia_fusagasuga_mes(datos, mes, grado, anio):
    """
    Limpia y procesa una hoja de datos de asistencia con las columnas Mes, Grado y Año.
    
    Args:
        datos (pd.DataFrame): Hoja de datos cargada desde Excel.
        mes (str): Mes asociado al archivo.
        grado (str): Grado correspondiente a la hoja.
        anio (int): Año correspondiente a los datos.
    
    Returns:
        pd.DataFrame: Datos procesados con columnas estándar.
    """
    # Eliminar las primeras dos filas que suelen ser irrelevantes (encabezados extra o notas).
    datos = datos.iloc[2:, :].reset_index(drop=True)
    
    # Eliminar la primera columna, que suele ser un índice o un dato irrelevante.
    datos = datos.drop(datos.columns[0], axis=1)
    
    # Mantener solo las primeras 34 columnas, eliminando cualquier dato adicional.
    datos = datos.iloc[:, :34]
    
    # Tomar la primera fila de datos como encabezados de las columnas.
    datos.columns = datos.iloc[0]
    
    # Eliminar la fila que ahora es redundante (la que se usó como encabezados).
    datos = datos[1:]
    
    # Normalizar los encabezados: convertirlos a cadenas y eliminar inconsistencias.
    datos.columns = [
        str(col).strip().replace(".0", "") if isinstance(col, (int, float)) else str(col).strip()
        for col in datos.columns
    ]
    
    # Eliminar las columnas cuyos encabezados sean 'nan' o vacíos.
    datos = datos.loc[:, ~datos.columns.str.lower().str.contains('nan')]
     # Agregar tres nuevas columnas con información contextual: mes, grado y año.
    datos["Mes"] = mes
    datos["Grado"] = grado
    datos["Año"] = anio

    # Eliminar filas donde el valor de la columna "Nombre" sea nulo.
    datos = datos.dropna(subset=["Nombre del estudiante"])
    
    # Nota: La línea comentada a continuación sirve para excluir filas específicas si contienen ":"
    # datos = datos[~datos.iloc[:, 0].str.contains(":", na=False)]
    
    return datos  # Retornar el DataFrame procesado.

def consolidar_asistencia_fusagasuga(archivo_excel, mes, anio, salida="asistencia_consolidada.csv"):
    """
    Consolida los datos de asistencia de todas las hojas de un archivo Excel.
    
    Args:
        archivo_excel (str): Ruta al archivo Excel.
        mes (str): Mes correspondiente al archivo.
        anio (int): Año correspondiente al archivo.
        salida (str): Nombre del archivo CSV consolidado de salida.
    
    Returns:
        pd.DataFrame: DataFrame consolidado con datos de todas las hojas.
    """
    # Leer el archivo Excel completo.
    excel_data = pd.ExcelFile(archivo_excel)
    
    # Crear una lista para almacenar los DataFrames procesados de cada hoja.
    dataframes = []
    
    # Iterar sobre todas las hojas del archivo Excel.
    for hoja in excel_data.sheet_names:
        print(f"Procesando hoja: {hoja}")  # Mensaje informativo para cada hoja procesada.
        
        # Leer los datos de la hoja actual.
        datos = excel_data.parse(hoja)
        
        # Procesar los datos de la hoja utilizando la función anterior.
        datos_procesados = procesar_hoja_asistencia_fusagasuga_mes(datos, mes, hoja, anio)
        
        # Agregar los datos procesados a la lista.
        dataframes.append(datos_procesados)
    
    # Concatenar todos los DataFrames en un único DataFrame consolidado.
    datos_consolidados = pd.concat(dataframes, ignore_index=True)
    
    # Guardar el DataFrame consolidado en un archivo CSV.
    datos_consolidados.to_csv(salida, index=False)
    
    # Mensaje de confirmación al finalizar.
    print(f"Consolidación completada. Archivo guardado como '{salida}'.")
    
    return datos_consolidados  # Retornar el DataFrame consolidado.

def consolidar_archivos_fusagasuga_csv(directorio, salida="consolidado_asistencia_final_2024.csv"):
    """
    Lee múltiples archivos CSV en un directorio y los consolida en un único DataFrame.
    
    Args:
        directorio (str): Ruta al directorio donde se encuentran los archivos CSV.
        salida (str): Nombre del archivo CSV consolidado de salida.
    
    Returns:
        pd.DataFrame: DataFrame consolidado con los datos de todos los archivos.
    """
    # Lista para almacenar los DataFrames de cada archivo.
    dataframes = []
    
    # Iterar sobre todos los archivos en el directorio.
    for archivo in os.listdir(directorio):
        if archivo.endswith(".csv"):  # Solo procesar archivos con extensión .csv
            ruta_completa = os.path.join(directorio, archivo)
            print(f"Leyendo archivo: {ruta_completa}")
            
            # Leer el archivo CSV en un DataFrame.
            df = pd.read_csv(ruta_completa)
            
            # Agregar el nombre del archivo como columna para referencia (opcional).
            #df["Archivo"] = archivo
            
            # Añadir el DataFrame a la lista.
            dataframes.append(df)
    
    # Concatenar todos los DataFrames en un único DataFrame consolidado.
    consolidado = pd.concat(dataframes, ignore_index=True)
    
    # Guardar el DataFrame consolidado en un archivo CSV.
    consolidado.to_csv(salida, index=False)
    print(f"Consolidación completada. Archivo guardado como '{salida}'.")
    
    return consolidado  # Retornar el DataFrame consolidado.

def quitar_tildes(texto):
    # Normaliza el texto (NFD separa los caracteres y sus diacríticos)
    texto_normalizado = unicodedata.normalize('NFD', texto)
    # Filtra los caracteres que no son diacríticos
    texto_sin_tildes = ''.join(c for c in texto_normalizado if unicodedata.category(c) != 'Mn')
    # Retorna el texto normalizado a NFC
    return unicodedata.normalize('NFC', texto_sin_tildes)

def limpiar_espacios(texto):
    if not isinstance(texto, str):
        return texto  # Retorna el valor sin cambios si no es una cadena
    # Elimina espacios iniciales y finales, y convierte múltiples espacios en uno solo
    return ' '.join(texto.split())

def convertir_a_formato_largo(df, id_vars, dias_col_start, dias_col_end):
    """
    Convierte un DataFrame con columnas de días en formato ancho a formato largo.
    
    Parámetros:
    - df (DataFrame): El DataFrame de entrada.
    - id_vars (list): Lista de columnas que deben mantenerse como identificadores.
    - dias_col_start (int): Primera columna que representa los días (índice de columna).
    - dias_col_end (int): Última columna que representa los días (índice de columna, inclusive).
    
    Retorna:
    - DataFrame en formato largo.
    """
    # Identificar las columnas de los días basadas en sus índices
    columnas_dias = df.columns[dias_col_start:dias_col_end + 1]
    
    # Convertir al formato largo
    df_long = df.melt(
        id_vars=id_vars,               # Mantener estas columnas sin cambios
        value_vars=columnas_dias,      # Columnas que representan los días
        var_name="Día",                # Nueva columna para los días
        value_name="Estado"            # Nueva columna para los estados de asistencia
    )
    
    # Convertir la columna 'Día' a numérica
    df_long["Día"] = pd.to_numeric(df_long["Día"])
    return df_long

def convertir_a_numero_mes_grado(df):
    # Diccionario para convertir meses a números
    meses_a_numeros = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, 
    "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8, 
    "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
    }

    # Diccionario para convertir grados a un formato numérico o estándar
    grados_a_numeros = {
        "Prejardín": -2, "Jardín": -1, "Transición": 0, 
        "Primero": 1, "Segundo": 2, "Tercero": 3, "Cuarto": 4,
        "Quinto": 5, "Sexto": 6, "Séptimo": 7, "Octavo": 8, 
        "Noveno": 9, "Décimo": 10, "Undécimo": 11
    }

    # Convertir la columna 'Mes' usando el diccionario
    df["Mes_Num"] = df["Mes"].map(meses_a_numeros)

    # Convertir la columna 'Grado' usando el diccionario
    df["Grado_Num"] = df["Grado"].map(grados_a_numeros)
    return df

def agregar_bimestre_con_dias(df):
    """
    Asigna el bimestre basado en Mes_Num y Día, incluyendo excepciones como en abril.

    Parámetros:
    - df: DataFrame con las columnas 'Mes_Num' y 'Día'.

    Retorna:
    - DataFrame con la columna 'Bimestre' agregada.
    """
    # Definir condiciones para los bimestres
    condiciones = [
        # Primer bimestre
        ((df["Mes_Num"] == 2) | (df["Mes_Num"] == 3)) | 
        ((df["Mes_Num"] == 4) & (df["Día"] <= 12)),
        
        # Segundo bimestre
        ((df["Mes_Num"] == 4) & (df["Día"] > 12)) | 
        (df["Mes_Num"] == 5) | 
        ((df["Mes_Num"] == 6) & (df["Día"] <= 14)),
        
        # Tercer bimestre
        ((df["Mes_Num"] == 6) & (df["Día"] > 14)) | 
        ((df["Mes_Num"] == 7) | (df["Mes_Num"] == 8)) | 
        ((df["Mes_Num"] == 9) & (df["Día"] <= 6)),
        
        # Cuarto bimestre
        ((df["Mes_Num"] == 9) & (df["Día"] > 6)) | 
        ((df["Mes_Num"] == 10)) | 
        ((df["Mes_Num"] == 11) & (df["Día"] <= 15))
            
    ]

    # Valores de los bimestres
    bimestres = ["I", "II", "III", "IV"]

    # Asignar los bimestres
    df["Bimestre"] = np.select(condiciones, bimestres, default=np.nan)

    return df

def mapear_estado_asistencia(df):
    """
    Mapea los valores de asistencia en el DataFrame a valores numéricos.

    Parámetros:
    - df: DataFrame con una columna llamada 'Estado' que contiene los valores de asistencia.

    Retorna:
    - DataFrame con una nueva columna llamada 'Estado_Num' que contiene los valores mapeados.
    """
    # Diccionario de mapeo de asistencia
    estado_map = {
        "A": 1,    # Asistió
        "FJ": 0.5, # Falla justificada
        "X": 0,    # Falla sin justificar
        "T": 0.75, # Llegó tarde
        "D": -1    # Descanso
    }
    
    # Mapear los valores de la columna 'Estado' usando el diccionario
    df["Estado_Num"] = df["Estado"].map(estado_map)
    
    return df

