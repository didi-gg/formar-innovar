import pandas as pd

import numpy as np
import hashlib
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def apply_index(df):
    # Convertir las columnas Año y Periodo a tipo string
    df['Año'] = df['Año'].astype(str)
    df['Periodo_Num'] = df['Periodo_Num'].astype(str)

    # Concatenar Año y Periodo en una nueva columna
    df['Fecha'] = df['Año'] + '-' + df['Periodo_Num']

    # Convertir la columna combinada a tipo datetime 
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%Y-%m')  
    # Establecer la nueva columna como índice 
    df.set_index('Fecha', inplace=True)

    # Ordenar por el índice 
    df.sort_index(inplace=True)
    return df

# Función para aplicar hash a un valor
def hash_value(value):
    return hashlib.sha256(value.encode()).hexdigest()

def agrupar_por_estudiante_y_periodo_old(df, columna_estado='estado'):
    """
    Agrupa un DataFrame por estudiante y período, calculando el promedio de asistencia
    (o cualquier métrica de la columna especificada).

    Args:
        df (pd.DataFrame): DataFrame original con datos de estudiantes, períodos y estado.
        columna_estado (str): Nombre de la columna de la cual se calculará el promedio.

    Returns:
        pd.DataFrame: Nuevo DataFrame agrupado con el promedio de la columna especificada.
    """
    # Verificar que las columnas necesarias existen
    if not all(col in df.columns for col in ['Documento_Hash', 'Periodo_Num', columna_estado]):
        raise ValueError(f"El DataFrame debe contener las columnas: 'Documento_Hash', 'Periodo_Num', '{columna_estado}'")

    # Agrupar por estudiante y período y calcular métricas
    nuevo_df = df.groupby(['Documento_Hash', 'Periodo_Num'], as_index=False).agg(
        promedio_asistencia=(columna_estado, 'mean')
    )

    return nuevo_df

def agrupar_por_estudiante_y_periodo(df, columna_estado='estado'):
    """
    Agrupa un DataFrame por estudiante y período, calculando el promedio de asistencia
    y manteniendo otras columnas no agrupadas.

    Args:
        df (pd.DataFrame): DataFrame original con datos de estudiantes, períodos y estado.
        columna_estado (str): Nombre de la columna de la cual se calculará el promedio.

    Returns:
        pd.DataFrame: Nuevo DataFrame agrupado con el promedio de la columna especificada
                      y columnas adicionales conservadas.
    """
    # Verificar que las columnas necesarias existen
    columnas_requeridas = ['Documento_Hash', 'Periodo_Num', columna_estado]
    if not all(col in df.columns for col in columnas_requeridas):
        raise ValueError(f"El DataFrame debe contener las columnas: {columnas_requeridas}. "
                         f"Columnas encontradas: {list(df.columns)}")

    # Verificar si la columna_estado contiene valores numéricos
    if not pd.api.types.is_numeric_dtype(df[columna_estado]):
        raise ValueError(f"La columna '{columna_estado}' debe contener valores numéricos.")

    # Agrupar por estudiante y período, conservando columnas adicionales
    nuevo_df = df.groupby(['Documento_Hash', 'Periodo_Num'], as_index=False).agg(
        promedio_asistencia=(columna_estado, 'mean'),
        **{col: ('first') for col in df.columns if col not in ['Documento_Hash', 'Periodo_Num', columna_estado]}
    )

    return nuevo_df



def graficos_analitica_descriptiva_v(calificaciones_df, list_cols, filter_col, rotacion):
    num_cols = len(list_cols)  
    plt.figure(figsize=(20, 5))  

    for i, col in enumerate(list_cols):
        plt.subplot(1, num_cols, i + 1)
        sns.boxplot(x=filter_col, y=col, data=calificaciones_df, palette='Set2')
        plt.title(f"{col}", fontsize=14)
        plt.xlabel(filter_col, fontsize=12)
        plt.ylabel(col, fontsize=12)
        plt.xticks(rotation=rotacion, fontsize=10)
        plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.show()

def analitica_descriptiva(df,list_cols, filter):
    resultados = {}
    for col in list_cols:
        descriptivo = df.groupby(filter)[col].describe()
        resultados[col] = descriptivo
        print(f"______ Análisis descriptivo: {col} por {filter} ______")
        print(descriptivo)
        
        print("_______________________________________________________\n")
    return resultados

# Función para graficar el análisis descriptivo con boxplots y distribución
def graficos_analitica_descriptiva(df, list_cols, filter_col, rotacion=0):
    """
    Genera gráficos de caja y distribución para las columnas especificadas,
    agrupadas por una columna de filtro.
    """
    num_cols = len(list_cols)
    
    # Crear figura
    plt.figure(figsize=(10, num_cols * 5))
    
    # Gráficos de caja
    for i, col in enumerate(list_cols):
        plt.subplot(num_cols, 2, i * 2 + 1)  # Coloca en la primera columna
        sns.boxplot(y=filter_col, x=col, data=df, palette='Set2')
        plt.title(f"Boxplot de {col} por {filter_col}", fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel(filter_col, fontsize=12)
        plt.xticks(rotation=rotacion, fontsize=10)
        plt.yticks(fontsize=10)

        # Gráficos de distribución
        plt.subplot(num_cols, 2, i * 2 + 2)  # Coloca en la segunda columna
        sns.kdeplot(df[col], shade=True, color='b', label=f'Distribución de {col}')
        plt.title(f"Distribución de {col}", fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Frecuencia', fontsize=12)
        plt.xticks(rotation=rotacion, fontsize=10)
        plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.show()

#from ydata_profiling import ProfileReport
def generar_reporte_automatico(df, nombre_reporte='EDA_Report.html'):
    """
    Genera un reporte automático de EDA en HTML.
    """
    profile = ProfileReport(netflix_df, title="Reporte de Análisis Exploratorio", explorative=True)
    profile.to_file(nombre_reporte)
    print(f"Reporte generado: {nombre_reporte}")