import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
import seaborn as sns
import plotly.express as px
from scipy.stats import chi2_contingency, f_oneway

class EDANotesAnalyzer:
    """
    Clase para realizar análisis exploratorio de datos (EDA) en un conjunto de calificaciones estudiantiles.

    Atributos:
        df (pd.DataFrame): El DataFrame que contiene los datos a analizar.

    Métodos:
        plot_null_heatmap(): Muestra un heatmap interactivo de los valores nulos.
        create_risk_variable(threshold): Crea una variable binaria "En_Riesgo" con base en una nota mínima esperada.
        promedio_por_estudiante(): Calcula el promedio de resultado por estudiante, periodo y año.
        verificar_periodos_completos(): Verifica qué estudiantes no tienen todos los periodos registrados por año.
        plot_histograma_por_docente_asignatura(): Muestra histogramas de notas por docente y asignatura.
        analisis_bivariado(col1, col2): Realiza análisis bivariado entre dos variables numéricas.
        cramers_v(col1, col2): Calcula la asociación entre dos variables categóricas con V de Cramer.
        analisis_bivariado_num_cat(col_num, col_cat): Realiza ANOVA entre variable numérica y categórica.
        realizar_anova(): Realiza ANOVA entre todas las variables numéricas y categóricas disponibles.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def plot_null_heatmap(self) -> HTML:
        null_df = self.df.loc[:, self.df.isnull().any()]
        null_matrix_numeric = null_df.isnull().astype(int)
        fig = px.imshow(
            null_matrix_numeric,
            labels=dict(x="Mapa de Calor de Valores Nulos", y="", color=""),
            x=null_matrix_numeric.columns,
            color_continuous_scale=["white", "#13678A"],
            title="",
        )
        fig.update_layout(coloraxis_showscale=False, xaxis=dict(side="top", tickangle=-90), yaxis=dict(autorange="reversed"))
        html = fig.to_html(include_plotlyjs="cdn")
        return HTML(html)

    def create_risk_variable(self, threshold: float = 80) -> pd.DataFrame:
        if 'Resultado' not in self.df.columns:
            raise ValueError("La columna 'Resultado' no existe en el DataFrame.")    
        # Cambiar True/False por 1/0
        self.df['En_Riesgo'] = (self.df['Resultado'] < threshold).astype(int)    
        return self.df

    def promedio_por_estudiante(self) -> pd.DataFrame:
        return self.df.groupby(['Estudiante', 'Periodo', 'Año'])['Resultado'].mean().reset_index()

    def verificar_periodos_completos(self) -> pd.DataFrame:
        max_periodos = self.df.groupby('Año')['Periodo'].nunique().max()
        periodos_por_estudiante_anio = self.df.groupby(['Estudiante', 'Año'])['Periodo'].nunique().reset_index()
        periodos_por_estudiante_anio.rename(columns={'Periodo': 'periodos_registrados'}, inplace=True)
        periodos_por_estudiante_anio['periodos_esperados'] = max_periodos
        periodos_por_estudiante_anio['completo'] = periodos_por_estudiante_anio['periodos_registrados'] == max_periodos
        return periodos_por_estudiante_anio[~periodos_por_estudiante_anio['completo']]

    def plot_histograma_por_docente_asignatura(self):
        docentes = self.df['Docente'].unique()
        n = len(docentes)
        cols = 2
        rows = (n + 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows), constrained_layout=True)
        axes = axes.flatten()
        for i, docente in enumerate(docentes):
            ax = axes[i]
            df_doc = self.df[self.df['Docente'] == docente]
            sns.histplot(data=df_doc, x='Resultado', hue='Asignatura', multiple='stack', bins=10, ax=ax, palette='Set2', edgecolor='black')
            ax.set_title(f'Distribución de Notas - {docente}', fontsize=14)
            ax.set_xlabel('Nota')
            ax.set_ylabel('Frecuencia')
            ax.legend(title='Asignatura', loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        sns.despine()
        plt.show()

    def analisis_bivariado(self, col1: str, col2: str) -> pd.DataFrame:
        """
        Realiza un análisis bivariado entre dos columnas numéricas, imprime gráficos y resultados,
        y devuelve un DataFrame con los resultados del análisis en un formato estructurado.
        
        :param col1: Nombre de la primera columna a analizar
        :param col2: Nombre de la segunda columna a analizar
        :return: DataFrame con los resultados del análisis
        """
        # Verificar si las columnas existen en el DataFrame
        if col1 not in self.df.columns or col2 not in self.df.columns:
            raise ValueError(f"Las columnas '{col1}' o '{col2}' no existen en el DataFrame.")
        
        # Análisis descriptivo de las dos columnas
        description_col1 = self.df[col1].describe()
        description_col2 = self.df[col2].describe()
        
        # Imprimir las estadísticas descriptivas
        print(f"\nEstadísticas descriptivas de la columna '{col1}':")
        print(description_col1)
        print(f"\nEstadísticas descriptivas de la columna '{col2}':")
        print(description_col2)

        # Graficar distribución de las columnas
        plt.figure(figsize=(8, 6))
        sns.histplot(self.df[col1], kde=True, color='blue', label=col1)
        sns.histplot(self.df[col2], kde=True, color='green', label=col2)
        plt.title(f'Distribución de {col1} y {col2}')
        plt.xlabel('Valor')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.show()

        # Graficar la relación bivariada
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=self.df[col1], y=self.df[col2], color='purple')
        plt.title(f'Relación entre {col1} y {col2}')
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.show()

        # Calcular coeficiente de correlación de Pearson
        pearson_corr = self.df[col1].corr(self.df[col2], method='pearson')
        print(f"- Coeficiente de correlación de Pearson entre {col1} y {col2}: {pearson_corr:.2f}")

        # Calcular coeficiente de correlación de Spearman
        spearman_corr = self.df[col1].corr(self.df[col2], method='spearman')
        print(f"- Coeficiente de correlación de Spearman entre {col1} y {col2}: {spearman_corr:.2f}")
        
        # Crear el DataFrame de resultados
        result_df = pd.DataFrame({
            'Columna 1': [col1],
            'Columna 2': [col2],
            'Coef. Pearson': [pearson_corr],
            'Coef. Spearman': [spearman_corr]
        })
        
        # Imprimir el DataFrame de resultados
        print("\nResumen de los resultados:")
        print(result_df)
        
        return result_df

    def cramers_v(self, col1: str, col2: str) -> pd.DataFrame:
        """Calcula V de Cramer para dos variables categóricas e imprime interpretación del resultado."""
        x = self.df[col1].dropna()
        y = self.df[col2].dropna()
        '''if not all(pd.api.types.is_categorical_dtype(self.df[col]) for col in [col1, col2]):
            raise ValueError(
                f"Ambas columnas deben ser categóricas (tipo 'object' o 'category'). "
                f"Tipos detectados: '{col1}'={self.df[col1].dtype}, '{col2}'={self.df[col2].dtype}."
            )'''
        contingency_table = pd.crosstab(x, y)
        chi2 = chi2_contingency(contingency_table)[0]
        n = contingency_table.sum().sum()
        phi2 = chi2 / n
        r, k = contingency_table.shape
        v = np.sqrt(phi2 / min(r - 1, k - 1))

        if v < 0.1:
            interpretacion = "asociación muy débil"
        elif v < 0.3:
            interpretacion = "asociación débil"
        elif v < 0.5:
            interpretacion = "asociación moderada"
        else:
            interpretacion = "asociación fuerte"

        resultado = pd.DataFrame({
            'Columna 1': [col1],
            'Columna 2': [col2],
            'V de Cramer': [v],
            'Interpretación': [interpretacion]
        })

        return resultado

    def realizar_anova(self) -> pd.DataFrame:
        """Realiza ANOVA y devuelve un DataFrame con los resultados."""
        resultados = []
        numericas = self.df.select_dtypes(include=['float64', 'int64']).columns
        categoricas = self.df.select_dtypes(include=['object']).columns

        for var_num in numericas:
            for var_cat in categoricas:
                if self.df[var_cat].nunique() > 1 and self.df[var_num].dropna().nunique() > 1:
                    categorias = [self.df[var_num][self.df[var_cat] == cat] for cat in self.df[var_cat].dropna().unique()]
                    try:
                        f_stat, p_val = f_oneway(*categorias)
                        interpretacion = (
                            "Diferencias significativas (p < 0.05)"
                            if p_val < 0.05 else "No significativas (p ≥ 0.05)"
                        )
                        resultados.append({
                            'Variable numérica': var_num,
                            'Variable categórica': var_cat,
                            'Estadístico F': round(f_stat, 2),
                            'Valor p': round(p_val, 4),
                            'Interpretación': interpretacion
                        })
                    except Exception as e:
                        resultados.append({
                            'Variable numérica': var_num,
                            'Variable categórica': var_cat,
                            'Estadístico F': None,
                            'Valor p': None,
                            'Interpretación': f"Error: {e}"
                        })

        return pd.DataFrame(resultados)

    def analisis_bivariado_num_cat(self, col_num: str, col_cat: str) -> pd.DataFrame:
        """Realiza ANOVA entre una variable numérica y una categórica y devuelve un DataFrame con los resultados."""
        sns.boxplot(x=self.df[col_cat], y=self.df[col_num])
        plt.title(f"Distribución de {col_num} por categorías de {col_cat}")
        plt.xlabel(col_cat)
        plt.ylabel(col_num)
        plt.grid(True)
        plt.show()

        categorias = [self.df[col_num][self.df[col_cat] == cat] for cat in self.df[col_cat].dropna().unique()]
        f_stat, p_val = f_oneway(*categorias)

        interpretacion = (
            "Diferencias significativas (p < 0.05)"
            if p_val < 0.05 else "No significativas (p ≥ 0.05)"
        )

        resultado = pd.DataFrame({
            'Variable numérica': [col_num],
            'Variable categórica': [col_cat],
            'Estadístico F': [round(f_stat, 2)],
            'Valor p': [round(p_val, 4)],
            'Interpretación': [interpretacion]
        })

        return resultado