import pandas as pd
import plotly.express as px

class EDAAnalyzer:
    def plot_null_heatmap(self, dataframe):
        """
        Crea un gráfico de calor interactivo mostrando valores nulos
        solo para las columnas que contienen al menos un valor nulo.

        Args:
            dataframe (pd.DataFrame): DataFrame a analizar.
        """
        # Filtrar columnas con valores nulos
        null_df = dataframe.loc[:, dataframe.isnull().any()]

        # Crear un dataframe booleano (True para nulo, False para no nulo)
        null_matrix = null_df.isnull()

        # Convertir booleanos a números (1 o 0) para visualización
        null_matrix_numeric = null_matrix.astype(int)

        # Crear gráfico de calor con Plotly (solo color para nulos)
        fig = px.imshow(
            null_matrix_numeric,
            labels=dict(x="Mapa de Calor de Valores Nulos", y="", color=""),
            x=null_matrix_numeric.columns,
            color_continuous_scale=["white", "#13678A"],
            title=''
        )

        fig.update_layout(
            coloraxis_showscale=False,
            xaxis=dict(side="top", tickangle=-90),
            yaxis=dict(autorange="reversed")
        )

        fig.show()

    def plot_boxplots(self, dataframe, columns):
            for col in columns:
                fig = px.box(dataframe, y=col, title=f'Boxplot - {col}')
                fig.show()

    def plot_histograms(self, dataframe, columns, bins=20):
        for col in columns:
            fig = px.histogram(dataframe, x=col, nbins=bins, title=f'Histograma - {col}')
            fig.show()

    def plot_barplots(self, dataframe, columns):
        for col in columns:
            data = dataframe[col].value_counts().reset_index()
            data.columns = [col, 'conteo']
            fig = px.bar(data, x=col, y='conteo', title=f'{col}')
            fig.show()

    def plot_ecdf(self, dataframe, columns):
        """
        Crea gráficos ECDF (Empirical Cumulative Distribution Function)
        para columnas numéricas seleccionadas.
        """
        for col in columns:
            fig = px.ecdf(dataframe, x=col, title=f'ECDF - {col}')
            fig.show()

    def detect_outliers(self, dataframe, columns):
        """
        Detecta valores atípicos usando el método del rango intercuartílico (IQR) y genera boxplots.
        """
        atipicos = {}
        for col in columns:
            Q1 = dataframe[col].quantile(0.25)
            Q3 = dataframe[col].quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            outliers = dataframe[(dataframe[col] < limite_inferior) | (dataframe[col] > limite_superior)]
            atipicos[col] = outliers[col]

            print(f"\n{col} - Atípicos detectados: {len(outliers)}")
            fig = px.box(dataframe, y=col, title=f'Boxplot con Atípicos - {col}')
            fig.show()
        return atipicos
    
    def detect_rare_categories(self, dataframe, columns, threshold=5):
        """
        Detecta categorías poco frecuentes (atípicos) en columnas categóricas.

        Args:
            dataframe (pd.DataFrame): DataFrame a analizar.
            columns (list): Lista de columnas categóricas.
            umbral (int): Frecuencia mínima para no considerarse atípico.
        """
        categorias_raras = {}
        for col in columns:
            print(col)
            frecuencia = dataframe[col].value_counts()
            raras = frecuencia[frecuencia < threshold].index.tolist()
            categorias_raras[col] = raras

            print(f"\n{col} - Categorías raras (< {threshold} apariciones):")
            print(raras)

        self.plot_barplots(dataframe, list(categorias_raras.keys()))
        return categorias_raras

# Ejemplo de uso:
# eda = EDAAnalyzer()
# eda.plot_null_heatmap(df)