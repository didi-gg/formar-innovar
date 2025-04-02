import pandas as pd
import plotly.express as px
from IPython.display import HTML
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math


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
            title="",
        )

        fig.update_layout(coloraxis_showscale=False, xaxis=dict(side="top", tickangle=-90), yaxis=dict(autorange="reversed"))

        html = fig.to_html(include_plotlyjs="cdn")
        return HTML(html)

    def plot_boxplots(self, dataframe, columns):
        for col in columns:
            fig = px.box(dataframe, y=col, title=f"Boxplot - {col}")
            html = fig.to_html(include_plotlyjs="cdn")
            return HTML(html)

    def plot_histograms(self, dataframe, columns, bins=20):
        for col in columns:
            fig = px.histogram(dataframe, x=col, nbins=bins, title=f"Histograma - {col}")
            html = fig.to_html(include_plotlyjs="cdn")
            return HTML(html)

    def plot_single_barplot(self, dataframe, column):
        data = dataframe[column].value_counts().reset_index()
        data.columns = [column, "conteo"]

        fig = px.bar(
            data,
            x=column,
            y="conteo",
            title=column.replace("_", " ").title(),
        )

        fig.update_layout(
            height=500,
            title_x=0.5,
            margin=dict(l=50, r=50, t=80, b=50),
        )

        html = fig.to_html(include_plotlyjs="cdn")
        return HTML(html)

    def plot_barplots(self, title, dataframe, columns):
        cols_per_row = 2
        num_plots = len(columns)
        rows = math.ceil(num_plots / cols_per_row)

        max_spacing = 1 / max(rows - 1, 1)
        vertical_spacing = min(0.2, max_spacing - 0.01) if rows > 1 else 0.0

        fig = make_subplots(
            rows=rows,
            cols=cols_per_row,
            subplot_titles=[col.replace("_", " ").title() for col in columns],
            horizontal_spacing=0.15,
            vertical_spacing=vertical_spacing,
        )

        for i, col in enumerate(columns):
            row = i // cols_per_row + 1
            col_pos = i % cols_per_row + 1
            data = dataframe[col].value_counts().reset_index()
            data.columns = [col, "conteo"]

            fig.add_trace(go.Bar(x=data[col], y=data["conteo"], name=col), row=row, col=col_pos)

        fig.update_layout(
            height=350 * rows,
            width=1200,
            title_text=title,
            showlegend=False,
            title_x=0.5,
        )

        html = fig.to_html(include_plotlyjs="cdn")
        return HTML(html)

    def plot_ecdf(self, dataframe, columns):
        """
        Crea gráficos ECDF (Empirical Cumulative Distribution Function)
        para columnas numéricas seleccionadas.
        """
        for col in columns:
            fig = px.ecdf(dataframe, x=col, title=f"ECDF - {col}")
            html = fig.to_html(include_plotlyjs="cdn")
            return HTML(html)

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

            if len(outliers) > 0:
                atipicos[col] = outliers[col]
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
            frecuencia = dataframe[col].value_counts()
            raras = frecuencia[frecuencia < threshold].index.tolist()
            if len(raras) > 0:
                categorias_raras[col] = raras
        return categorias_raras
