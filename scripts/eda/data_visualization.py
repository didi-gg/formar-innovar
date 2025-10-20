"""
Script para análisis de visualización de datos exploratorio (EDA).
Incluye gráficas de distribución, matriz de correlación, scatter plots y box plots.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import pearsonr, spearmanr, chi2_contingency
import itertools
from scipy.stats import contingency

# Configuración de warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para suprimir mensajes de debug
import matplotlib
matplotlib.set_loglevel("WARNING")

# Agregar el directorio padre al path para importar utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.eda_analysis_base import EDAAnalysisBase


class DataVisualizationAnalyzer(EDAAnalysisBase):
    """Analizador de visualización de datos para EDA."""

    def _initialize_analysis_attributes(self):
        """Inicializar atributos específicos del análisis de visualización."""
        self.correlation_matrix = None
        self.numeric_features = None
        self.categorical_features = None
        self.results = {}

    def analyze_target_distribution(self, df):
        """
        Analiza y visualiza la distribución de la variable respuesta numérica.

        Args:
            df (pd.DataFrame): Dataset con los datos
        """
        self.logger.info("Analizando distribución de la variable respuesta...")

        if self.TARGET_NUMERIC not in df.columns:
            self.logger.warning(f"Variable objetivo numérica '{self.TARGET_NUMERIC}' no encontrada")
            return

        target_data = df[self.TARGET_NUMERIC].dropna()

        # Crear figura con múltiples subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Análisis de Distribución - {self.TARGET_NUMERIC}', fontsize=16, fontweight='bold')

        # 1. Histograma con curva de densidad
        axes[0, 0].hist(target_data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(target_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {target_data.mean():.2f}')
        axes[0, 0].axvline(target_data.median(), color='orange', linestyle='--', linewidth=2, label=f'Mediana: {target_data.median():.2f}')

        # Agregar curva de densidad
        x_range = np.linspace(target_data.min(), target_data.max(), 100)
        kde = stats.gaussian_kde(target_data)
        axes[0, 0].plot(x_range, kde(x_range), 'r-', linewidth=2, label='Densidad')

        axes[0, 0].set_title('Histograma con Curva de Densidad')
        axes[0, 0].set_xlabel(self.TARGET_NUMERIC)
        axes[0, 0].set_ylabel('Densidad')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Box plot
        box_plot = axes[0, 1].boxplot(target_data, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightgreen')
        axes[0, 1].set_title('Box Plot')
        axes[0, 1].set_ylabel(self.TARGET_NUMERIC)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Q-Q plot para normalidad
        stats.probplot(target_data, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normalidad)')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Estadísticas descriptivas como texto
        axes[1, 1].axis('off')
        stats_text = f"""
        Estadísticas Descriptivas:

        Observaciones: {len(target_data)}
        Media: {target_data.mean():.3f}
        Mediana: {target_data.median():.3f}
        Desv. Estándar: {target_data.std():.3f}
        Mínimo: {target_data.min():.3f}
        Máximo: {target_data.max():.3f}

        Cuartiles:
        Q1 (25%): {target_data.quantile(0.25):.3f}
        Q3 (75%): {target_data.quantile(0.75):.3f}
        IQR: {target_data.quantile(0.75) - target_data.quantile(0.25):.3f}

        Asimetría: {target_data.skew():.3f}
        Curtosis: {target_data.kurtosis():.3f}

        Test de Normalidad (Shapiro-Wilk):
        """

        # Test de normalidad (solo si hay menos de 5000 observaciones)
        if len(target_data) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(target_data)
            stats_text += f"Estadístico: {shapiro_stat:.4f}\n        p-valor: {shapiro_p:.4f}"
            if shapiro_p > 0.05:
                stats_text += "\n        Distribución NORMAL"
            else:
                stats_text += "\n        Distribución NO NORMAL"
        else:
            stats_text += "Muestra muy grande para Shapiro-Wilk"

        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

        plt.tight_layout()
        plt.savefig(f'{self.results_path}/target_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Guardar estadísticas
        self.results['target_distribution'] = {
            'count': len(target_data),
            'mean': target_data.mean(),
            'median': target_data.median(),
            'std': target_data.std(),
            'min': target_data.min(),
            'max': target_data.max(),
            'q1': target_data.quantile(0.25),
            'q3': target_data.quantile(0.75),
            'skewness': target_data.skew(),
            'kurtosis': target_data.kurtosis()
        }

        self.logger.info("✅ Análisis de distribución de variable respuesta completado")

    def create_correlation_matrix(self, df):
        """
        Crea y visualiza la matriz de correlación de variables numéricas.

        Args:
            df (pd.DataFrame): Dataset con los datos
        """
        self.logger.info("Creando matriz de correlación...")

        # Obtener variables numéricas disponibles
        numeric_cols = self.get_numeric_columns_available(df)

        if len(numeric_cols) < 2:
            self.logger.warning("Insuficientes variables numéricas para matriz de correlación")
            return

        # Filtrar datos válidos
        df_numeric = df[numeric_cols].select_dtypes(include=[np.number])
        df_clean = df_numeric.dropna()

        if df_clean.empty:
            self.logger.warning("No hay datos válidos para la matriz de correlación")
            return

        # Calcular matriz de correlación
        correlation_matrix = df_clean.corr()
        self.correlation_matrix = correlation_matrix

        # Crear visualización
        plt.figure(figsize=(16, 14))

        # Crear máscara para la matriz triangular superior
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        # Crear heatmap
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={"shrink": .8},
                   annot_kws={'size': 8})

        plt.title('Matriz de Correlación de Variables Numéricas', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Identificar correlaciones más fuertes con la variable objetivo
        if self.TARGET_NUMERIC in correlation_matrix.columns:
            target_correlations = correlation_matrix[self.TARGET_NUMERIC].drop(self.TARGET_NUMERIC).abs().sort_values(ascending=False)

            # Visualizar top correlaciones con variable objetivo
            plt.figure(figsize=(12, 8))
            top_corr = target_correlations.head(15)
            colors = ['red' if abs(correlation_matrix[self.TARGET_NUMERIC][var]) >= 0.5 else 'steelblue' for var in top_corr.index]

            bars = plt.barh(range(len(top_corr)), top_corr.values, color=colors)
            plt.yticks(range(len(top_corr)), top_corr.index)
            plt.xlabel('Correlación Absoluta')
            plt.title(f'Top 15 Correlaciones con {self.TARGET_NUMERIC}', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()

            # Agregar valores en las barras
            for i, (bar, val) in enumerate(zip(bars, top_corr.values)):
                original_corr = correlation_matrix[self.TARGET_NUMERIC][top_corr.index[i]]
                plt.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{original_corr:.3f}', va='center', fontsize=9)

            plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Correlación fuerte (|r| ≥ 0.5)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{self.results_path}/target_correlations.png', dpi=300, bbox_inches='tight')
            plt.close()

            self.results['target_correlations'] = target_correlations.to_dict()

        # Guardar matriz de correlación
        correlation_matrix.to_csv(f'{self.results_path}/correlation_matrix.csv')
        self.results['correlation_matrix_shape'] = correlation_matrix.shape

        self.logger.info("✅ Matriz de correlación completada")

    def create_scatter_plots_vs_target(self, df, max_plots=None):
        """
        Crea scatter plots de variables numéricas vs variable objetivo.

        Args:
            df (pd.DataFrame): Dataset con los datos
            max_plots (int): Número máximo de gráficos a crear
        """
        self.logger.info("Creando scatter plots vs variable objetivo...")

        if self.TARGET_NUMERIC not in df.columns:
            self.logger.warning(f"Variable objetivo '{self.TARGET_NUMERIC}' no encontrada")
            return

        # Obtener variables numéricas (excluyendo la variable objetivo)
        numeric_cols = [col for col in self.get_numeric_columns_available(df) 
                       if col != self.TARGET_NUMERIC and col in df.columns]

        if len(numeric_cols) == 0:
            self.logger.warning("No hay variables numéricas para scatter plots")
            return

        # Limitar número de gráficos solo si se especifica max_plots
        if max_plots is not None and len(numeric_cols) > max_plots:
            # Priorizar variables con mayor correlación si está disponible
            if hasattr(self, 'correlation_matrix') and self.correlation_matrix is not None:
                if self.TARGET_NUMERIC in self.correlation_matrix.columns:
                    target_corr = self.correlation_matrix[self.TARGET_NUMERIC].abs().sort_values(ascending=False)
                    numeric_cols = [col for col in target_corr.index if col in numeric_cols][:max_plots]
                else:
                    numeric_cols = numeric_cols[:max_plots]
            else:
                numeric_cols = numeric_cols[:max_plots]
        elif hasattr(self, 'correlation_matrix') and self.correlation_matrix is not None:
            # Si no hay límite, ordenar por correlación para mejor visualización
            if self.TARGET_NUMERIC in self.correlation_matrix.columns:
                target_corr = self.correlation_matrix[self.TARGET_NUMERIC].abs().sort_values(ascending=False)
                numeric_cols = [col for col in target_corr.index if col in numeric_cols]

        # Advertencia si hay muchas variables
        if len(numeric_cols) > 30:
            self.logger.warning(f"Generando {len(numeric_cols)} scatter plots. Esto puede tomar tiempo y generar archivos grandes.")

        # Calcular número de filas y columnas para subplots
        n_plots = len(numeric_cols)
        n_cols = 4
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        fig.suptitle(f'Scatter Plots vs {self.TARGET_NUMERIC}', fontsize=16, fontweight='bold', y=0.98)

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, col in enumerate(numeric_cols):
            row = i // n_cols
            col_idx = i % n_cols
            ax = axes[row, col_idx]

            # Filtrar datos válidos
            mask = df[[col, self.TARGET_NUMERIC]].notna().all(axis=1)
            x_data = df.loc[mask, col]
            y_data = df.loc[mask, self.TARGET_NUMERIC]

            if len(x_data) > 0:
                # Crear scatter plot
                ax.scatter(x_data, y_data, alpha=0.6, s=30, color='steelblue')

                # Agregar línea de tendencia
                if len(x_data) > 1:
                    z = np.polyfit(x_data, y_data, 1)
                    p = np.poly1d(z)
                    ax.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2)

                    # Calcular correlación
                    corr, p_val = pearsonr(x_data, y_data)
                    ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

                ax.set_xlabel(col)
                ax.set_ylabel(self.TARGET_NUMERIC)
                ax.set_title(f'{col} vs {self.TARGET_NUMERIC}', fontsize=10)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Sin datos válidos', transform=ax.transAxes, 
                       ha='center', va='center')
                ax.set_title(f'{col} vs {self.TARGET_NUMERIC}', fontsize=10)

        # Ocultar subplots vacíos
        for i in range(n_plots, n_rows * n_cols):
            row = i // n_cols
            col_idx = i % n_cols
            axes[row, col_idx].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Dejar espacio para el título
        plt.savefig(f'{self.results_path}/scatter_plots_vs_target.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.results['scatter_plots_count'] = len(numeric_cols)
        self.logger.info("✅ Scatter plots vs variable objetivo completados")

    def create_bivariate_analysis(self, df, max_pairs=None):
        """
        Crea scatter plots para análisis bivariado entre variables numéricas.

        Args:
            df (pd.DataFrame): Dataset con los datos
            max_pairs (int): Número máximo de pares a analizar
        """
        self.logger.info("Creando análisis bivariado...")

        # Obtener variables numéricas
        numeric_cols = [col for col in self.get_numeric_columns_available(df) if col in df.columns]

        if len(numeric_cols) < 2:
            self.logger.warning("Insuficientes variables numéricas para análisis bivariado")
            return

        # Seleccionar pares más interesantes basados en correlación si está disponible
        if hasattr(self, 'correlation_matrix') and self.correlation_matrix is not None:
            # Obtener pares con mayor correlación absoluta
            corr_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    var1, var2 = numeric_cols[i], numeric_cols[j]
                    if var1 in self.correlation_matrix.columns and var2 in self.correlation_matrix.columns:
                        corr_val = abs(self.correlation_matrix.loc[var1, var2])
                        corr_pairs.append((var1, var2, corr_val))
            
            # Ordenar por correlación y limitar solo si se especifica max_pairs
            corr_pairs.sort(key=lambda x: x[2], reverse=True)
            if max_pairs is not None:
                selected_pairs = [(pair[0], pair[1]) for pair in corr_pairs[:max_pairs]]
            else:
                selected_pairs = [(pair[0], pair[1]) for pair in corr_pairs]
        else:
            # Si no hay matriz de correlación, tomar todos los pares o limitarlos
            all_pairs = list(itertools.combinations(numeric_cols, 2))
            if max_pairs is not None:
                selected_pairs = all_pairs[:max_pairs]
            else:
                selected_pairs = all_pairs

        if len(selected_pairs) == 0:
            self.logger.warning("No hay pares válidos para análisis bivariado")
            return

        # Calcular número de filas y columnas para subplots
        n_plots = len(selected_pairs)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        fig.suptitle('Análisis Bivariado - Scatter Plots', fontsize=16, fontweight='bold', y=0.98)

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, (var1, var2) in enumerate(selected_pairs):
            row = i // n_cols
            col_idx = i % n_cols
            ax = axes[row, col_idx]

            # Filtrar datos válidos
            mask = df[[var1, var2]].notna().all(axis=1)
            x_data = df.loc[mask, var1]
            y_data = df.loc[mask, var2]

            if len(x_data) > 0:
                # Crear scatter plot
                ax.scatter(x_data, y_data, alpha=0.6, s=30, color='darkgreen')

                # Agregar línea de tendencia
                if len(x_data) > 1:
                    z = np.polyfit(x_data, y_data, 1)
                    p = np.poly1d(z)
                    ax.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2)

                    # Calcular correlaciones
                    pearson_corr, pearson_p = pearsonr(x_data, y_data)
                    spearman_corr, spearman_p = spearmanr(x_data, y_data)

                    ax.text(0.05, 0.95, f'Pearson: {pearson_corr:.3f}\nSpearman: {spearman_corr:.3f}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

                ax.set_xlabel(var1)
                ax.set_ylabel(var2)
                ax.set_title(f'{var1} vs {var2}', fontsize=10)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Sin datos válidos', transform=ax.transAxes, 
                       ha='center', va='center')
                ax.set_title(f'{var1} vs {var2}', fontsize=10)

        # Ocultar subplots vacíos
        for i in range(n_plots, n_rows * n_cols):
            row = i // n_cols
            col_idx = i % n_cols
            axes[row, col_idx].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Dejar espacio para el título
        plt.savefig(f'{self.results_path}/bivariate_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.results['bivariate_pairs_count'] = len(selected_pairs)
        self.logger.info("✅ Análisis bivariado completado")

    def create_box_plots(self, df, max_plots=None):
        """
        Crea box plots de atributos numéricos.

        Args:
            df (pd.DataFrame): Dataset con los datos
            max_plots (int): Número máximo de gráficos a crear
        """
        self.logger.info("Creando box plots de atributos numéricos...")

        # Obtener variables numéricas
        numeric_cols = [col for col in self.get_numeric_columns_available(df) if col in df.columns]

        if len(numeric_cols) == 0:
            self.logger.warning("No hay variables numéricas para box plots")
            return

        # Limitar número de gráficos solo si se especifica max_plots
        if max_plots is not None and len(numeric_cols) > max_plots:
            numeric_cols = numeric_cols[:max_plots]

        # Calcular número de filas y columnas para subplots
        n_plots = len(numeric_cols)
        n_cols = 4
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        fig.suptitle('Box Plots de Atributos Numéricos', fontsize=16, fontweight='bold', y=0.98)

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, col in enumerate(numeric_cols):
            row = i // n_cols
            col_idx = i % n_cols
            ax = axes[row, col_idx]

            # Filtrar datos válidos
            data = df[col].dropna()

            if len(data) > 0:
                # Crear box plot
                box_plot = ax.boxplot(data, patch_artist=True)
                box_plot['boxes'][0].set_facecolor('lightcoral')
                box_plot['boxes'][0].set_alpha(0.7)

                # Agregar estadísticas
                stats_text = f"""
                Media: {data.mean():.2f}
                Mediana: {data.median():.2f}
                Q1: {data.quantile(0.25):.2f}
                Q3: {data.quantile(0.75):.2f}
                """

                ax.text(1.1, 0.5, stats_text, transform=ax.transAxes, 
                       verticalalignment='center', fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

                ax.set_title(col, fontsize=10)
                ax.set_ylabel('Valores')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Sin datos válidos', transform=ax.transAxes, 
                       ha='center', va='center')
                ax.set_title(col, fontsize=10)

        # Ocultar subplots vacíos
        for i in range(n_plots, n_rows * n_cols):
            row = i // n_cols
            col_idx = i % n_cols
            axes[row, col_idx].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Dejar espacio para el título
        plt.savefig(f'{self.results_path}/box_plots_numeric.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Crear box plots agrupados por variable categórica objetivo si existe
        if self.TARGET_CATEGORICAL in df.columns:
            self.create_grouped_box_plots(df, numeric_cols)  # Mostrar todas las variables

        self.results['box_plots_count'] = len(numeric_cols)
        self.logger.info("✅ Box plots completados")

    def create_grouped_box_plots(self, df, numeric_cols):
        """
        Crea box plots agrupados por la variable categórica objetivo.

        Args:
            df (pd.DataFrame): Dataset con los datos
            numeric_cols (list): Lista de columnas numéricas
        """
        self.logger.info("Creando box plots agrupados por variable objetivo...")
        
        if len(numeric_cols) == 0:
            return
        
        # Advertencia si hay muchas variables
        if len(numeric_cols) > 20:
            self.logger.warning(f"Generando {len(numeric_cols)} box plots agrupados. Esto puede generar archivos muy grandes.")

        # Calcular número de filas y columnas para subplots
        n_plots = len(numeric_cols)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        fig.suptitle(f'Box Plots Agrupados por {self.TARGET_CATEGORICAL}', fontsize=16, fontweight='bold', y=0.98)

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, col in enumerate(numeric_cols):
            row = i // n_cols
            col_idx = i % n_cols
            ax = axes[row, col_idx]

            # Filtrar datos válidos
            mask = df[[col, self.TARGET_CATEGORICAL]].notna().all(axis=1)
            df_valid = df.loc[mask]

            if len(df_valid) > 0:
                # Crear box plot agrupado
                sns.boxplot(data=df_valid, x=self.TARGET_CATEGORICAL, y=col, ax=ax)
                ax.set_title(f'{col} por {self.TARGET_CATEGORICAL}', fontsize=10)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Sin datos válidos', transform=ax.transAxes, 
                       ha='center', va='center')
                ax.set_title(f'{col} por {self.TARGET_CATEGORICAL}', fontsize=10)

        # Ocultar subplots vacíos
        for i in range(n_plots, n_rows * n_cols):
            row = i // n_cols
            col_idx = i % n_cols
            axes[row, col_idx].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Dejar espacio para el título
        plt.savefig(f'{self.results_path}/box_plots_grouped.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info("✅ Box plots agrupados completados")

    def analyze_categorical_variables(self, df):
        """
        Analiza y visualiza variables categóricas con gráficas de barras.
        
        Args:
            df (pd.DataFrame): Dataset con los datos
        """
        self.logger.info("Analizando variables categóricas...")
        
        # Usar el método de la clase base para identificar tipos de variables
        variable_types = self.identify_variable_types(df)
        categorical_vars = variable_types['categorical']
        
        if len(categorical_vars) == 0:
            self.logger.warning("No se encontraron variables categóricas para analizar")
            return
        
        self.logger.info(f"Variables categóricas encontradas: {len(categorical_vars)}")
        self.logger.info(f"Lista de variables categóricas: {categorical_vars}")
        
        # Crear gráficas de barras individuales
        self.create_categorical_bar_plots(df, categorical_vars)
        
        # Crear análisis de asociación con variable objetivo
        if self.TARGET_CATEGORICAL in df.columns:
            self.create_categorical_association_analysis(df, categorical_vars)
        
        self.results['categorical_analysis'] = {
            'variables_analyzed': len(categorical_vars),
            'variables_list': categorical_vars
        }

    def create_categorical_bar_plots(self, df, categorical_vars):
        """
        Crea gráficas de barras con conteo y porcentaje para variables categóricas.
        
        Args:
            df (pd.DataFrame): Dataset con los datos
            categorical_vars (list): Lista de variables categóricas
        """
        self.logger.info("Creando gráficas de barras para variables categóricas...")
        
        # Advertencia si hay muchas variables
        if len(categorical_vars) > 20:
            self.logger.warning(f"Generando {len(categorical_vars)} gráficas de barras. Esto puede generar archivos grandes.")
        
        # Calcular número de filas y columnas para subplots
        n_plots = len(categorical_vars)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        fig.suptitle('Distribución de Variables Categóricas', fontsize=16, fontweight='bold', y=0.98)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, var in enumerate(categorical_vars):
            row = i // n_cols
            col_idx = i % n_cols
            ax = axes[row, col_idx]
            
            # Filtrar datos válidos
            data = df[var].dropna()
            
            if len(data) > 0:
                # Calcular frecuencias
                value_counts = data.value_counts()
                percentages = (value_counts / len(data) * 100).round(1)
                
                # Limitar categorías si hay demasiadas
                if len(value_counts) > 10:
                    value_counts = value_counts.head(10)
                    percentages = percentages.head(10)
                    title_suffix = f" (Top 10 de {data.nunique()} categorías)"
                else:
                    title_suffix = ""
                
                # Crear gráfica de barras
                bars = ax.bar(range(len(value_counts)), value_counts.values, 
                             color='steelblue', alpha=0.7, edgecolor='black')
                
                # Agregar etiquetas con conteo y porcentaje
                for j, (bar, count, pct) in enumerate(zip(bars, value_counts.values, percentages.values)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{count}\n({pct}%)', ha='center', va='bottom', fontsize=8)
                
                # Configurar ejes
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45, ha='right', fontsize=9)
                ax.set_ylabel('Frecuencia')
                ax.set_title(f'{var}{title_suffix}', fontsize=10)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Agregar información adicional
                info_text = f'Total: {len(data)}\nCategorías: {data.nunique()}\nMás frecuente: {value_counts.index[0]}'
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                
            else:
                ax.text(0.5, 0.5, 'Sin datos válidos', transform=ax.transAxes, 
                       ha='center', va='center')
                ax.set_title(var, fontsize=10)
        
        # Ocultar subplots vacíos
        for i in range(n_plots, n_rows * n_cols):
            row = i // n_cols
            col_idx = i % n_cols
            axes[row, col_idx].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Dejar espacio para el título
        plt.savefig(f'{self.results_path}/categorical_bar_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("✅ Gráficas de barras categóricas completadas")

    def create_categorical_association_analysis(self, df, categorical_vars):
        """
        Crea análisis de asociación entre variables categóricas y variable objetivo.
        
        Args:
            df (pd.DataFrame): Dataset con los datos
            categorical_vars (list): Lista de variables categóricas
        """
        self.logger.info("Creando análisis de asociación categórica...")
        
        # Filtrar variables (excluir la variable objetivo)
        analysis_vars = [var for var in categorical_vars if var != self.TARGET_CATEGORICAL]
        
        if len(analysis_vars) == 0:
            self.logger.warning("No hay variables categóricas para análisis de asociación")
            return
        
        # Advertencia si hay muchas variables
        if len(analysis_vars) > 20:
            self.logger.warning(f"Generando {len(analysis_vars)} gráficas de asociación. Esto puede generar archivos muy grandes.")
        
        # Calcular número de filas y columnas para subplots - MOSTRAR TODAS
        n_plots = len(analysis_vars)  # Sin límite - mostrar todas las variables
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        fig.suptitle(f'Asociación con Variable Objetivo: {self.TARGET_CATEGORICAL}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        association_results = []
        
        for i, var in enumerate(analysis_vars):
            row = i // n_cols
            col_idx = i % n_cols
            ax = axes[row, col_idx]
            
            # Filtrar datos válidos
            mask = df[[var, self.TARGET_CATEGORICAL]].notna().all(axis=1)
            df_valid = df.loc[mask]
            
            if len(df_valid) > 0:
                # Crear tabla de contingencia
                contingency_table = pd.crosstab(df_valid[var], df_valid[self.TARGET_CATEGORICAL])
                
                # Calcular porcentajes por fila (distribución condicional)
                contingency_pct = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
                
                # Crear gráfica de barras apiladas
                contingency_pct.plot(kind='bar', stacked=True, ax=ax, 
                                   colormap='viridis', alpha=0.8)
                
                ax.set_title(f'{var} vs {self.TARGET_CATEGORICAL}', fontsize=10)
                ax.set_xlabel(var)
                ax.set_ylabel('Porcentaje')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Calcular V de Cramér si es posible
                try:
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    n = contingency_table.sum().sum()
                    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                    
                    # Agregar estadísticas
                    stats_text = f'V de Cramér: {cramers_v:.3f}\np-valor: {p_value:.3f}'
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                    
                    association_results.append({
                        'variable': var,
                        'cramers_v': cramers_v,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'categories': contingency_table.shape[0]
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error calculando asociación para {var}: {e}")
                
            else:
                ax.text(0.5, 0.5, 'Sin datos válidos', transform=ax.transAxes, 
                       ha='center', va='center')
                ax.set_title(f'{var} vs {self.TARGET_CATEGORICAL}', fontsize=10)
        
        # Ocultar subplots vacíos
        for i in range(n_plots, n_rows * n_cols):
            row = i // n_cols
            col_idx = i % n_cols
            axes[row, col_idx].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Dejar espacio para el título
        plt.savefig(f'{self.results_path}/categorical_association_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Guardar resultados de asociación
        if association_results:
            association_df = pd.DataFrame(association_results)
            association_df = association_df.sort_values('cramers_v', ascending=False)
            association_df.to_csv(f'{self.results_path}/categorical_association_results.csv', index=False)
            
            self.results['categorical_associations'] = association_results
        
        self.logger.info("✅ Análisis de asociación categórica completado")

    def run_analysis(self):
        self.logger.info("Iniciando análisis de visualización de datos")
        self.logger.info(f"Dataset: {self.dataset_path}")
        self.logger.info(f"Resultados se guardarán en: {self.results_path}")

        # Validar que el dataset exista
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"No se encontró el dataset en: {self.dataset_path}")

        # Crear directorio de resultados
        self.create_results_directory()

        # Cargar datos
        df = self.load_data()

        # Validar que existan las columnas objetivo
        self.validate_target_variables(df)

        # Ejecutar análisis
        self.logger.info("Ejecutando análisis de distribución de variable respuesta...")
        self.analyze_target_distribution(df)

        self.logger.info("Ejecutando análisis de matriz de correlación...")
        self.create_correlation_matrix(df)

        self.logger.info("Ejecutando scatter plots vs variable objetivo...")
        self.create_scatter_plots_vs_target(df)  # Sin límite - mostrará todas las variables
        
        self.logger.info("Ejecutando análisis bivariado...")
        self.create_bivariate_analysis(df, max_pairs=50)  # Aumentar límite para más pares
        
        self.logger.info("Ejecutando box plots...")
        self.create_box_plots(df)  # Sin límite - mostrará todas las variables
        
        self.logger.info("Ejecutando análisis de variables categóricas...")
        self.analyze_categorical_variables(df)

        self.logger.info("✅ Análisis de visualización completado exitosamente")

        return self.results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Análisis de visualización de datos para EDA')
    parser.add_argument('--dataset', '-d', type=str, required=True,
                       help='Ruta al archivo CSV del dataset (requerido)')
    parser.add_argument('--results', '-r', type=str, required=True,
                       help='Nombre del folder para guardar resultados (requerido, se creará en reports/)')

    args = parser.parse_args()

    # Crear y ejecutar analizador
    analyzer = DataVisualizationAnalyzer(args.dataset, args.results)

    try:
        analyzer.run_analysis()
        analyzer.logger.info("✅ Análisis completado exitosamente")
    except FileNotFoundError as e:
        analyzer.logger.error(f"❌ Error: {e}")
        raise
    except ValueError as e:
        analyzer.logger.error(f"❌ Error de validación: {e}")
        raise
    except Exception as e:
        analyzer.logger.error(f"❌ Error inesperado: {e}")
        raise
    finally:
        analyzer.close()


if __name__ == "__main__":
    main()
