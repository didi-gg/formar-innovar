"""
Script para filtrado estadístico de variables usando pruebas chi-cuadrado, 
correlación y análisis ANOVA
"""

import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from scipy.stats import chi2_contingency, pearsonr, spearmanr, f_oneway

# Configuración de warnings y logging
warnings.filterwarnings('ignore')

# Configurar matplotlib para suprimir mensajes de debug
import matplotlib
matplotlib.set_loglevel("WARNING")
import matplotlib.pyplot as plt
plt.set_loglevel("WARNING")

# Configurar logging para suprimir mensajes de debug
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# Agregar el directorio padre al path para importar utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.eda_analysis_base import EDAAnalysisBase

class StatisticalFilter(EDAAnalysisBase):
    """
    Clase para realizar filtrado estadístico de variables usando:
    - Pruebas chi-cuadrado para variables categóricas
    - Correlación de Pearson/Spearman para variables continuas
    - Análisis ANOVA para evaluar efectos de variables categóricas sobre la variable objetivo
    """

    def _initialize_analysis_attributes(self):
        """Inicializar atributos específicos del análisis de filtrado estadístico."""
        # Resultados del análisis
        self.results = {
            'chi2_results': [],
            'correlation_results': [],
            'anova_results': [],
            'selected_features': {
                'chi2_significant': [],
                'correlation_significant': [],
                'anova_significant': []
            }
        }

    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """Cargar y preparar los datos."""
        df = super().load_data(file_path)

        # Filtrar columnas excluidas
        features_to_analyze = self.get_valid_features(df, exclude_targets=False)
        df_filtered = df[features_to_analyze + [self.TARGET_CATEGORICAL, self.TARGET_NUMERIC]].copy()

        self.logger.info(f"Variables a analizar: {len(features_to_analyze)}")
        return df_filtered

    def identify_variable_types(self, df: pd.DataFrame) -> dict:
        """Identificar tipos de variables (categóricas vs continuas)."""
        # Obtener características válidas excluyendo variables objetivo
        valid_features = self.get_valid_features(df, exclude_targets=True)

        # Usar el método de la clase base
        return super().identify_variable_types(df, valid_features)

    def chi_square_test(self, df: pd.DataFrame, categorical_vars: list) -> list:
        """Realizar pruebas chi-cuadrado para variables categóricas."""
        self.logger.info("Realizando pruebas chi-cuadrado...")
        chi2_results = []

        for var in categorical_vars:
            try:
                # Crear tabla de contingencia
                contingency_table = pd.crosstab(df[var], df[self.TARGET_CATEGORICAL])

                # Realizar prueba chi-cuadrado
                chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

                # Calcular V de Cramér (medida de asociación)
                n = contingency_table.sum().sum()
                cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))

                result = {
                    'variable': var,
                    'chi2_statistic': chi2_stat,
                    'p_value': p_value,
                    'degrees_freedom': dof,
                    'cramers_v': cramers_v,
                    'significant': p_value < self.ALPHA,
                    'effect_size': self.interpret_cramers_v(cramers_v)
                }

                chi2_results.append(result)

                if result['significant']:
                    self.results['selected_features']['chi2_significant'].append(var)

            except Exception as e:
                self.logger.warning(f"Error en prueba chi-cuadrado para {var}: {str(e)}")

        # Ordenar por p-value
        chi2_results.sort(key=lambda x: x['p_value'])
        self.results['chi2_results'] = chi2_results

        self.logger.info(f"Variables significativas (chi-cuadrado): {len(self.results['selected_features']['chi2_significant'])}")
        return chi2_results

    def correlation_analysis(self, df: pd.DataFrame, continuous_vars: list) -> list:
        """Realizar análisis de correlación para variables continuas."""
        self.logger.info("Realizando análisis de correlación...")
        correlation_results = []

        for var in continuous_vars:
            try:
                # Filtrar valores no nulos
                mask = df[[var, self.TARGET_NUMERIC]].notna().all(axis=1)
                if mask.sum() < 10:  # Muy pocos datos válidos
                    continue

                x = df.loc[mask, var]
                y = df.loc[mask, self.TARGET_NUMERIC]

                # Correlación de Pearson
                pearson_r, pearson_p = pearsonr(x, y)

                # Correlación de Spearman
                spearman_r, spearman_p = spearmanr(x, y)

                result = {
                    'variable': var,
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'pearson_significant': pearson_p < self.ALPHA,
                    'spearman_significant': spearman_p < self.ALPHA,
                    'effect_size_pearson': self.interpret_correlation(abs(pearson_r)),
                    'effect_size_spearman': self.interpret_correlation(abs(spearman_r)),
                    'n_observations': mask.sum()
                }

                correlation_results.append(result)

                # Agregar a variables significativas si alguna correlación es significativa
                if result['pearson_significant'] or result['spearman_significant']:
                    self.results['selected_features']['correlation_significant'].append(var)

            except Exception as e:
                self.logger.warning(f"Error en análisis de correlación para {var}: {str(e)}")

        # Ordenar por p-value de Pearson
        correlation_results.sort(key=lambda x: x['pearson_p'])
        self.results['correlation_results'] = correlation_results

        self.logger.info(f"Variables significativas (correlación): {len(self.results['selected_features']['correlation_significant'])}")
        return correlation_results

    def anova_analysis(self, df: pd.DataFrame, categorical_vars: list) -> list:
        """Realizar análisis ANOVA para variables categóricas vs variable objetivo numérica."""
        self.logger.info("Realizando análisis ANOVA...")
        anova_results = []

        for var in categorical_vars:
            try:
                # Filtrar valores no nulos
                mask = df[[var, self.TARGET_NUMERIC]].notna().all(axis=1)
                if mask.sum() < 10:
                    continue

                # Agrupar datos por categorías
                groups = []
                categories = df.loc[mask, var].unique()

                for category in categories:
                    group_data = df.loc[mask & (df[var] == category), self.TARGET_NUMERIC]
                    if len(group_data) >= 2:  # Al menos 2 observaciones por grupo
                        groups.append(group_data)

                if len(groups) < 2:  # Necesitamos al menos 2 grupos
                    continue

                # Realizar ANOVA
                f_statistic, p_value = f_oneway(*groups)

                # Calcular eta cuadrado (tamaño del efecto)
                ss_between = sum(len(group) * (np.mean(group) - np.mean(df.loc[mask, self.TARGET_NUMERIC]))**2 
                               for group in groups)
                ss_total = np.sum((df.loc[mask, self.TARGET_NUMERIC] - np.mean(df.loc[mask, self.TARGET_NUMERIC]))**2)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0

                result = {
                    'variable': var,
                    'f_statistic': f_statistic,
                    'p_value': p_value,
                    'eta_squared': eta_squared,
                    'significant': p_value < self.ALPHA,
                    'effect_size': self.interpret_eta_squared(eta_squared),
                    'n_groups': len(groups),
                    'n_observations': mask.sum()
                }

                anova_results.append(result)

                if result['significant']:
                    self.results['selected_features']['anova_significant'].append(var)

            except Exception as e:
                self.logger.warning(f"Error en análisis ANOVA para {var}: {str(e)}")

        # Ordenar por p-value
        anova_results.sort(key=lambda x: x['p_value'])
        self.results['anova_results'] = anova_results

        self.logger.info(f"Variables significativas (ANOVA): {len(self.results['selected_features']['anova_significant'])}")
        return anova_results

    def create_visualizations(self, output_dir: str):
        """Crear visualizaciones de los resultados."""
        self.logger.info("Creando visualizaciones...")

        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)

        # 1. Gráfico de barras para chi-cuadrado
        self._plot_chi2_results(output_dir)

        # 2. Gráfico de correlaciones
        self._plot_correlation_results(output_dir)

        # 3. Gráfico ANOVA
        self._plot_anova_results(output_dir)

    def _plot_chi2_results(self, output_dir: str):
        """Gráfico de resultados chi-cuadrado."""
        if not self.results['chi2_results']:
            return

        df_chi2 = pd.DataFrame(self.results['chi2_results'])
        df_chi2 = df_chi2.head(20)  # Top 20 más significativos

        # Revertir el orden para que los más significativos aparezcan arriba
        df_chi2 = df_chi2.iloc[::-1]

        plt.figure(figsize=(12, 8))
        colors = ['red' if p < self.ALPHA else 'blue' for p in df_chi2['p_value']]

        # Manejar p-valores extremadamente pequeños para evitar infinitos
        log_p_values = []
        for p in df_chi2['p_value']:
            if p == 0 or p < 1e-300:  # P-valor extremadamente pequeño
                log_p_values.append(300)  # Límite máximo para visualización
            else:
                log_p_values.append(-np.log10(p))

        plt.barh(df_chi2['variable'], log_p_values, color=colors, alpha=0.7)
        plt.axvline(-np.log10(self.ALPHA), color='red', linestyle='--', 
                   label=f'Umbral significancia (α={self.ALPHA})')
        plt.xlabel('-log10(p-value)')
        plt.ylabel('Variables')
        plt.title('Pruebas Chi-cuadrado: Asociación con Variable Objetivo')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/chi2_results.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_correlation_results(self, output_dir: str):
        """Gráfico de resultados de correlación."""
        if not self.results['correlation_results']:
            return

        df_corr = pd.DataFrame(self.results['correlation_results'])
        df_corr = df_corr.head(20)  # Top 20 más significativos

        # Revertir el orden para que los más significativos aparezcan arriba
        df_corr = df_corr.iloc[::-1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Correlación de Pearson
        colors = ['red' if p < self.ALPHA else 'blue' for p in df_corr['pearson_p']]
        ax1.barh(df_corr['variable'], df_corr['pearson_r'], color=colors, alpha=0.7)
        ax1.set_xlabel('Correlación de Pearson')
        ax1.set_ylabel('Variables')
        ax1.set_title('Correlación de Pearson con Nota Final')
        ax1.axvline(0, color='black', linestyle='-', alpha=0.3)

        # Correlación de Spearman
        colors = ['red' if p < self.ALPHA else 'blue' for p in df_corr['spearman_p']]
        ax2.barh(df_corr['variable'], df_corr['spearman_r'], color=colors, alpha=0.7)
        ax2.set_xlabel('Correlación de Spearman')
        ax2.set_ylabel('Variables')
        ax2.set_title('Correlación de Spearman con Nota Final')
        ax2.axvline(0, color='black', linestyle='-', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_results.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_anova_results(self, output_dir: str):
        """Gráfico de resultados ANOVA."""
        if not self.results['anova_results']:
            return

        df_anova = pd.DataFrame(self.results['anova_results'])
        df_anova = df_anova.head(20)  # Top 20 más significativos

        # Revertir el orden para que los más significativos aparezcan arriba
        df_anova = df_anova.iloc[::-1]

        plt.figure(figsize=(12, 8))
        colors = ['red' if p < self.ALPHA else 'blue' for p in df_anova['p_value']]

        # Manejar p-valores extremadamente pequeños para evitar infinitos
        log_p_values = []
        for p in df_anova['p_value']:
            if p == 0 or p < 1e-300:  # P-valor extremadamente pequeño
                log_p_values.append(300)  # Límite máximo para visualización
            else:
                log_p_values.append(-np.log10(p))

        plt.barh(df_anova['variable'], log_p_values, color=colors, alpha=0.7)
        plt.axvline(-np.log10(self.ALPHA), color='red', linestyle='--', 
                   label=f'Umbral significancia (α={self.ALPHA})')
        plt.xlabel('-log10(p-value)')
        plt.ylabel('Variables')
        plt.title('Análisis ANOVA: Efecto sobre Nota Final')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/anova_results.png", dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self, output_dir: str):
        """Guardar resultados en archivos CSV."""
        self.logger.info("Guardando resultados...")

        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)

        # Guardar resultados chi-cuadrado
        if self.results['chi2_results']:
            df_chi2 = pd.DataFrame(self.results['chi2_results'])
            self.save_to_csv(df_chi2, f"{output_dir}/chi2_test_results.csv")

        # Guardar resultados correlación
        if self.results['correlation_results']:
            df_corr = pd.DataFrame(self.results['correlation_results'])
            self.save_to_csv(df_corr, f"{output_dir}/correlation_analysis_results.csv")

        # Guardar resultados ANOVA
        if self.results['anova_results']:
            df_anova = pd.DataFrame(self.results['anova_results'])
            self.save_to_csv(df_anova, f"{output_dir}/anova_analysis_results.csv")

        # Guardar listas de variables significativas
        all_significant = set()
        all_significant.update(self.results['selected_features']['chi2_significant'])
        all_significant.update(self.results['selected_features']['correlation_significant'])
        all_significant.update(self.results['selected_features']['anova_significant'])

        significant_vars_df = pd.DataFrame({
            'variable': list(all_significant),
            'chi2_significant': [var in self.results['selected_features']['chi2_significant'] for var in all_significant],
            'correlation_significant': [var in self.results['selected_features']['correlation_significant'] for var in all_significant],
            'anova_significant': [var in self.results['selected_features']['anova_significant'] for var in all_significant]
        })

        self.save_to_csv(significant_vars_df, f"{output_dir}/selected_features_statistical.csv")

    def run_analysis(self):
        """Ejecutar análisis completo."""
        self.logger.info("Iniciando análisis estadístico de filtrado de variables...")
        self.logger.info(f"Dataset: {self.dataset_path}")
        self.logger.info(f"Resultados se guardarán en: {self.results_path}")

        # Validar que el dataset exista
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"No se encontró el dataset en: {self.dataset_path}")

        # Crear directorio de resultados si no existe
        self.create_results_directory()

        # Cargar datos
        df = self.load_data(self.dataset_path)

        # Identificar tipos de variables
        var_types = self.identify_variable_types(df)

        # Realizar pruebas estadísticas
        if var_types['categorical']:
            self.chi_square_test(df, var_types['categorical'])
            self.anova_analysis(df, var_types['categorical'])

        if var_types['continuous']:
            self.correlation_analysis(df, var_types['continuous'])

        # Crear visualizaciones
        self.create_visualizations(self.results_path)

        # Guardar resultados
        self.save_results(self.results_path)

        # Resumen final
        self.logger.info("=== RESUMEN DEL ANÁLISIS ESTADÍSTICO ===")
        self.logger.info(f"Variables categóricas: {var_types['categorical']}")
        self.logger.info(f"Variables continuas: {var_types['continuous']}")
        self.logger.info(f"Variables categóricas analizadas: {len(var_types['categorical'])}")
        self.logger.info(f"Variables continuas analizadas: {len(var_types['continuous'])}")
        self.logger.info(f"Variables significativas (Chi-cuadrado): {len(self.results['selected_features']['chi2_significant'])}")
        self.logger.info(f"Variables significativas (Correlación): {len(self.results['selected_features']['correlation_significant'])}")
        self.logger.info(f"Variables significativas (ANOVA): {len(self.results['selected_features']['anova_significant'])}")

        total_significant = len(set().union(
            self.results['selected_features']['chi2_significant'],
            self.results['selected_features']['correlation_significant'],
            self.results['selected_features']['anova_significant']
        ))
        self.logger.info(f"Total de variables significativas únicas: {total_significant}")

        self.logger.info("Análisis completado exitosamente!")

def main():
    """Función principal."""
    import argparse

    parser = argparse.ArgumentParser(description='Análisis de filtrado estadístico de variables')
    parser.add_argument('--dataset', '-d', type=str, required=True,
                       help='Ruta al archivo CSV del dataset (requerido)')
    parser.add_argument('--results', '-r', type=str, required=True,
                       help='Nombre del folder para guardar resultados (requerido, se creará en reports/)')

    args = parser.parse_args()

    # Crear y ejecutar analizador
    analyzer = StatisticalFilter(args.dataset, args.results)

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
