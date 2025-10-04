"""
Script para análisis de homogeneidad de grupos con pruebas de normalidad,
homogeneidad de varianzas y comparaciones entre grupos basado en BaseScript.
"""

import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import (
    shapiro, kstest, anderson, levene, bartlett, 
    ttest_ind, mannwhitneyu, f_oneway, kruskal,
    ks_2samp, normaltest, jarque_bera
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de warnings y logging
warnings.filterwarnings('ignore')

# Configurar matplotlib para suprimir mensajes de debug
import matplotlib
matplotlib.set_loglevel("WARNING")
plt.set_loglevel("WARNING")

# Configurar logging para suprimir mensajes de debug
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# Agregar el directorio padre al path para importar utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.eda_analysis_base import EDAAnalysisBase

class HomogeneityAnalysis(EDAAnalysisBase):
    """
    Clase para análisis de homogeneidad de grupos que incluye:
    1. Pruebas de normalidad (Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling)
    2. Pruebas de homogeneidad de varianzas (Levene, Bartlett)
    3. Comparaciones entre grupos (t-Student, Mann-Whitney, ANOVA, Kruskal-Wallis)
    4. Visualizaciones (histogramas, Q-Q plots)
    """

    def _initialize_analysis_attributes(self):
        """Inicializar atributos específicos del análisis de homogeneidad."""
        # Resultados del análisis
        self.results = {
            'normality_tests': {},
            'variance_homogeneity_tests': {},
            'group_comparisons': {},
            'summary': {}
        }

    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """Cargar y preparar los datos."""
        df = super().load_data(file_path)

        # Filtrar solo las columnas numéricas disponibles y variables de agrupación
        available_numeric = self.get_numeric_columns_available(df)
        available_groups = [col for col in self.GROUP_VARIABLES if col in df.columns]

        if not available_numeric:
            raise ValueError("No se encontraron columnas numéricas especificadas en el dataset")

        if not available_groups:
            raise ValueError("No se encontraron variables de agrupación en el dataset")

        # Seleccionar columnas relevantes
        columns_to_keep = available_numeric + available_groups
        df_filtered = df[columns_to_keep].copy()

        self.logger.info(f"Variables numéricas disponibles: {len(available_numeric)}")
        self.logger.info(f"Variables de agrupación disponibles: {available_groups}")

        # Actualizar listas con columnas disponibles
        self.numeric_columns = available_numeric
        self.group_variables = available_groups

        return df_filtered

    def test_normality(self, df: pd.DataFrame) -> dict:
        """Realizar pruebas de normalidad para todas las variables numéricas."""
        self.logger.info("Realizando pruebas de normalidad...")
        normality_results = {}

        for col in self.numeric_columns:
            if col not in df.columns:
                continue

            # Filtrar valores no nulos
            data = df[col].dropna()

            if len(data) < 3:  # Muy pocos datos
                continue

            col_results = {
                'variable': col,
                'n_observations': len(data),
                'mean': data.mean(),
                'std': data.std(),
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data)
            }

            try:
                # 1. Shapiro-Wilk (recomendado para muestras pequeñas y medias)
                if len(data) <= 5000:  # Shapiro-Wilk tiene límite de muestra
                    shapiro_stat, shapiro_p = shapiro(data)
                    col_results['shapiro_statistic'] = shapiro_stat
                    col_results['shapiro_p_value'] = shapiro_p
                    col_results['shapiro_normal'] = shapiro_p > self.ALPHA
                else:
                    col_results['shapiro_statistic'] = np.nan
                    col_results['shapiro_p_value'] = np.nan
                    col_results['shapiro_normal'] = np.nan

                # 2. Kolmogorov-Smirnov con corrección Lilliefors
                # Usamos kstest con distribución normal estándar
                data_standardized = (data - data.mean()) / data.std()
                ks_stat, ks_p = kstest(data_standardized, 'norm')
                col_results['ks_statistic'] = ks_stat
                col_results['ks_p_value'] = ks_p
                col_results['ks_normal'] = ks_p > self.ALPHA

                # 3. Anderson-Darling (buena sensibilidad en colas)
                anderson_result = anderson(data, dist='norm')
                col_results['anderson_statistic'] = anderson_result.statistic
                # Anderson-Darling usa valores críticos, no p-values
                # Usamos el nivel de significancia del 5% (índice 2)
                critical_value_5pct = anderson_result.critical_values[2]  # 5%
                col_results['anderson_critical_5pct'] = critical_value_5pct
                col_results['anderson_normal'] = anderson_result.statistic < critical_value_5pct

                # 4. D'Agostino-Pearson (omnibus test)
                dagostino_stat, dagostino_p = normaltest(data)
                col_results['dagostino_statistic'] = dagostino_stat
                col_results['dagostino_p_value'] = dagostino_p
                col_results['dagostino_normal'] = dagostino_p > self.ALPHA

                # 5. Jarque-Bera test
                jb_stat, jb_p = jarque_bera(data)
                col_results['jarque_bera_statistic'] = jb_stat
                col_results['jarque_bera_p_value'] = jb_p
                col_results['jarque_bera_normal'] = jb_p > self.ALPHA

                # Resumen de normalidad (mayoría de tests)
                normal_tests = [
                    col_results.get('shapiro_normal', None),
                    col_results['ks_normal'],
                    col_results['anderson_normal'],
                    col_results['dagostino_normal'],
                    col_results['jarque_bera_normal']
                ]

                # Filtrar valores None
                valid_tests = [t for t in normal_tests if t is not None]
                if valid_tests:
                    col_results['overall_normal'] = sum(valid_tests) > len(valid_tests) / 2
                    col_results['normal_test_count'] = sum(valid_tests)
                    col_results['total_tests'] = len(valid_tests)
                else:
                    col_results['overall_normal'] = False
                    col_results['normal_test_count'] = 0
                    col_results['total_tests'] = 0

            except Exception as e:
                self.logger.warning(f"Error en pruebas de normalidad para {col}: {str(e)}")
                col_results['error'] = str(e)

            normality_results[col] = col_results

        self.results['normality_tests'] = normality_results

        # Contar variables normales
        normal_vars = sum(1 for result in normality_results.values() 
                         if result.get('overall_normal', False))
        self.logger.info(f"Variables con distribución normal: {normal_vars}/{len(normality_results)}")

        return normality_results

    def test_variance_homogeneity(self, df: pd.DataFrame) -> dict:
        """Realizar pruebas de homogeneidad de varianzas entre grupos."""
        self.logger.info("Realizando pruebas de homogeneidad de varianzas...")
        variance_results = {}

        for group_var in self.group_variables:
            if group_var not in df.columns:
                continue

            group_results = {}

            for numeric_var in self.numeric_columns:
                if numeric_var not in df.columns:
                    continue

                # Filtrar datos válidos
                mask = df[[group_var, numeric_var]].notna().all(axis=1)
                if mask.sum() < 10:
                    continue

                # Obtener grupos
                groups = df.loc[mask, group_var].unique()
                if len(groups) < 2:
                    continue

                # Preparar datos por grupo
                group_data = []
                group_info = []

                for group in groups:
                    group_mask = mask & (df[group_var] == group)
                    data = df.loc[group_mask, numeric_var]
                    if len(data) >= 2:  # Al menos 2 observaciones por grupo
                        group_data.append(data)
                        group_info.append({
                            'group': group,
                            'n': len(data),
                            'mean': data.mean(),
                            'std': data.std(),
                            'var': data.var()
                        })

                if len(group_data) < 2:
                    continue

                var_results = {
                    'variable': numeric_var,
                    'group_variable': group_var,
                    'groups_info': group_info,
                    'n_groups': len(group_data)
                }

                try:
                    # 1. Test de Levene (robusto a desviaciones de normalidad)
                    levene_stat, levene_p = levene(*group_data)
                    var_results['levene_statistic'] = levene_stat
                    var_results['levene_p_value'] = levene_p
                    var_results['levene_homogeneous'] = levene_p > self.ALPHA

                    # 2. Test de Bartlett (más estricto, asume normalidad)
                    bartlett_stat, bartlett_p = bartlett(*group_data)
                    var_results['bartlett_statistic'] = bartlett_stat
                    var_results['bartlett_p_value'] = bartlett_p
                    var_results['bartlett_homogeneous'] = bartlett_p > self.ALPHA

                    # Ratio de varianzas máxima/mínima
                    variances = [data.var() for data in group_data]
                    var_results['max_var'] = max(variances)
                    var_results['min_var'] = min(variances)
                    var_results['variance_ratio'] = max(variances) / min(variances) if min(variances) > 0 else np.inf

                except Exception as e:
                    self.logger.warning(f"Error en pruebas de homogeneidad para {numeric_var} por {group_var}: {str(e)}")
                    var_results['error'] = str(e)

                group_results[numeric_var] = var_results

            variance_results[group_var] = group_results

        self.results['variance_homogeneity_tests'] = variance_results
        return variance_results

    def compare_groups(self, df: pd.DataFrame) -> dict:
        """Realizar comparaciones entre grupos."""
        self.logger.info("Realizando comparaciones entre grupos...")
        comparison_results = {}

        for group_var in self.group_variables:
            if group_var not in df.columns:
                continue

            group_results = {}

            # Obtener grupos únicos
            groups = df[group_var].dropna().unique()
            n_groups = len(groups)

            if n_groups < 2:
                continue

            for numeric_var in self.numeric_columns:
                if numeric_var not in df.columns:
                    continue

                # Filtrar datos válidos
                mask = df[[group_var, numeric_var]].notna().all(axis=1)
                if mask.sum() < 10:
                    continue

                # Preparar datos por grupo
                group_data = []
                group_names = []

                for group in groups:
                    group_mask = mask & (df[group_var] == group)
                    data = df.loc[group_mask, numeric_var]
                    if len(data) >= 2:
                        group_data.append(data)
                        group_names.append(group)

                if len(group_data) < 2:
                    continue

                comp_results = {
                    'variable': numeric_var,
                    'group_variable': group_var,
                    'groups': group_names,
                    'n_groups': len(group_data)
                }

                # Obtener información de normalidad y homogeneidad
                is_normal = self.results['normality_tests'].get(numeric_var, {}).get('overall_normal', False)

                variance_test = self.results['variance_homogeneity_tests'].get(group_var, {}).get(numeric_var, {})
                is_homogeneous = variance_test.get('levene_homogeneous', False)

                comp_results['assumes_normality'] = is_normal
                comp_results['assumes_homogeneity'] = is_homogeneous

                try:
                    if len(group_data) == 2:
                        # Comparación entre 2 grupos
                        group1, group2 = group_data

                        # Estadísticas descriptivas
                        comp_results['group1_mean'] = group1.mean()
                        comp_results['group1_std'] = group1.std()
                        comp_results['group1_n'] = len(group1)
                        comp_results['group2_mean'] = group2.mean()
                        comp_results['group2_std'] = group2.std()
                        comp_results['group2_n'] = len(group2)
                        comp_results['mean_difference'] = group1.mean() - group2.mean()

                        if is_normal and is_homogeneous:
                            # t-Student (paramétrico)
                            t_stat, t_p = ttest_ind(group1, group2)
                            comp_results['test_used'] = 't-Student'
                            comp_results['statistic'] = t_stat
                            comp_results['p_value'] = t_p
                            comp_results['significant'] = t_p < self.ALPHA

                            # Tamaño del efecto (Cohen's d)
                            pooled_std = np.sqrt(((len(group1)-1)*group1.var() + (len(group2)-1)*group2.var()) / 
                                               (len(group1) + len(group2) - 2))
                            cohens_d = (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0
                            comp_results['effect_size'] = cohens_d
                            comp_results['effect_interpretation'] = self.interpret_cohens_d(abs(cohens_d))
                        else:
                            # Mann-Whitney U (no paramétrico)
                            u_stat, u_p = mannwhitneyu(group1, group2, alternative='two-sided')
                            comp_results['test_used'] = 'Mann-Whitney U'
                            comp_results['statistic'] = u_stat
                            comp_results['p_value'] = u_p
                            comp_results['significant'] = u_p < self.ALPHA

                            # Tamaño del efecto (r = Z / sqrt(N))
                            n_total = len(group1) + len(group2)
                            z_score = stats.norm.ppf(u_p/2)  # Aproximación
                            r_effect = abs(z_score) / np.sqrt(n_total)
                            comp_results['effect_size'] = r_effect
                            comp_results['effect_interpretation'] = self.interpret_r_effect(r_effect)

                        # Kolmogorov-Smirnov para comparar distribuciones completas
                        ks_stat, ks_p = ks_2samp(group1, group2)
                        comp_results['ks_statistic'] = ks_stat
                        comp_results['ks_p_value'] = ks_p
                        comp_results['distributions_different'] = ks_p < self.ALPHA

                    else:
                        # Comparación entre más de 2 grupos
                        # Estadísticas descriptivas por grupo
                        group_stats = []
                        for i, (data, name) in enumerate(zip(group_data, group_names)):
                            group_stats.append({
                                'group': name,
                                'mean': data.mean(),
                                'std': data.std(),
                                'n': len(data)
                            })
                        comp_results['group_statistics'] = group_stats

                        if is_normal and is_homogeneous:
                            # ANOVA (paramétrico)
                            f_stat, f_p = f_oneway(*group_data)
                            comp_results['test_used'] = 'ANOVA'
                            comp_results['statistic'] = f_stat
                            comp_results['p_value'] = f_p
                            comp_results['significant'] = f_p < self.ALPHA

                            # Eta cuadrado (tamaño del efecto)
                            ss_between = sum(len(group) * (group.mean() - 
                                           np.concatenate(group_data).mean())**2 for group in group_data)
                            ss_total = sum((np.concatenate(group_data) - 
                                          np.concatenate(group_data).mean())**2)
                            eta_squared = ss_between / ss_total if ss_total > 0 else 0
                            comp_results['effect_size'] = eta_squared
                            comp_results['effect_interpretation'] = self.interpret_eta_squared(eta_squared)
                        else:
                            # Kruskal-Wallis (no paramétrico)
                            h_stat, h_p = kruskal(*group_data)
                            comp_results['test_used'] = 'Kruskal-Wallis'
                            comp_results['statistic'] = h_stat
                            comp_results['p_value'] = h_p
                            comp_results['significant'] = h_p < self.ALPHA

                            # Epsilon cuadrado (tamaño del efecto para Kruskal-Wallis)
                            n_total = sum(len(group) for group in group_data)
                            epsilon_squared = (h_stat - len(group_data) + 1) / (n_total - len(group_data))
                            comp_results['effect_size'] = max(0, epsilon_squared)  # No puede ser negativo
                            comp_results['effect_interpretation'] = self.interpret_epsilon_squared(comp_results['effect_size'])

                except Exception as e:
                    self.logger.warning(f"Error en comparación de grupos para {numeric_var} por {group_var}: {str(e)}")
                    comp_results['error'] = str(e)

                group_results[numeric_var] = comp_results

            comparison_results[group_var] = group_results

        self.results['group_comparisons'] = comparison_results
        return comparison_results

    def create_visualizations(self, df: pd.DataFrame, output_dir: str):
        """Crear visualizaciones de los resultados."""
        self.logger.info("Creando visualizaciones...")

        # Crear directorio principal si no existe
        os.makedirs(output_dir, exist_ok=True)

        # Crear subfolders para organizar los análisis
        normality_dir = os.path.join(output_dir, "01_normality_analysis")
        variance_dir = os.path.join(output_dir, "02_variance_homogeneity")
        comparisons_dir = os.path.join(output_dir, "03_group_comparisons")

        os.makedirs(normality_dir, exist_ok=True)
        os.makedirs(variance_dir, exist_ok=True)
        os.makedirs(comparisons_dir, exist_ok=True)

        # 1. Histogramas y Q-Q plots para variables principales
        self._create_normality_plots(df, normality_dir)

        # 2. Gráficos de comparación entre grupos
        self._create_group_comparison_plots(df, comparisons_dir)

        # 3. Resumen de pruebas de normalidad
        self._create_normality_summary_plot(normality_dir)

        # 4. Resumen de homogeneidad de varianzas
        self._create_variance_summary_plot(variance_dir)

        # 5. Resumen de comparaciones significativas
        self._create_comparison_summary_plot(comparisons_dir)

    def _create_normality_plots(self, df: pd.DataFrame, output_dir: str):
        """Crear histogramas y Q-Q plots para las variables más importantes."""
        # Seleccionar las 12 variables más importantes (incluyendo nota_final)
        important_vars = [
            'nota_final', 'edad_estudiante', 'student_total_interactions', 
            'student_total_views', 'total_hours', 'intensidad',
            'percent_modules_viewed', 'interaction_to_view_ratio',
            'teacher_experiencia_nivel', 'modules_participated',
            'avg_days_since_last_update', 'total_hermanos'
        ]

        # Filtrar variables que existen en el dataset
        available_vars = [var for var in important_vars if var in df.columns][:12]

        # Crear subplots
        n_vars = len(available_vars)
        n_cols = 3
        n_rows = (n_vars + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, var in enumerate(available_vars):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            # Filtrar datos válidos
            data = df[var].dropna()

            if len(data) > 0:
                # Histograma con curva normal superpuesta
                ax.hist(data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')

                # Curva normal teórica
                mu, sigma = data.mean(), data.std()
                x = np.linspace(data.min(), data.max(), 100)
                normal_curve = stats.norm.pdf(x, mu, sigma)
                ax.plot(x, normal_curve, 'r-', linewidth=2, label='Normal teórica')

                ax.set_title(f'{var}\n(n={len(data)})')
                ax.set_xlabel('Valor')
                ax.set_ylabel('Densidad')
                ax.legend()
                ax.grid(True, alpha=0.3)

        # Ocultar subplots vacíos
        for i in range(n_vars, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/histograms_normality.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Crear Q-Q plots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, var in enumerate(available_vars):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            data = df[var].dropna()

            if len(data) > 0:
                stats.probplot(data, dist="norm", plot=ax)
                ax.set_title(f'Q-Q Plot: {var}')
                ax.grid(True, alpha=0.3)

        # Ocultar subplots vacíos
        for i in range(n_vars, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/qq_plots_normality.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_group_comparison_plots(self, df: pd.DataFrame, output_dir: str):
        """Crear gráficos de comparación entre grupos."""
        for group_var in self.group_variables:
            if group_var not in df.columns:
                continue

            # Seleccionar variables importantes para visualizar
            important_vars = ['nota_final', 'student_total_interactions', 'student_total_views', 
                            'total_hours', 'percent_modules_viewed', 'edad_estudiante']
            available_vars = [var for var in important_vars if var in df.columns][:6]

            n_vars = len(available_vars)
            n_cols = 2
            n_rows = (n_vars + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)

            for i, var in enumerate(available_vars):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col]

                # Crear boxplot
                df_plot = df[[group_var, var]].dropna()
                if len(df_plot) > 0:
                    df_plot.boxplot(column=var, by=group_var, ax=ax)
                    ax.set_title(f'{var} por {group_var}')
                    ax.set_xlabel(group_var)
                    ax.set_ylabel(var)

            # Ocultar subplots vacíos
            for i in range(n_vars, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                axes[row, col].set_visible(False)

            plt.tight_layout()
            plt.savefig(f"{output_dir}/group_comparisons_{group_var}.png", dpi=300, bbox_inches='tight')
            plt.close()

    def _create_normality_summary_plot(self, output_dir: str):
        """Crear resumen visual de pruebas de normalidad."""
        normality_data = []

        for var, results in self.results['normality_tests'].items():
            normality_data.append({
                'Variable': var,
                'Shapiro-Wilk': results.get('shapiro_normal', None),
                'Kolmogorov-Smirnov': results.get('ks_normal', False),
                'Anderson-Darling': results.get('anderson_normal', False),
                'D\'Agostino-Pearson': results.get('dagostino_normal', False),
                'Jarque-Bera': results.get('jarque_bera_normal', False),
                'Consenso': results.get('overall_normal', False)
            })

        df_norm = pd.DataFrame(normality_data)

        if len(df_norm) > 0:
            # Crear heatmap
            plt.figure(figsize=(12, max(8, len(df_norm) * 0.3)))

            # Preparar datos para heatmap (convertir booleanos a números)
            heatmap_data = df_norm.set_index('Variable').copy()
            for col in heatmap_data.columns:
                heatmap_data[col] = heatmap_data[col].map({True: 1, False: 0, None: -1})

            # Crear heatmap
            sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', center=0, 
                       cbar_kws={'label': 'Normal (1) / No Normal (0) / No Aplicable (-1)'},
                       fmt='.0f')

            plt.title('Resumen de Pruebas de Normalidad por Variable')
            plt.xlabel('Pruebas de Normalidad')
            plt.ylabel('Variables')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/normality_summary_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()

    def _create_variance_summary_plot(self, output_dir: str):
        """Crear resumen visual de homogeneidad de varianzas."""
        for group_var, group_results in self.results['variance_homogeneity_tests'].items():
            variance_data = []

            for var, results in group_results.items():
                variance_data.append({
                    'Variable': var,
                    'Levene': results.get('levene_homogeneous', False),
                    'Bartlett': results.get('bartlett_homogeneous', False),
                    'Ratio_Varianzas': results.get('variance_ratio', 0)
                })

            if variance_data:
                df_var = pd.DataFrame(variance_data)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, max(6, len(df_var) * 0.3)))

                # Heatmap de homogeneidad
                heatmap_data = df_var[['Variable', 'Levene', 'Bartlett']].set_index('Variable')
                heatmap_data = heatmap_data.astype(float).fillna(-1)  # Llenar NaN con -1

                sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', ax=ax1,
                           cbar_kws={'label': 'Homogéneo (1) / No Homogéneo (0)'}, fmt='.0f')
                ax1.set_title(f'Homogeneidad de Varianzas por {group_var}')

                # Gráfico de ratios de varianza
                ax2.barh(df_var['Variable'], df_var['Ratio_Varianzas'])
                ax2.axvline(x=4, color='red', linestyle='--', label='Ratio crítico (4:1)')
                ax2.set_xlabel('Ratio Varianza Máx/Mín')
                ax2.set_title(f'Ratios de Varianza por {group_var}')
                ax2.legend()

                plt.tight_layout()
                plt.savefig(f"{output_dir}/variance_homogeneity_{group_var}.png", dpi=300, bbox_inches='tight')
                plt.close()

    def _create_comparison_summary_plot(self, output_dir: str):
        """Crear resumen de comparaciones significativas."""
        for group_var, group_results in self.results['group_comparisons'].items():
            comparison_data = []

            for var, results in group_results.items():
                comparison_data.append({
                    'Variable': var,
                    'Test': results.get('test_used', 'N/A'),
                    'P_Value': results.get('p_value', 1.0),
                    'Significativo': results.get('significant', False),
                    'Tamaño_Efecto': results.get('effect_size', 0),
                    'Interpretación': results.get('effect_interpretation', 'N/A')
                })

            if comparison_data:
                df_comp = pd.DataFrame(comparison_data)

                # Ordenar por p-value
                df_comp = df_comp.sort_values('P_Value')

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(df_comp) * 0.3)))

                # Gráfico de p-values
                colors = ['red' if sig else 'blue' for sig in df_comp['Significativo']]

                # Manejar p-valores extremadamente pequeños para evitar infinitos
                log_p_values = []
                for p in df_comp['P_Value']:
                    if p == 0 or p < 1e-300:  # P-valor extremadamente pequeño
                        log_p_values.append(300)  # Límite máximo para visualización
                    else:
                        log_p_values.append(-np.log10(p))

                ax1.barh(df_comp['Variable'], log_p_values, color=colors, alpha=0.7)
                ax1.axvline(-np.log10(self.ALPHA), color='red', linestyle='--', 
                           label=f'Umbral significancia (α={self.ALPHA})')
                ax1.set_xlabel('-log10(p-value)')
                ax1.set_title(f'Significancia de Comparaciones por {group_var}')
                ax1.legend()

                # Agregar nota sobre valores extremos
                max_log_p = max(log_p_values)
                if max_log_p >= 300:
                    ax1.text(0.02, 0.98, 'Nota: Valores ≥300 representan\np-valores extremadamente pequeños', 
                            transform=ax1.transAxes, fontsize=8, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

                # Gráfico de tamaños de efecto
                ax2.barh(df_comp['Variable'], df_comp['Tamaño_Efecto'], color=colors, alpha=0.7)
                ax2.set_xlabel('Tamaño del Efecto')
                ax2.set_title(f'Tamaños de Efecto por {group_var}')

                plt.tight_layout()
                plt.savefig(f"{output_dir}/comparison_summary_{group_var}.png", dpi=300, bbox_inches='tight')
                plt.close()

    def save_results(self, output_dir: str):
        """Guardar resultados en archivos CSV."""
        self.logger.info("Guardando resultados...")

        # Crear directorio principal si no existe
        os.makedirs(output_dir, exist_ok=True)

        # Crear subfolders para organizar los resultados CSV
        normality_dir = os.path.join(output_dir, "01_normality_analysis")
        variance_dir = os.path.join(output_dir, "02_variance_homogeneity")
        comparisons_dir = os.path.join(output_dir, "03_group_comparisons")
        summary_dir = os.path.join(output_dir, "00_summary")

        os.makedirs(normality_dir, exist_ok=True)
        os.makedirs(variance_dir, exist_ok=True)
        os.makedirs(comparisons_dir, exist_ok=True)
        os.makedirs(summary_dir, exist_ok=True)

        # 1. Guardar resultados de normalidad
        if self.results['normality_tests']:
            normality_df = pd.DataFrame([
                {
                    'variable': var,
                    'n_observations': results.get('n_observations', 0),
                    'mean': results.get('mean', 0),
                    'std': results.get('std', 0),
                    'skewness': results.get('skewness', 0),
                    'kurtosis': results.get('kurtosis', 0),
                    'shapiro_statistic': results.get('shapiro_statistic', np.nan),
                    'shapiro_p_value': results.get('shapiro_p_value', np.nan),
                    'shapiro_normal': results.get('shapiro_normal', None),
                    'ks_statistic': results.get('ks_statistic', np.nan),
                    'ks_p_value': results.get('ks_p_value', np.nan),
                    'ks_normal': results.get('ks_normal', False),
                    'anderson_statistic': results.get('anderson_statistic', np.nan),
                    'anderson_critical_5pct': results.get('anderson_critical_5pct', np.nan),
                    'anderson_normal': results.get('anderson_normal', False),
                    'dagostino_statistic': results.get('dagostino_statistic', np.nan),
                    'dagostino_p_value': results.get('dagostino_p_value', np.nan),
                    'dagostino_normal': results.get('dagostino_normal', False),
                    'jarque_bera_statistic': results.get('jarque_bera_statistic', np.nan),
                    'jarque_bera_p_value': results.get('jarque_bera_p_value', np.nan),
                    'jarque_bera_normal': results.get('jarque_bera_normal', False),
                    'overall_normal': results.get('overall_normal', False),
                    'normal_test_count': results.get('normal_test_count', 0),
                    'total_tests': results.get('total_tests', 0)
                }
                for var, results in self.results['normality_tests'].items()
            ])
            self.save_to_csv(normality_df, f"{normality_dir}/normality_tests_results.csv")

        # 2. Guardar resultados de homogeneidad de varianzas
        variance_rows = []
        for group_var, group_results in self.results['variance_homogeneity_tests'].items():
            for var, results in group_results.items():
                variance_rows.append({
                    'group_variable': group_var,
                    'numeric_variable': var,
                    'n_groups': results.get('n_groups', 0),
                    'levene_statistic': results.get('levene_statistic', np.nan),
                    'levene_p_value': results.get('levene_p_value', np.nan),
                    'levene_homogeneous': results.get('levene_homogeneous', False),
                    'bartlett_statistic': results.get('bartlett_statistic', np.nan),
                    'bartlett_p_value': results.get('bartlett_p_value', np.nan),
                    'bartlett_homogeneous': results.get('bartlett_homogeneous', False),
                    'max_variance': results.get('max_var', np.nan),
                    'min_variance': results.get('min_var', np.nan),
                    'variance_ratio': results.get('variance_ratio', np.nan)
                })

        if variance_rows:
            variance_df = pd.DataFrame(variance_rows)
            self.save_to_csv(variance_df, f"{variance_dir}/variance_homogeneity_results.csv")

        # 3. Guardar resultados de comparaciones entre grupos
        comparison_rows = []
        for group_var, group_results in self.results['group_comparisons'].items():
            for var, results in group_results.items():
                comparison_rows.append({
                    'group_variable': group_var,
                    'numeric_variable': var,
                    'n_groups': results.get('n_groups', 0),
                    'test_used': results.get('test_used', 'N/A'),
                    'statistic': results.get('statistic', np.nan),
                    'p_value': results.get('p_value', np.nan),
                    'significant': results.get('significant', False),
                    'effect_size': results.get('effect_size', np.nan),
                    'effect_interpretation': results.get('effect_interpretation', 'N/A'),
                    'assumes_normality': results.get('assumes_normality', False),
                    'assumes_homogeneity': results.get('assumes_homogeneity', False),
                    'distributions_different': results.get('distributions_different', None),
                    'ks_statistic': results.get('ks_statistic', np.nan),
                    'ks_p_value': results.get('ks_p_value', np.nan)
                })

        if comparison_rows:
            comparison_df = pd.DataFrame(comparison_rows)
            self.save_to_csv(comparison_df, f"{comparisons_dir}/group_comparisons_results.csv")

        # 4. Crear resumen ejecutivo
        summary_data = {
            'total_variables_analyzed': len(self.results['normality_tests']),
            'normal_variables': sum(1 for r in self.results['normality_tests'].values() 
                                  if r.get('overall_normal', False)),
            'group_variables': len(self.group_variables),
            'total_comparisons': sum(len(group_results) for group_results in self.results['group_comparisons'].values()),
            'significant_comparisons': sum(
                sum(1 for r in group_results.values() if r.get('significant', False))
                for group_results in self.results['group_comparisons'].values()
            )
        }

        summary_df = pd.DataFrame([summary_data])
        self.save_to_csv(summary_df, f"{summary_dir}/analysis_summary.csv")

        # 5. Crear reporte de resumen en texto
        self._create_summary_report(summary_dir, summary_data)

    def _create_summary_report(self, output_dir: str, summary_data: dict):
        """Crear reporte de resumen en formato texto."""
        report_path = os.path.join(output_dir, "homogeneity_analysis_report.txt")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("REPORTE DE ANÁLISIS DE HOMOGENEIDAD DE GRUPOS\n")
            f.write("=" * 80 + "\n\n")

            # Información general
            f.write("RESUMEN EJECUTIVO\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total de variables numéricas analizadas: {summary_data['total_variables_analyzed']}\n")
            f.write(f"Variables con distribución normal: {summary_data['normal_variables']}\n")
            f.write(f"Variables de agrupación: {summary_data['group_variables']}\n")
            f.write(f"Total de comparaciones realizadas: {summary_data['total_comparisons']}\n")
            f.write(f"Comparaciones estadísticamente significativas: {summary_data['significant_comparisons']}\n\n")

            # Porcentajes
            if summary_data['total_variables_analyzed'] > 0:
                normal_pct = (summary_data['normal_variables'] / summary_data['total_variables_analyzed']) * 100
                f.write(f"Porcentaje de variables normales: {normal_pct:.1f}%\n")

            if summary_data['total_comparisons'] > 0:
                significant_pct = (summary_data['significant_comparisons'] / summary_data['total_comparisons']) * 100
                f.write(f"Porcentaje de comparaciones significativas: {significant_pct:.1f}%\n\n")

            # Detalles de normalidad
            f.write("ANÁLISIS DE NORMALIDAD\n")
            f.write("-" * 40 + "\n")

            normal_vars = []
            non_normal_vars = []

            for var, results in self.results['normality_tests'].items():
                if results.get('overall_normal', False):
                    normal_vars.append(var)
                else:
                    non_normal_vars.append(var)

            f.write(f"Variables con distribución normal ({len(normal_vars)}):\n")
            for var in normal_vars:
                f.write(f"  - {var}\n")

            f.write(f"\nVariables sin distribución normal ({len(non_normal_vars)}):\n")
            for var in non_normal_vars:
                f.write(f"  - {var}\n")

            # Detalles de comparaciones significativas
            f.write("\n\nCOMPARACIONES SIGNIFICATIVAS\n")
            f.write("-" * 40 + "\n")

            for group_var, group_results in self.results['group_comparisons'].items():
                significant_comparisons = [(var, results) for var, results in group_results.items() 
                                         if results.get('significant', False)]

                if significant_comparisons:
                    f.write(f"\nPor variable de agrupación '{group_var}':\n")
                    for var, results in significant_comparisons:
                        test_used = results.get('test_used', 'N/A')
                        p_value = results.get('p_value', 1.0)
                        effect_size = results.get('effect_size', 0)
                        effect_interp = results.get('effect_interpretation', 'N/A')

                        f.write(f"  - {var}:\n")
                        f.write(f"    * Test utilizado: {test_used}\n")
                        f.write(f"    * p-valor: {p_value:.6f}\n")
                        f.write(f"    * Tamaño del efecto: {effect_size:.4f} ({effect_interp})\n")

            # Recomendaciones
            f.write("\n\nRECOMENDACIONES\n")
            f.write("-" * 40 + "\n")

            if summary_data['normal_variables'] < summary_data['total_variables_analyzed'] / 2:
                f.write("• La mayoría de variables no siguen distribución normal.\n")
                f.write("  Considere usar tests no paramétricos para análisis futuros.\n")

            if summary_data['significant_comparisons'] > 0:
                f.write("• Se encontraron diferencias significativas entre grupos.\n")
                f.write("  Revise los resultados detallados para interpretación específica.\n")

            f.write("\n• Consulte las visualizaciones en los subfolders correspondientes.\n")
            f.write("• Los archivos CSV contienen resultados detallados de cada análisis.\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("Fin del reporte\n")
            f.write("=" * 80 + "\n")

    def run_analysis(self):
        """Ejecutar análisis completo de homogeneidad."""
        self.logger.info("Iniciando análisis de homogeneidad de grupos...")
        self.logger.info(f"Dataset: {self.dataset_path}")
        self.logger.info(f"Resultados se guardarán en: {self.results_path}")

        # Validar que el dataset exista
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"No se encontró el dataset en: {self.dataset_path}")

        # Crear directorio de resultados si no existe
        self.create_results_directory()

        # Cargar datos
        df = self.load_data(self.dataset_path)

        # 1. Pruebas de normalidad
        self.test_normality(df)

        # 2. Pruebas de homogeneidad de varianzas
        self.test_variance_homogeneity(df)

        # 3. Comparaciones entre grupos
        self.compare_groups(df)

        # 4. Crear visualizaciones
        self.create_visualizations(df, self.results_path)

        # 5. Guardar resultados
        self.save_results(self.results_path)

        # Resumen final
        self.logger.info("=== RESUMEN DEL ANÁLISIS DE HOMOGENEIDAD ===")
        self.logger.info(f"Variables numéricas analizadas: {len(self.results['normality_tests'])}")

        normal_vars = sum(1 for r in self.results['normality_tests'].values() 
                         if r.get('overall_normal', False))
        self.logger.info(f"Variables con distribución normal: {normal_vars}")

        total_comparisons = sum(len(group_results) for group_results in self.results['group_comparisons'].values())
        significant_comparisons = sum(
            sum(1 for r in group_results.values() if r.get('significant', False))
            for group_results in self.results['group_comparisons'].values()
        )

        self.logger.info(f"Total de comparaciones realizadas: {total_comparisons}")
        self.logger.info(f"Comparaciones significativas: {significant_comparisons}")

        self.logger.info("\n=== ESTRUCTURA DE RESULTADOS ===")
        self.logger.info("Los resultados se han organizado en los siguientes subfolders:")
        self.logger.info("  📁 00_summary/ - Resumen ejecutivo y reporte general")
        self.logger.info("  📁 01_normality_analysis/ - Pruebas de normalidad y visualizaciones")
        self.logger.info("  📁 02_variance_homogeneity/ - Análisis de homogeneidad de varianzas")
        self.logger.info("  📁 03_group_comparisons/ - Comparaciones entre grupos y significancia")

        self.logger.info("Análisis de homogeneidad completado exitosamente!")

def main():
    """Función principal."""
    import argparse

    parser = argparse.ArgumentParser(description='Análisis de homogeneidad de grupos')
    parser.add_argument('--dataset', '-d', type=str, required=True,
                       help='Ruta al archivo CSV del dataset (requerido)')
    parser.add_argument('--results', '-r', type=str, required=True,
                       help='Nombre del folder para guardar resultados (requerido, se creará en reports/)')

    args = parser.parse_args()

    # Crear y ejecutar analizador
    analyzer = HomogeneityAnalysis(args.dataset, args.results)

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
