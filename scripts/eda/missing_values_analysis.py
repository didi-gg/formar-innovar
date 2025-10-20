"""
Script para análisis de valores faltantes en el dataset.
Genera mapas de calor y gráficas de porcentajes de valores faltantes por feature.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

# Configuración de warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para suprimir mensajes de debug
import matplotlib
matplotlib.set_loglevel("WARNING")

# Agregar el directorio padre al path para importar utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.eda_analysis_base import EDAAnalysisBase


class MissingValuesAnalyzer(EDAAnalysisBase):
    """
    Analizador de valores faltantes que genera:
    1. Gráfica de barras con porcentajes de valores faltantes
    2. Tabla resumen clara de valores faltantes
    3. Gráfica resumen con estadísticas generales
    4. Estadísticas detalladas de valores faltantes
    5. Análisis de patrones de valores faltantes
    """

    def _initialize_analysis_attributes(self):
        """Inicializar atributos específicos del análisis de valores faltantes."""
        self.missing_stats = {}
        self.missing_patterns = {}
        self.results = {}

    def identify_missing_values(self, df):
        """
        Identifica valores faltantes incluyendo NaN, valores vacíos y 0.

        Args:
            df (pd.DataFrame): DataFrame a analizar

        Returns:
            pd.DataFrame: DataFrame booleano donde True indica valor faltante
        """
        missing_mask = df.copy()

        for col in df.columns:
            if df[col].dtype == 'object':  # Columnas de texto
                # Para texto: NaN, cadenas vacías, espacios, 'nan', 'none', '0'
                missing_mask[col] = (
                    df[col].isnull() | 
                    (df[col] == '') | 
                    (df[col].astype(str).str.strip() == '') |
                    (df[col].astype(str).str.lower().isin(['nan', 'none', 'null'])) |
                    (df[col] == '0')  # Incluir '0' como valor faltante
                )
            else:  # Columnas numéricas
                # Para variables numéricas: NaN y 0 son faltantes
                missing_mask[col] = df[col].isnull() | (df[col] == 0)

        return missing_mask

    def calculate_missing_statistics(self, df):
        """
        Calcula estadísticas detalladas de valores faltantes.

        Args:
            df (pd.DataFrame): DataFrame a analizar

        Returns:
            dict: Estadísticas de valores faltantes
        """
        try:
            self.logger.info("Calculando estadísticas de valores faltantes...")
            self.logger.info("Considerando como faltantes: NaN y 0 para variables numéricas, cadenas vacías, espacios y '0' para variables de texto")

            # Obtener features válidas (excluyendo las columnas base excluidas)
            valid_features = self.get_valid_features(df, exclude_targets=False)
            df_valid = df[valid_features]

            # Identificar valores faltantes (incluyendo vacíos y 0)
            missing_mask = self.identify_missing_values(df_valid)

            # Estadísticas básicas (solo para features válidas)
            total_cells = df_valid.shape[0] * df_valid.shape[1]
            missing_cells = missing_mask.sum().sum()
            missing_percentage = (missing_cells / total_cells) * 100

            # Estadísticas por columna (solo features válidas)
            missing_by_column = missing_mask.sum()
            missing_percentage_by_column = (missing_by_column / len(df_valid)) * 100

            # Crear DataFrame con estadísticas
            missing_df = pd.DataFrame({
                'feature': missing_by_column.index,
                'missing_count': missing_by_column.values,
                'missing_percentage': missing_percentage_by_column.values
            }).sort_values('missing_percentage', ascending=False)

            # Filtrar solo columnas con valores faltantes
            missing_df_filtered = missing_df[missing_df['missing_count'] > 0]

            # Estadísticas por fila (solo para features válidas)
            missing_by_row = missing_mask.sum(axis=1)
            rows_with_missing = (missing_by_row > 0).sum()
            rows_with_missing_percentage = (rows_with_missing / len(df_valid)) * 100

            # Patrones de valores faltantes
            complete_rows = len(df_valid) - rows_with_missing
            complete_columns = len(df_valid.columns) - len(missing_df_filtered)

            stats = {
                'total_cells': total_cells,
                'missing_cells': missing_cells,
                'missing_percentage_total': missing_percentage,
                'missing_by_column': missing_df,
                'missing_by_column_filtered': missing_df_filtered,
                'rows_with_missing': rows_with_missing,
                'rows_with_missing_percentage': rows_with_missing_percentage,
                'complete_rows': complete_rows,
                'complete_columns': complete_columns,
                'total_features': len(df_valid.columns),
                'features_with_missing': len(missing_df_filtered),
                'valid_features_count': len(valid_features),
                'excluded_features_count': len(df.columns) - len(valid_features)
            }

            self.missing_stats = stats
            self.logger.info(f"Estadísticas calculadas: {missing_cells:,} valores faltantes ({missing_percentage:.2f}%)")
            self.logger.info(f"Features con valores faltantes: {len(missing_df_filtered)}/{len(df.columns)}")

            return stats

        except Exception as e:
            self.logger.error(f"Error calculando estadísticas de valores faltantes: {e}")
            raise

    def create_missing_percentage_chart(self, df, min_percentage=0.0):
        """
        Crea gráfica de barras con porcentajes de valores faltantes.

        Args:
            df (pd.DataFrame): DataFrame a analizar
            min_percentage (float): Porcentaje mínimo para mostrar en la gráfica
        """
        try:
            self.logger.info("Creando gráfica de porcentajes de valores faltantes...")

            # Obtener features válidas (excluyendo las columnas base excluidas)
            valid_features = self.get_valid_features(df, exclude_targets=False)
            df_valid = df[valid_features]

            # Identificar valores faltantes (incluyendo vacíos y 0)
            missing_mask = self.identify_missing_values(df_valid)

            # Calcular porcentajes de valores faltantes solo para features válidas
            missing_percentages = (missing_mask.sum() / len(df_valid)) * 100

            # Mostrar TODAS las features con valores faltantes (sin filtro de porcentaje mínimo)
            missing_filtered = missing_percentages[missing_percentages > 0].sort_values(ascending=True)

            if missing_filtered.empty:
                self.logger.info("No hay features válidas con valores faltantes")
                return

            # Crear figura adaptativa al número de variables
            fig_height = max(10, min(len(missing_filtered) * 0.5, 30))  # Entre 10 y 30
            plt.figure(figsize=(14, fig_height))

            # Crear gráfica de barras horizontal
            bars = plt.barh(range(len(missing_filtered)), missing_filtered.values, 
                           color='#d62728', alpha=0.7, edgecolor='black', linewidth=0.5)

            # Personalizar gráfica - ajustar tamaño de fuente según número de variables
            fontsize_y = max(6, min(10, 150 // len(missing_filtered)))
            plt.yticks(range(len(missing_filtered)), missing_filtered.index, fontsize=fontsize_y)
            plt.xlabel('Porcentaje de Valores Faltantes (%)', fontsize=12, fontweight='bold')
            plt.ylabel('Features', fontsize=12, fontweight='bold')
            plt.title(f'Porcentaje de Valores Faltantes por Feature\n(Todas las {len(missing_filtered)} features con valores faltantes)\nIncluye: NaN y 0 para numéricas, cadenas vacías, espacios y "0" para texto', 
                     fontsize=13, fontweight='bold', pad=20)

            # Agregar valores en las barras
            for i, (bar, value) in enumerate(zip(bars, missing_filtered.values)):
                plt.text(value + 0.5, i, f'{value:.1f}%', 
                        va='center', ha='left', fontsize=9, fontweight='bold')

            # Agregar líneas de referencia
            plt.axvline(x=10, color='orange', linestyle='--', alpha=0.7, label='10%')
            plt.axvline(x=25, color='red', linestyle='--', alpha=0.7, label='25%')
            plt.axvline(x=50, color='darkred', linestyle='--', alpha=0.7, label='50%')

            # Configurar eje X
            plt.xlim(0, max(100, missing_filtered.max() * 1.1))
            plt.grid(axis='x', alpha=0.3)
            plt.legend(loc='lower right')

            # Ajustar layout
            plt.tight_layout()

            # Guardar
            output_path = os.path.join(self.results_path, 'missing_values_percentage.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Gráfica de porcentajes guardada: {output_path}")

        except Exception as e:
            self.logger.error(f"Error creando gráfica de porcentajes: {e}")
            raise

    def analyze_missing_patterns(self, df):
        """
        Analiza patrones de valores faltantes entre features.

        Args:
            df (pd.DataFrame): DataFrame a analizar

        Returns:
            dict: Análisis de patrones
        """
        try:
            self.logger.info("Analizando patrones de valores faltantes...")

            # Obtener features válidas con valores faltantes
            valid_features = self.get_valid_features(df, exclude_targets=False)
            df_valid = df[valid_features]

            # Identificar valores faltantes (incluyendo vacíos y 0)
            missing_mask = self.identify_missing_values(df_valid)
            missing_cols = [col for col in valid_features if missing_mask[col].any()]

            if not missing_cols:
                return {'message': 'No hay valores faltantes para analizar patrones'}

            patterns = {}

            # 1. Correlación entre valores faltantes
            if len(missing_cols) > 1:
                missing_corr = missing_mask[missing_cols].astype(int).corr()

                # Encontrar correlaciones altas (>0.5)
                high_corr_pairs = []
                for i in range(len(missing_corr.columns)):
                    for j in range(i+1, len(missing_corr.columns)):
                        corr_value = missing_corr.iloc[i, j]
                        if abs(corr_value) > 0.5:
                            high_corr_pairs.append({
                                'feature1': missing_corr.columns[i],
                                'feature2': missing_corr.columns[j],
                                'correlation': corr_value
                            })

                patterns['missing_correlations'] = {
                    'correlation_matrix': missing_corr,
                    'high_correlation_pairs': high_corr_pairs
                }

            # 2. Combinaciones más comunes de valores faltantes
            if len(missing_cols) <= 10:  # Solo para un número manejable de features
                missing_combinations = missing_mask[missing_cols].value_counts().head(10)
                patterns['common_combinations'] = missing_combinations

            # 3. Features que siempre faltan juntas
            if len(missing_cols) > 1:
                always_missing_together = []
                for i, col1 in enumerate(missing_cols):
                    for col2 in missing_cols[i+1:]:
                        # Verificar si cuando falta col1, siempre falta col2
                        mask1_missing = missing_mask[col1]
                        mask2_missing = missing_mask[col2]

                        if mask1_missing.any():
                            together_rate = (mask1_missing & mask2_missing).sum() / mask1_missing.sum()
                            if together_rate > 0.9:  # 90% de las veces faltan juntas
                                always_missing_together.append({
                                    'feature1': col1,
                                    'feature2': col2,
                                    'together_rate': together_rate
                                })

                patterns['always_missing_together'] = always_missing_together

            self.missing_patterns = patterns
            self.logger.info(f"Análisis de patrones completado")

            return patterns

        except Exception as e:
            self.logger.error(f"Error analizando patrones: {e}")
            raise

    def analyze_feature_diversity(self, df):
        """
        Analiza la diversidad de valores en las features válidas.
        Para variables numéricas: calcula varianza y coeficiente de variación
        Para variables categóricas: cuenta valores únicos y detecta features con un solo valor

        Args:
            df (pd.DataFrame): DataFrame a analizar

        Returns:
            dict: Análisis de diversidad
        """
        try:
            self.logger.info("Analizando diversidad de valores en features válidas...")

            # Obtener features válidas
            valid_features = self.get_valid_features(df, exclude_targets=False)
            df_valid = df[valid_features]

            diversity_stats = {
                'numeric_features': [],
                'categorical_features': [],
                'low_variance_numeric': [],
                'single_value_categorical': [],
                'low_diversity_categorical': []
            }

            # Usar la lógica de clasificación de la clase base
            variable_types = self.identify_variable_types(df_valid, valid_features)
            categorical_features = variable_types['categorical']
            continuous_features = variable_types['continuous']

            for col in valid_features:
                # Obtener datos no nulos para el análisis
                non_null_data = df_valid[col].dropna()

                if len(non_null_data) == 0:
                    continue

                if col in categorical_features:  # Variables categóricas
                    unique_values = non_null_data.unique()
                    n_unique = len(unique_values)
                    n_total = len(non_null_data)
                    diversity_ratio = n_unique / n_total if n_total > 0 else 0

                    # Calcular distribución de valores
                    value_counts = non_null_data.value_counts()
                    most_common_pct = (value_counts.iloc[0] / n_total * 100) if len(value_counts) > 0 else 0

                    feature_info = {
                        'feature': col,
                        'unique_values': n_unique,
                        'total_values': n_total,
                        'diversity_ratio': diversity_ratio,
                        'most_common_percentage': most_common_pct,
                        'sample_values': list(unique_values[:5])  # Primeros 5 valores únicos
                    }

                    diversity_stats['categorical_features'].append(feature_info)

                    # Detectar features problemáticas
                    if n_unique == 1:
                        diversity_stats['single_value_categorical'].append(feature_info)
                    elif n_unique <= 2 or diversity_ratio < 0.01:  # Muy poca diversidad
                        diversity_stats['low_diversity_categorical'].append(feature_info)

                elif col in continuous_features:  # Variables numéricas
                    # Convertir a numérico si es posible
                    try:
                        numeric_data = pd.to_numeric(non_null_data, errors='coerce').dropna()

                        if len(numeric_data) == 0:
                            continue

                        # Calcular estadísticas de variabilidad
                        mean_val = numeric_data.mean()
                        std_val = numeric_data.std()
                        var_val = numeric_data.var()

                        # Coeficiente de variación corregido - usar valor absoluto de la media
                        if abs(mean_val) > 1e-10:  # Evitar división por cero
                            cv = (std_val / abs(mean_val) * 100)
                        else:
                            cv = 0


                        # Contar valores únicos
                        unique_values = len(numeric_data.unique())

                        feature_info = {
                            'feature': col,
                            'mean': mean_val,
                            'std': std_val,
                            'variance': var_val,
                            'cv': cv,  # Coeficiente de variación
                            'unique_values': unique_values,
                            'total_values': len(numeric_data),
                            'min_value': numeric_data.min(),
                            'max_value': numeric_data.max(),
                            'range': numeric_data.max() - numeric_data.min()
                        }

                        diversity_stats['numeric_features'].append(feature_info)

                        # Detectar features con baja varianza usando múltiples criterios
                        is_low_variance = False
                        data_range = numeric_data.max() - numeric_data.min()

                        # Criterio 1: Muy pocos valores únicos (≤ 2)
                        if unique_values <= 2:
                            is_low_variance = True

                        # Criterio 2: Pocos valores únicos Y rango pequeño relativo a la media
                        elif unique_values <= 5 and data_range > 0:
                            # Para variables como años, el rango relativo es más importante que el CV
                            relative_range = data_range / abs(mean_val) if abs(mean_val) > 0 else 0
                            if relative_range < 0.01:  # Rango < 1% de la media
                                is_low_variance = True

                        # Criterio 3: CV muy bajo Y pocos valores únicos
                        elif cv < 1 and unique_values <= 10:
                            is_low_variance = True

                        # Criterio 4: Varianza muy pequeña relativa al rango
                        elif data_range > 0:
                            normalized_var = var_val / (data_range ** 2)
                            if normalized_var < 0.001:  # Varianza muy muy pequeña
                                is_low_variance = True

                        # Excepción: No considerar como baja varianza si tiene suficientes valores únicos
                        # y el rango es razonable (ej: años, IDs, etc.)
                        if unique_values >= 4 and data_range >= 3:
                            # Para variables como años, si hay al menos 4 valores diferentes
                            # en un rango de al menos 3, no es baja varianza
                            if col in ['año_ingreso', 'año', 'year'] or 'año' in col.lower():
                                is_low_variance = False

                        if is_low_variance:
                            diversity_stats['low_variance_numeric'].append(feature_info)

                    except Exception as e:
                        self.logger.warning(f"No se pudo procesar feature numérica {col}: {e}")
                        continue

            self.logger.info(f"Análisis de diversidad completado: {len(diversity_stats['numeric_features'])} numéricas, {len(diversity_stats['categorical_features'])} categóricas")

            return diversity_stats

        except Exception as e:
            self.logger.error(f"Error analizando diversidad de features: {e}")
            raise

    def create_diversity_analysis_chart(self, df):
        """
        Crea gráficas simples para analizar la diversidad de valores en features válidas.

        Args:
            df (pd.DataFrame): DataFrame a analizar
        """
        try:
            self.logger.info("Creando gráficas simples de análisis de diversidad...")

            # Analizar diversidad
            diversity_stats = self.analyze_feature_diversity(df)

            # Crear figura más grande para acomodar nombres completos y top 20 features categóricas
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 12), facecolor='white')

            # 1. Features numéricas - Top 20 con menor varianza
            if diversity_stats['numeric_features']:
                numeric_df = pd.DataFrame(diversity_stats['numeric_features'])

                # Ordenar por CV de menor a mayor y tomar las top 20
                top_low_variance = numeric_df.sort_values('cv', ascending=True).head(20)

                if len(top_low_variance) > 0:
                    features_to_show = top_low_variance

                    y_pos = range(len(features_to_show))
                    ax1.barh(y_pos, features_to_show['cv'], color='red', alpha=0.7)

                    # Etiquetas completas
                    ax1.set_yticks(y_pos)
                    ax1.set_yticklabels(features_to_show['feature'], fontsize=9)

                    ax1.set_xlabel('CV (%)')
                    ax1.set_title(f'Top 20 Features Numéricas - Menor Varianza\n({len(features_to_show)} mostradas)', color='darkblue')

                    # Agregar valores
                    for i, value in enumerate(features_to_show['cv']):
                        ax1.text(value + 0.01, i, f'{value:.1f}%', va='center', fontsize=8)
                else:
                    ax1.text(0.5, 0.5, 'Sin features\nnuméricas', ha='center', va='center', 
                            transform=ax1.transAxes, fontsize=12)
                    ax1.set_title('Features Numéricas', color='gray')
            else:
                ax1.text(0.5, 0.5, 'Sin features\nnuméricas', ha='center', va='center', 
                        transform=ax1.transAxes, fontsize=12)
                ax1.set_title('Features Numéricas')

            # 2. Features categóricas - Top 20 con menor diversidad
            if diversity_stats['categorical_features']:
                cat_df = pd.DataFrame(diversity_stats['categorical_features'])

                # Ordenar por valores únicos de menor a mayor y tomar las top 20
                top_low_diversity = cat_df.sort_values('unique_values', ascending=True).head(20)

                if len(top_low_diversity) > 0:
                    features_to_show = top_low_diversity

                    y_pos = range(len(features_to_show))
                    ax2.barh(y_pos, features_to_show['unique_values'], color='orange', alpha=0.7)

                    # Etiquetas completas
                    ax2.set_yticks(y_pos)
                    ax2.set_yticklabels(features_to_show['feature'], fontsize=9)

                    ax2.set_xlabel('Valores Únicos')
                    ax2.set_title(f'Top 20 Features Categóricas - Menor Diversidad\n({len(features_to_show)} mostradas)', color='darkorange')

                    # Agregar valores
                    for i, value in enumerate(features_to_show['unique_values']):
                        ax2.text(value + 0.1, i, f'{int(value)}', va='center', fontsize=8)
                else:
                    ax2.text(0.5, 0.5, 'Sin features\ncategóricas', ha='center', va='center', 
                            transform=ax2.transAxes, fontsize=12)
                    ax2.set_title('Features Categóricas', color='gray')
            else:
                ax2.text(0.5, 0.5, 'Sin features\ncategóricas', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Features Categóricas')

            # Título general
            fig.suptitle('Análisis de Diversidad - Features Problemáticas', fontsize=14, fontweight='bold')

            # Layout simple
            plt.tight_layout()

            # Guardar con configuración mínima
            output_path = os.path.join(self.results_path, 'feature_diversity_analysis.png')
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Gráfica simple guardada: {output_path}")

            return diversity_stats

        except Exception as e:
            self.logger.error(f"Error creando gráficas de diversidad: {e}")
            raise

    def run_analysis(self):
        """
        Ejecuta el análisis completo de valores faltantes.

        Returns:
            dict: Resultados del análisis
        """
        try:
            self.logger.info("Iniciando análisis de valores faltantes...")

            # Crear directorio de resultados
            self.create_results_directory()

            # Cargar datos
            df = self.load_data()


            # Calcular estadísticas
            stats = self.calculate_missing_statistics(df)

            # Crear visualizaciones
            self.create_missing_percentage_chart(df)

            # Analizar diversidad de features válidas
            diversity_stats = self.create_diversity_analysis_chart(df)

            # Analizar patrones
            patterns = self.analyze_missing_patterns(df)


            # Preparar resultados
            results = {
                'statistics': stats,
                'patterns': patterns,
                'diversity_stats': diversity_stats,
                'dataset_shape': df.shape,
                'analysis_completed': True
            }

            self.results = results

            self.logger.info("✅ Análisis de valores faltantes completado exitosamente")
            self.logger.info(f"Archivos generados en: {self.results_path}")

            return results

        except Exception as e:
            self.logger.error(f"❌ Error en análisis de valores faltantes: {e}")
            raise


def main():
    """Función principal para ejecutar el análisis desde línea de comandos."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Análisis de valores faltantes en dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Analizar dataset completo
  python 00_missing_values_analysis.py --dataset full

  # Analizar dataset con Moodle
  python 00_missing_values_analysis.py --dataset moodle

  # Analizar dataset sin Moodle
  python 00_missing_values_analysis.py --dataset no_moodle

  # Usar archivo personalizado
  python 00_missing_values_analysis.py --file data/mi_dataset.csv --output mi_analisis
        """
    )

    parser.add_argument('--dataset', '-d', type=str,
                       choices=['moodle', 'no_moodle', 'full'],
                       help='Dataset predefinido a analizar')

    parser.add_argument('--file', '-f', type=str,
                       help='Ruta a archivo CSV personalizado')

    parser.add_argument('--output', '-o', type=str,
                       help='Carpeta de salida personalizada')

    args = parser.parse_args()

    # Configurar parámetros
    if args.dataset:
        datasets = {
            'moodle': {
                'path': "data/processed/full_short_dataset_moodle.csv",
                'folder': "missing_analysis_moodle"
            },
            'no_moodle': {
                'path': "data/processed/full_short_dataset_no_moodle.csv",
                'folder': "missing_analysis_no_moodle"
            },
            'full': {
                'path': "data/processed/full_short_dataset.csv",
                'folder': "missing_analysis_full"
            }
        }

        config = datasets[args.dataset]
        dataset_path = config['path']
        results_folder = config['folder']

    elif args.file:
        dataset_path = args.file
        results_folder = args.output or "missing_analysis_custom"

    else:
        print("❌ Error: Debe especificar --dataset o --file")
        parser.print_help()
        return

    # Validar que el archivo existe
    if not os.path.exists(dataset_path):
        print(f"❌ Error: No se encontró el archivo {dataset_path}")
        return

    try:
        # Crear analizador
        analyzer = MissingValuesAnalyzer(
            dataset_path=dataset_path,
            results_folder=results_folder
        )

        # Ejecutar análisis
        results = analyzer.run_analysis()

        print(f"✅ Análisis completado exitosamente")
        print(f"📁 Resultados guardados en: reports/{results_folder}/")

        # Mostrar resumen
        stats = results['statistics']
        print(f"\n📊 RESUMEN:")
        print(f"   Total de registros: {stats['dataset_shape'][0]:,}")
        print(f"   Total de features: {stats['dataset_shape'][1]:,}")
        print(f"   Features con valores faltantes: {stats['features_with_missing']}")
        print(f"   Porcentaje total de valores faltantes: {stats['missing_percentage_total']:.2f}%")

    except Exception as e:
        print(f"❌ Error ejecutando análisis: {e}")
        raise

    finally:
        # Cerrar conexión
        if 'analyzer' in locals():
            analyzer.close()


if __name__ == "__main__":
    main()
