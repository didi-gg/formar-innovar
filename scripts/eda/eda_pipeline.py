"""
Script maestro para ejecutar un pipeline completo de EDA (Exploratory Data Analysis).
Este script orquesta la ejecución de todos los análisis de EDA de manera secuencial.
"""

import os
import sys
import pandas as pd
import warnings
from datetime import datetime
from pathlib import Path

# Configuración de warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para suprimir mensajes de debug
import matplotlib
matplotlib.set_loglevel("WARNING")

# Agregar el directorio padre al path para importar utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.base_script import BaseScript

# Importar los scripts de análisis
from scripts.eda.features_selection import XGBoostFeatureSelector
from scripts.eda.statistical_filtering import StatisticalFilter
from scripts.eda.homogeneity_analysis import HomogeneityAnalysis


class EDAMasterPipeline(BaseScript):
    """
    Pipeline maestro para ejecutar análisis completo de EDA.

    Ejecuta en secuencia:
    1. Selección de características (XGBoost + SHAP)
    2. Filtrado estadístico (Chi-cuadrado, correlación, ANOVA)
    3. Análisis de homogeneidad (normalidad, varianzas, comparaciones)
    """

    # Configuración de datasets predefinidos
    DATASETS = {
        'moodle': {
            'path': "data/processed/full_short_dataset_moodle.csv",
            'folder': "eda_analysis_moodle",
            'description': "Dataset con materias que tienen datos de Moodle"
        },
        'no_moodle': {
            'path': "data/processed/full_short_dataset_no_moodle.csv", 
            'folder': "eda_analysis_no_moodle",
            'description': "Dataset con materias sin datos de Moodle"
        },
        'full': {
            'path': "data/processed/full_short_dataset.csv",
            'folder': "eda_analysis_full", 
            'description': "Dataset completo con todas las materias"
        }
    }

    def __init__(self, dataset_type):
        """
        Inicializa el pipeline de EDA.

        Args:
            dataset_type (str): Tipo de dataset predefinido ('moodle', 'no_moodle', 'full')
        """
        super().__init__()

        # Validar que el tipo de dataset es válido
        if dataset_type not in self.DATASETS:
            raise ValueError(f"Tipo de dataset '{dataset_type}' no válido. Opciones: {list(self.DATASETS.keys())}")

        # Configurar usando el dataset predefinido
        dataset_config = self.DATASETS[dataset_type]
        self.dataset_path = dataset_config['path']
        self.results_base_folder = dataset_config['folder']
        self.dataset_type = dataset_type
        self.dataset_description = dataset_config['description']

        self.results_base_path = f'reports/{self.results_base_folder}'

        # Validar parámetros
        self._validate_parameters()

        # Configurar subfolder para cada análisis
        self.analysis_folders = {
            'features_selection': f'{self.results_base_folder}/01_features_selection',
            'statistical_filtering': f'{self.results_base_folder}/02_statistical_filtering',
            'homogeneity_analysis': f'{self.results_base_folder}/03_homogeneity_analysis'
        }

        # Resultados de cada análisis
        self.results = {}

    @classmethod
    def validate_dataset_exists(cls, dataset_type):
        """Valida que un dataset predefinido exista en el sistema."""
        if dataset_type not in cls.DATASETS:
            return False, f"Tipo de dataset '{dataset_type}' no válido"

        dataset_path = cls.DATASETS[dataset_type]['path']
        if not os.path.exists(dataset_path):
            return False, f"Archivo no encontrado: {dataset_path}"

        return True, "Dataset disponible"

    def _validate_parameters(self):
        """Valida los parámetros de inicialización."""
        # Validar dataset_path
        if not isinstance(self.dataset_path, str):
            raise ValueError("dataset_path debe ser una cadena de texto")

        if not self.dataset_path.endswith('.csv'):
            raise ValueError("dataset_path debe ser un archivo CSV (.csv)")

        # Validar results_base_folder
        if not isinstance(self.results_base_folder, str):
            raise ValueError("results_base_folder debe ser una cadena de texto")

        # Validar caracteres permitidos en el nombre del folder
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', self.results_base_folder):
            raise ValueError("results_base_folder solo puede contener letras, números, guiones y guiones bajos")

    def _create_directories(self):
        """Crea los directorios necesarios para los resultados."""
        try:
            # Crear directorio base
            os.makedirs(self.results_base_path, exist_ok=True)
            self.logger.info(f"Directorio base creado: {self.results_base_path}")

            # Crear subdirectorios para cada análisis
            subdirs = ['01_features_selection', '02_statistical_filtering', '03_homogeneity_analysis']
            for subdir in subdirs:
                full_path = f'{self.results_base_path}/{subdir}'
                os.makedirs(full_path, exist_ok=True)
                self.logger.info(f"Directorio creado: {full_path}")

        except Exception as e:
            self.logger.error(f"Error creando directorios: {e}")
            raise

    def _load_and_validate_dataset(self):
        """Carga y valida el dataset."""
        try:
            if not os.path.exists(self.dataset_path):
                raise FileNotFoundError(f"No se encontró el dataset en: {self.dataset_path}")

            df = pd.read_csv(self.dataset_path)
            self.logger.info(f"Dataset cargado exitosamente desde {self.dataset_path}")
            self.logger.info(f"Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")

            # Validar columnas esenciales
            required_columns = ['nivel']  # Columna objetivo mínima requerida
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Columnas requeridas faltantes: {missing_columns}")

            self.logger.info("Validación del dataset completada exitosamente")
            return df

        except Exception as e:
            self.logger.error(f"Error cargando o validando dataset: {e}")
            raise

    def run_features_selection(self):
        """Ejecuta el análisis de selección de características."""
        try:
            self.logger.info("=" * 60)
            self.logger.info("INICIANDO ANÁLISIS 1: SELECCIÓN DE CARACTERÍSTICAS")
            self.logger.info("=" * 60)

            # Crear instancia del selector de características
            selector = XGBoostFeatureSelector(
                dataset_path=self.dataset_path,
                results_folder=f'{self.results_base_folder}/01_features_selection'
            )

            # Ejecutar análisis
            results = selector.run_analysis()

            # Guardar resultados
            self.results['features_selection'] = {
                'accuracy': results['accuracy'],
                'total_features': len(results['feature_importance']),
                'top_10_features': results['feature_importance'].head(10)['feature'].tolist(),
                'feature_importance_df': results['feature_importance'],
                'datos_totales': results['datos_totales']
            }

            self.logger.info("✅ Análisis de selección de características completado")
            self.logger.info(f"Precisión del modelo: {results['accuracy']:.4f}")
            self.logger.info(f"Total de características analizadas: {len(results['feature_importance'])}")

            # Cerrar conexión del selector
            selector.close()

            return results

        except Exception as e:
            self.logger.error(f"❌ Error en análisis de selección de características: {e}")
            raise

    def run_statistical_filtering(self):
        """Ejecuta el análisis de filtrado estadístico."""
        try:
            self.logger.info("=" * 60)
            self.logger.info("INICIANDO ANÁLISIS 2: FILTRADO ESTADÍSTICO")
            self.logger.info("=" * 60)

            # Crear instancia del filtro estadístico
            filter_analyzer = StatisticalFilter(
                dataset_path=self.dataset_path,
                results_folder=f'{self.results_base_folder}/02_statistical_filtering'
            )

            # Ejecutar análisis
            filter_analyzer.run_analysis()

            # Guardar resultados
            self.results['statistical_filtering'] = {
                'chi2_significant': len(filter_analyzer.results['selected_features']['chi2_significant']),
                'correlation_significant': len(filter_analyzer.results['selected_features']['correlation_significant']),
                'anova_significant': len(filter_analyzer.results['selected_features']['anova_significant']),
                'total_significant': len(set().union(
                    filter_analyzer.results['selected_features']['chi2_significant'],
                    filter_analyzer.results['selected_features']['correlation_significant'],
                    filter_analyzer.results['selected_features']['anova_significant']
                )),
                'status': 'completed'
            }

            self.logger.info("✅ Análisis de filtrado estadístico completado")
            self.logger.info(f"Variables significativas (Chi-cuadrado): {self.results['statistical_filtering']['chi2_significant']}")
            self.logger.info(f"Variables significativas (Correlación): {self.results['statistical_filtering']['correlation_significant']}")
            self.logger.info(f"Variables significativas (ANOVA): {self.results['statistical_filtering']['anova_significant']}")
            self.logger.info(f"Total variables significativas únicas: {self.results['statistical_filtering']['total_significant']}")

            # Cerrar conexión del analizador
            filter_analyzer.close()

        except Exception as e:
            self.logger.error(f"❌ Error en análisis de filtrado estadístico: {e}")
            raise

    def run_homogeneity_analysis(self):
        """Ejecuta el análisis de homogeneidad."""
        try:
            self.logger.info("=" * 60)
            self.logger.info("INICIANDO ANÁLISIS 3: ANÁLISIS DE HOMOGENEIDAD")
            self.logger.info("=" * 60)

            # Crear instancia del analizador de homogeneidad
            homogeneity_analyzer = HomogeneityAnalysis(
                dataset_path=self.dataset_path,
                results_folder=f'{self.results_base_folder}/03_homogeneity_analysis'
            )

            # Ejecutar análisis
            homogeneity_analyzer.run_analysis()

            # Guardar resultados
            normal_vars = sum(1 for r in homogeneity_analyzer.results['normality_tests'].values() 
                             if r.get('overall_normal', False))
            total_comparisons = sum(len(group_results) for group_results in homogeneity_analyzer.results['group_comparisons'].values())
            significant_comparisons = sum(
                sum(1 for r in group_results.values() if r.get('significant', False))
                for group_results in homogeneity_analyzer.results['group_comparisons'].values()
            )

            self.results['homogeneity_analysis'] = {
                'total_variables': len(homogeneity_analyzer.results['normality_tests']),
                'normal_variables': normal_vars,
                'total_comparisons': total_comparisons,
                'significant_comparisons': significant_comparisons,
                'status': 'completed'
            }

            self.logger.info("✅ Análisis de homogeneidad completado")
            self.logger.info(f"Variables analizadas: {self.results['homogeneity_analysis']['total_variables']}")
            self.logger.info(f"Variables con distribución normal: {self.results['homogeneity_analysis']['normal_variables']}")
            self.logger.info(f"Comparaciones realizadas: {self.results['homogeneity_analysis']['total_comparisons']}")
            self.logger.info(f"Comparaciones significativas: {self.results['homogeneity_analysis']['significant_comparisons']}")

            # Cerrar conexión del analizador
            homogeneity_analyzer.close()

        except Exception as e:
            self.logger.error(f"❌ Error en análisis de homogeneidad: {e}")
            raise

    def generate_summary_report(self):
        """Genera un reporte resumen de todos los análisis."""
        try:
            self.logger.info("=" * 60)
            self.logger.info("GENERANDO REPORTE RESUMEN")
            self.logger.info("=" * 60)

            summary_path = f'{self.results_base_path}/eda_summary_report.txt'

            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("REPORTE RESUMEN - ANÁLISIS EXPLORATORIO DE DATOS (EDA)\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Dataset analizado: {self.dataset_path}\n")
                f.write(f"Directorio de resultados: {self.results_base_path}\n\n")

                # Resumen de cada análisis
                for analysis_name, results in self.results.items():
                    f.write(f"\n{analysis_name.upper().replace('_', ' ')}\n")
                    f.write("-" * 40 + "\n")

                    if analysis_name == 'features_selection':
                        f.write(f"Precisión del modelo: {results['accuracy']:.4f}\n")
                        f.write(f"Total de características: {results['total_features']}\n")
                        f.write(f"Datos analizados: {results['datos_totales']}\n")
                        f.write("Top 10 características más importantes:\n")
                        for i, feature in enumerate(results['top_10_features'], 1):
                            f.write(f"  {i}. {feature}\n")

                    elif analysis_name == 'statistical_filtering':
                        f.write(f"Estado: {results.get('status', 'completado')}\n")
                        f.write(f"Variables significativas (Chi-cuadrado): {results.get('chi2_significant', 0)}\n")
                        f.write(f"Variables significativas (Correlación): {results.get('correlation_significant', 0)}\n")
                        f.write(f"Variables significativas (ANOVA): {results.get('anova_significant', 0)}\n")
                        f.write(f"Total variables significativas únicas: {results.get('total_significant', 0)}\n")

                    elif analysis_name == 'homogeneity_analysis':
                        f.write(f"Estado: {results.get('status', 'completado')}\n")
                        f.write(f"Variables analizadas: {results.get('total_variables', 0)}\n")
                        f.write(f"Variables con distribución normal: {results.get('normal_variables', 0)}\n")
                        f.write(f"Comparaciones realizadas: {results.get('total_comparisons', 0)}\n")
                        f.write(f"Comparaciones significativas: {results.get('significant_comparisons', 0)}\n")

                    else:
                        f.write(f"Estado: {results.get('status', 'completado')}\n")
                        if 'message' in results:
                            f.write(f"Mensaje: {results['message']}\n")

                f.write(f"\n\nArchivos generados en: {self.results_base_path}/\n")

            self.logger.info(f"📄 Reporte resumen generado: {summary_path}")

        except Exception as e:
            self.logger.error(f"❌ Error generando reporte resumen: {e}")
            raise

    def main(self):
        """Ejecuta el pipeline completo de EDA."""
        try:
            start_time = datetime.now()

            self.logger.info("🚀 INICIANDO PIPELINE COMPLETO DE EDA")
            self.logger.info(f"Tipo de dataset: {self.dataset_type}")
            self.logger.info(f"Descripción: {self.dataset_description}")
            self.logger.info(f"Archivo: {self.dataset_path}")
            self.logger.info(f"Resultados se guardarán en: {self.results_base_path}")

            # 1. Preparar entorno
            self._create_directories()
            df = self._load_and_validate_dataset()

            # 2. Ejecutar análisis en secuencia
            self.run_features_selection()
            self.run_statistical_filtering()
            self.run_homogeneity_analysis()

            # 3. Generar reporte resumen
            self.generate_summary_report()

            # 4. Resumen final
            end_time = datetime.now()
            duration = end_time - start_time

            self.logger.info("=" * 60)
            self.logger.info("🎉 PIPELINE DE EDA COMPLETADO EXITOSAMENTE")
            self.logger.info("=" * 60)
            self.logger.info(f"Tiempo total de ejecución: {duration}")
            self.logger.info(f"Resultados disponibles en: {self.results_base_path}")

            return {
                'success': True,
                'duration': duration,
                'results_path': self.results_base_path,
                'analysis_results': self.results
            }

        except Exception as e:
            self.logger.error(f"❌ Error en pipeline de EDA: {e}")
            raise


def run_eda_pipeline(dataset_type):
    """
    Función auxiliar para ejecutar el pipeline completo de EDA.

    Args:
        dataset_type (str): Tipo de dataset predefinido ('moodle', 'no_moodle', 'full')

    Returns:
        dict: Resultados del pipeline completo
    """
    pipeline = EDAMasterPipeline(dataset_type)
    return pipeline.main()


def run_all_datasets():
    """
    Ejecuta el pipeline de EDA para todos los datasets disponibles.

    Returns:
        dict: Resultados de todos los análisis
    """
    results = {}

    for dataset_type in EDAMasterPipeline.DATASETS.keys():
        print(f"\n{'='*60}")
        print(f"EJECUTANDO ANÁLISIS PARA: {dataset_type.upper()}")
        print(f"{'='*60}")

        try:
            # Validar que el dataset existe
            is_valid, message = EDAMasterPipeline.validate_dataset_exists(dataset_type)
            if not is_valid:
                print(f"❌ Saltando {dataset_type}: {message}")
                results[dataset_type] = {'error': message}
                continue

            # Ejecutar análisis
            pipeline_result = run_eda_pipeline(dataset_type)
            results[dataset_type] = pipeline_result

        except Exception as e:
            print(f"❌ Error en análisis de {dataset_type}: {e}")
            results[dataset_type] = {'error': str(e)}

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Pipeline maestro para análisis exploratorio de datos (EDA)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Ejecutar análisis para todos los datasets (por defecto)
  python eda_pipeline.py

  # Ejecutar análisis para materias con Moodle
  python eda_pipeline.py --type moodle

  # Ejecutar análisis para materias sin Moodle  
  python eda_pipeline.py --type no_moodle

  # Ejecutar análisis para dataset completo
  python eda_pipeline.py --type full

  # Los resultados se guardan automáticamente en folders predefinidos
        """
    )

    parser.add_argument('--type', '-t', type=str, 
                       choices=['moodle', 'no_moodle', 'full', 'all'],
                       default='all',
                       help='Tipo de dataset a analizar (por defecto: all)')

    args = parser.parse_args()

    if args.type == 'all':
        print("🚀 EJECUTANDO ANÁLISIS PARA TODOS LOS DATASETS")
        results = run_all_datasets()

        # Resumen final
        print(f"\n{'='*60}")
        print("RESUMEN FINAL")
        print(f"{'='*60}")
        for dataset_type, result in results.items():
            if 'error' in result:
                print(f"❌ {dataset_type}: {result['error']}")
            else:
                print(f"✅ {dataset_type}: Completado exitosamente")
        exit(0)

    else:
        # Ejecutar análisis para un dataset específico
        # Validar que el dataset existe
        is_valid, message = EDAMasterPipeline.validate_dataset_exists(args.type)
        if not is_valid:
            print(f"❌ Error: {message}")
            exit(1)

        pipeline = EDAMasterPipeline(args.type)
        pipeline.main()
