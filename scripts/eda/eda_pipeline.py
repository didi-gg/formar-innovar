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
from typing import Dict, Any, Optional, Tuple

# Configuración de warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para suprimir mensajes de debug
import matplotlib
matplotlib.set_loglevel("WARNING")

# Agregar el directorio padre al path para importar utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.base_script import BaseScript

# Importar los scripts de análisis
from scripts.eda.missing_values_analysis import MissingValuesAnalyzer
from scripts.eda.features_selection import XGBoostFeatureSelector
from scripts.eda.statistical_filtering import StatisticalFilter
from scripts.eda.data_visualization import DataVisualizationAnalyzer
from scripts.eda.teacher_behavior_analysis import TeacherBehaviorAnalysis
from scripts.eda.course_analysis import CourseAnalysis
from scripts.eda.grades_analysis import GradesAnalysis
from scripts.eda.dropout_analysis import DropoutAnalysis
from scripts.eda.moodle_behavior_analysis import MoodleBehaviorAnalysis
from scripts.eda.students_analysis import StudentsAnalysis


class EDAMasterPipeline(BaseScript):
    """
    Pipeline maestro para ejecutar análisis completo de EDA.

    Ejecuta análisis dependientes del dataset en secuencia:
    0. Análisis de valores faltantes (mapas de calor, porcentajes, patrones)
    1. Selección de características (XGBoost + SHAP)
    2. Filtrado estadístico (Chi-cuadrado, correlación, ANOVA)
    3. Visualización de datos (distribución, correlación, scatter plots, box plots)

    Ejecuta análisis globales (una sola vez, independiente del dataset):
    - Análisis de cursos, calificaciones, retiro de estudiantes, Moodle, estudiantes y comportamiento docente
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
            'folder': None,
            'description': "Dataset completo con todas las materias (solo para análisis globales)"
        }
    }

    # Configuración de análisis dependientes del dataset
    DATASET_DEPENDENT_ANALYSES = [
        ('missing_values', MissingValuesAnalyzer, '00_missing_values'),
        ('features_selection', XGBoostFeatureSelector, '01_features_selection'),
        ('statistical_filtering', StatisticalFilter, '02_statistical_filtering'),
        ('data_visualization', DataVisualizationAnalyzer, '03_data_visualization'),
    ]

    # Configuración de análisis globales
    GLOBAL_ANALYSES = [
        ('course_analysis', CourseAnalysis, 'eda_courses', 'data/processed/full_short_dataset.csv'),
        ('grades_analysis', GradesAnalysis, 'eda_grades', 'data/interim/calificaciones/calificaciones_2021-2025.csv'),
        ('dropout_analysis', DropoutAnalysis, 'eda_analysis_dropout', 'data/interim/calificaciones/calificaciones_2021-2025.csv'),
        ('moodle_behavior', MoodleBehaviorAnalysis, 'eda_analysis_moodle_behavior', 'data/interim/moodle/student_course_interactions.csv'),
        ('students_analysis', StudentsAnalysis, 'eda_analysis_students', 'data/interim/estudiantes/estudiantes_clean.csv'),
        ('teacher_behavior', TeacherBehaviorAnalysis, 'eda_analysis_teacher_behavior', 'data/processed/full_short_dataset.csv'),
    ]

    def __init__(self, dataset_type: str):
        """
        Inicializa el pipeline de EDA.

        Args:
            dataset_type (str): Tipo de dataset predefinido ('moodle', 'no_moodle', 'full')
        """
        super().__init__()

        # Validar y configurar dataset
        if dataset_type not in self.DATASETS:
            raise ValueError(f"Tipo de dataset '{dataset_type}' no válido. Opciones: {list(self.DATASETS.keys())}")

        dataset_config = self.DATASETS[dataset_type]
        self.dataset_path = dataset_config['path']
        self.results_base_folder = dataset_config['folder']
        self.dataset_type = dataset_type
        self.dataset_description = dataset_config['description']
        self.results_base_path = f'reports/{self.results_base_folder}' if self.results_base_folder else None
        self.results = {}

    @classmethod
    def validate_dataset_exists(cls, dataset_type: str) -> Tuple[bool, str]:
        """Valida que un dataset predefinido exista en el sistema."""
        if dataset_type not in cls.DATASETS:
            return False, f"Tipo de dataset '{dataset_type}' no válido"

        dataset_path = cls.DATASETS[dataset_type]['path']
        if not os.path.exists(dataset_path):
            return False, f"Archivo no encontrado: {dataset_path}"

        return True, "Dataset disponible"

    def _create_directories(self):
        """Crea los directorios necesarios para los resultados."""
        if self.results_base_path is None:
            self.logger.info("No se crearán directorios (dataset usado solo para análisis globales)")
            return

        try:
            # Crear directorio base
            os.makedirs(self.results_base_path, exist_ok=True)
            self.logger.info(f"Directorio base creado: {self.results_base_path}")

            # Crear subdirectorios para análisis dependientes del dataset
            for _, _, subfolder in self.DATASET_DEPENDENT_ANALYSES:
                full_path = f'{self.results_base_path}/{subfolder}'
                os.makedirs(full_path, exist_ok=True)
                self.logger.info(f"Directorio creado: {full_path}")

        except Exception as e:
            self.logger.error(f"Error creando directorios: {e}")
            raise

    @staticmethod
    def _create_global_directories():
        """Crea los directorios para análisis globales (se ejecuta una sola vez)."""
        try:
            for _, _, folder, _ in EDAMasterPipeline.GLOBAL_ANALYSES:
                full_path = f'reports/{folder}'
                os.makedirs(full_path, exist_ok=True)
        except Exception as e:
            print(f"Error creando directorios globales: {e}")
            raise

    def _run_analysis(self, analysis_name: str, analyzer_class, dataset_path: str, results_folder: str) -> Dict[str, Any]:
        """
        Método genérico para ejecutar cualquier análisis.

        Args:
            analysis_name: Nombre del análisis
            analyzer_class: Clase del analizador
            dataset_path: Ruta del dataset
            results_folder: Carpeta de resultados

        Returns:
            dict: Resultados del análisis
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info(f"INICIANDO ANÁLISIS: {analysis_name.upper()}")
            self.logger.info("=" * 60)

            # Verificar que el dataset existe
            if not os.path.exists(dataset_path):
                self.logger.warning(f"⚠️ Dataset no encontrado: {dataset_path}")
                return {'status': 'skipped', 'reason': 'Dataset no encontrado'}

            # Crear instancia del analizador
            analyzer = analyzer_class(
                dataset_path=dataset_path,
                results_folder=results_folder
            )

            # Ejecutar análisis
            results = analyzer.run_analysis()

            # Cerrar conexión del analizador
            analyzer.close()

            self.logger.info(f"✅ Análisis {analysis_name} completado")
            return results

        except Exception as e:
            self.logger.error(f"❌ Error en análisis {analysis_name}: {e}")
            raise

    @staticmethod
    def run_global_analysis():
        """
        Ejecuta todos los análisis globales (una sola vez, independiente del dataset).

        Returns:
            dict: Resultados de los análisis globales
        """
        print(f"\n{'='*60}")
        print("EJECUTANDO ANÁLISIS GLOBALES")
        print(f"{'='*60}")

        try:
            # Crear directorios globales
            EDAMasterPipeline._create_global_directories()

            # Crear pipeline temporal para ejecutar análisis globales
            global_pipeline = EDAMasterPipeline('full')

            # Ejecutar cada análisis global
            for analysis_name, analyzer_class, folder, dataset_path in EDAMasterPipeline.GLOBAL_ANALYSES:
                print(f"📊 Ejecutando {analysis_name}...")
                global_pipeline._run_analysis(analysis_name, analyzer_class, dataset_path, folder)

            print("✅ Análisis globales completados exitosamente")

        except Exception as e:
            print(f"❌ Error en análisis globales: {e}")
            raise

    def _load_and_validate_dataset(self) -> pd.DataFrame:
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

    def run_dataset_dependent_analyses(self):
        """Ejecuta todos los análisis dependientes del dataset."""
        for analysis_name, analyzer_class, subfolder in self.DATASET_DEPENDENT_ANALYSES:
            try:
                results_folder = f'{self.results_base_folder}/{subfolder}'
                results = self._run_analysis(analysis_name, analyzer_class, self.dataset_path, results_folder)
                self.results[analysis_name] = results
            except Exception as e:
                self.logger.error(f"❌ Error en análisis {analysis_name}: {e}")
                self.results[analysis_name] = {'status': 'error', 'error': str(e)}



    def main(self, include_global_analysis: bool = False) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo de EDA.

        Args:
            include_global_analysis (bool): Si es True, ejecuta también los análisis globales
        """
        try:
            start_time = datetime.now()

            self.logger.info("🚀 INICIANDO PIPELINE COMPLETO DE EDA")
            self.logger.info(f"Tipo de dataset: {self.dataset_type}")
            self.logger.info(f"Descripción: {self.dataset_description}")
            self.logger.info(f"Archivo: {self.dataset_path}")
            self.logger.info(f"Resultados se guardarán en: {self.results_base_path}")

            # 1. Preparar entorno
            self._create_directories()
            self._load_and_validate_dataset()

            # 2. Ejecutar análisis dependientes del dataset
            self.run_dataset_dependent_analyses()

            # 3. Ejecutar análisis globales si se solicita
            if include_global_analysis:
                EDAMasterPipeline.run_global_analysis()

            # 4. Resumen final
            end_time = datetime.now()
            duration = end_time - start_time

            self.logger.info("=" * 60)
            self.logger.info("🎉 PIPELINE DE EDA COMPLETADO EXITOSAMENTE")
            self.logger.info("=" * 60)
            self.logger.info(f"Tiempo total de ejecución: {duration}")
            if self.results_base_path is not None:
                self.logger.info(f"Resultados disponibles en: {self.results_base_path}")
            else:
                self.logger.info("Dataset usado solo para análisis globales (sin folder de resultados)")

            return {
                'success': True,
                'duration': duration,
                'results_path': self.results_base_path,
                'analysis_results': self.results
            }

        except Exception as e:
            self.logger.error(f"❌ Error en pipeline de EDA: {e}")
            raise


def run_eda_pipeline(dataset_type: str, include_global_analysis: bool = True) -> Dict[str, Any]:
    """
    Función auxiliar para ejecutar el pipeline completo de EDA.

    Args:
        dataset_type (str): Tipo de dataset predefinido ('moodle', 'no_moodle', 'full')
        include_global_analysis (bool): Si es True, ejecuta también los análisis globales (por defecto True)

    Returns:
        dict: Resultados del pipeline completo
    """
    pipeline = EDAMasterPipeline(dataset_type)
    return pipeline.main(include_global_analysis=include_global_analysis)


def run_all_datasets() -> Dict[str, Any]:
    """
    Ejecuta el pipeline de EDA para todos los datasets disponibles.
    Los análisis globales se ejecutan solo una vez al final.

    Returns:
        dict: Resultados de todos los análisis
    """
    results = {}

    # 1. Ejecutar análisis para cada dataset (sin análisis globales)
    dataset_types = [dt for dt in EDAMasterPipeline.DATASETS.keys() if dt != 'full']
    for dataset_type in dataset_types:
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

            # Ejecutar análisis sin los análisis globales
            pipeline = EDAMasterPipeline(dataset_type)
            pipeline_result = pipeline.main(include_global_analysis=False)
            results[dataset_type] = pipeline_result

        except Exception as e:
            print(f"❌ Error en análisis de {dataset_type}: {e}")
            print(f"🛑 DETENIENDO PIPELINE COMPLETO")
            raise

    # 2. Ejecutar análisis globales una sola vez
    try:
        EDAMasterPipeline.run_global_analysis()
    except Exception as e:
        print(f"⚠️ Continuando sin análisis globales")
        results['global_analysis'] = {'error': str(e)}

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

    try:
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
        else:
            # Ejecutar análisis para un dataset específico (con análisis globales)
            is_valid, message = EDAMasterPipeline.validate_dataset_exists(args.type)
            if not is_valid:
                print(f"❌ Error: {message}")
                exit(1)

            pipeline = EDAMasterPipeline(args.type)
            pipeline.main(include_global_analysis=True)

    except Exception as e:
        print(f"\n{'='*60}")
        print("PIPELINE DETENIDO POR ERROR")
        print(f"{'='*60}")
        print(f"❌ Error: {e}")
        exit(1)
