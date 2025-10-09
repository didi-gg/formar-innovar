"""
Clase base para análisis de EDA (Exploratory Data Analysis).
Contiene constantes y lógica común utilizada por todos los scripts de análisis EDA.
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
import logging
import re
from abc import ABC, abstractmethod

# Configuración de warnings y logging
warnings.filterwarnings('ignore')

# Configurar matplotlib para suprimir mensajes de debug
import matplotlib
matplotlib.set_loglevel("WARNING")
import matplotlib.pyplot as plt
plt.set_loglevel("WARNING")

# Configurar logging para suprimir mensajes de debug
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# Agregar el directorio padre al path para importar utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.base_script import BaseScript

warnings.filterwarnings('ignore')
class EDAAnalysisBase(BaseScript, ABC):
    """
    Clase base abstracta para análisis de EDA que contiene:
    - Constantes comunes (variables excluidas, categóricas, numéricas)
    - Métodos de validación y carga de datos
    - Configuración de logging
    - Lógica común para todos los análisis EDA
    """

    # === CONSTANTES COMUNES ===

    # Variables a excluir del análisis
    EXCLUDED_FEATURES = [
        'sede',  # No es relevante
        'year',  # No es relevante
        'documento_identificación',  # No es relevante
        'moodle_user_id',  # No es relevante
        'edukrea_user_id',  # No es relevante
        'fecha_nacimiento',  # Ya está representada en edad_estudiante
        'id_grado',  # No es relevante
        'id_asignatura',  # No es relevante
        'id_docente',  # No es relevante
        'id_most_late_opened_module',  # No es relevante
        'id_least_opened_module',  # No es relevante
        'id_least_viewed_module',  # No es relevante
        'years_experience_ficc',  # Ya está representada en teacher_experiencia_nivel_ficc
        'years_experience_total',  # Ya está representada en teacher_experiencia_nivel
        'nivel',  # Variable objetivo categoría
        'resultado',  # Variable objetivo numérica
        'nota_final',  # Variable objetivo numérica
        'cog',  # Colinealidad con nota_final
        'proc',  # Colinealidad con nota_final
        'act',  # Colinealidad con nota_final
        'axi',  # Colinealidad con nota_final
        'valoración_emocional',  # Es una evaluación en lenguaje natural
        'teacher_experiencia_ficc_percentil',  # Ya está representada en teacher_experiencia_nivel_ficc
        'teacher_experiencia_total_percentil',  # Ya está representada en teacher_experiencia_nivel
        'teacher_nivel_educativo_percentil',  # Ya está representada en teacher_experiencia_nivel
        'teacher_nivel_educativo_num',  # Ya está representada en teacher_experiencia_nivel
    ]

    # Variables a excluir del análisis por entre 100% y 90% el mismo valor en todos los registros
    EXCLUDED_FEATURES_NULL_VARIANCE = [
        'max_days_after_end',
        'max_days_from_planned_start',
        'has_viewed_all_modules',
        'has_participated_all_modules',
        'min_interactions_per_module',
        'on_time_rate',
        'early_access_count',
        'late_access_count',
        'min_views_per_module',
        'mid_week_engagement',
        'median_interactions_per_module',
        'std_activity_grade',
        'login_regularity_score',
        'extra_activities',
        'common_trigrams',
        'count_jornada_madrugada',
        'avg_activity_grade',
        'max_activity_grade',
        'min_activity_grade',
        'graded_activities_count',
        'percent_graded_activities'
        'extra_activities',
        'min_time_per_module',
        'late_rate',
        'min_days_after_end',
        'min_days_from_planned_start',
        'interaction_to_view_ratio',
        'std_interactions_per_module',
        'max_interactions_in_a_module',
        'avg_interactions_per_module',
        
    ]

    # Variables numéricas conocidas
    NUMERIC_COLUMNS = [
        'edad_estudiante', 'participacion_clase', 'age', 'horas_semana_estudio_casa',
        'count_login_fri', 'student_total_interactions', 'count_login_mon',
        'avg_days_since_last_update', 'intensidad', 'total_hermanos', 'count_login_tue',
        'percent_collaboration', 'student_total_views', 'count_login_thu', 'max_inactividad',
        'interaction_to_view_ratio', 'percent_modules_viewed', 'teacher_experiencia_nivel_ficc',
        'teacher_experiencia_nivel', 'total_course_time_hours', 'total_hours',
        'median_views_per_module', 'percent_students_interacted', 'relative_interaction_percentile',
        'percent_students_viewed', 'sequence_match_ratio', 'num_students_viewed',
        'median_days_since_last_update', 'num_modules', 'min_days_from_planned_start',
        'std_days_from_planned_start', 'std_days_after_end', 'log_total_views',
        'percent_in_english', 'log_total_interactions', 'modules_participated',
        'median_days_from_planned_start', 'median_days_after_end', 'percent_modules_participated',
        'modules_viewed', 'avg_days_after_end', 'avg_days_from_planned_start',
        'percent_modules_viewed_interacciones_de_estudiantes', 'nota_final'
    ]

    # Variables categóricas conocidas
    KNOWN_CATEGORICAL = {
        'género', 'país_origen', 'tipo_vivienda', 'zona_vivienda', 
        'interés_estudios_superiores', 'medio_transporte', 'apoyo_familiar',
        'familia', 'actividades_extracurriculares', 'enfermedades',
        'proyección_vocacional', 'participación_clase', 'nee',
        'nivel_motivación', 'demuestra_confianza', 'rol_adicional',
        'nivel_educativo', 'estrato',
        'dia_preferido', 'jornada_preferida', 'time_engagement_level',
        'antigüedad', 'period', 'año_ingreso'
    }

    # Variables continuas conocidas (para evitar clasificación errónea)
    KNOWN_CONTINUOUS = {
        'intensidad', 'total_subjects', 'median_views_per_student', 
        'median_interactions_per_student', 'students_viewed_least_module',
        'median_interactions_per_module', 'min_views_per_module', 
        'min_interactions_per_module', 'max_days_after_end',
        'max_days_from_planned_start', 'on_time_rate', 'late_rate',
        'early_access_count', 'late_access_count', 'mid_week_engagement',
        'common_trigrams', 'correct_order_count', 'teacher_experiencia_nivel', 
        'teacher_experiencia_nivel_ficc'
    }

    # Variables objetivo
    TARGET_CATEGORICAL = 'nivel'  # Variable objetivo categórica
    TARGET_NUMERIC = 'nota_final'  # Variable objetivo numérica

    # Variables de agrupación para análisis
    GROUP_VARIABLES = ['sede', 'género']

    # Mapeo de niveles categóricos a numéricos
    LEVEL_MAPPING = {"Bajo": 0, "Básico": 1, "Alto": 2, "Superior": 3}

    # Nivel de significancia estadística
    ALPHA = 0.05

    def __init__(self, dataset_path=None, results_folder=None):
        """
        Inicializa la clase base de análisis EDA.

        Args:
            dataset_path (str): Ruta al archivo CSV del dataset (requerido)
            results_folder (str): Nombre del folder para guardar resultados (requerido, se creará en reports/)

        Raises:
            ValueError: Si dataset_path o results_folder son None
        """
        super().__init__()

        # Validar que los parámetros requeridos no sean None
        if dataset_path is None:
            raise ValueError("El parámetro 'dataset_path' es requerido y no puede ser None")

        if results_folder is None:
            raise ValueError("El parámetro 'results_folder' es requerido y no puede ser None")

        # Configurar logging adicional para suprimir mensajes de debug
        self._configure_logging()

        self.dataset_path = dataset_path
        self.results_folder = results_folder
        self.results_path = f'reports/{self.results_folder}'

        # Validar parámetros
        self._validate_parameters()

        # Inicializar atributos específicos de cada análisis
        self._initialize_analysis_attributes()

    def _configure_logging(self):
        """Configurar logging para suprimir mensajes de debug."""
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)

    def _validate_parameters(self):
        """Valida los parámetros de inicialización."""
        # Validar dataset_path
        if not isinstance(self.dataset_path, str):
            raise ValueError("dataset_path debe ser una cadena de texto")

        if not self.dataset_path.endswith('.csv'):
            raise ValueError("dataset_path debe ser un archivo CSV (.csv)")

        # Validar results_folder
        if not isinstance(self.results_folder, str):
            raise ValueError("results_folder debe ser una cadena de texto")

        # Validar caracteres permitidos en el nombre del folder (permitir barras para rutas)
        if not re.match(r'^[a-zA-Z0-9_/-]+$', self.results_folder):
            raise ValueError("results_folder solo puede contener letras, números, guiones, guiones bajos y barras diagonales")

    def get_beautiful_palette(self, n_colors, palette_name='plasma'):
        """
        Obtener una paleta de colores bonita.
        
        Args:
            n_colors (int): Número de colores necesarios
            palette_name (str): Nombre de la paleta ('plasma', 'viridis', 'inferno', 'magma', 'turbo', 'tab10', 'tab20', 'tab20b', 'tab20c', 'Set2', 'Set3')
        
        Returns:
            list: Lista de colores
        """
        palette_options = {
            'plasma': plt.cm.plasma,
            'viridis': plt.cm.viridis,
            'inferno': plt.cm.inferno,
            'magma': plt.cm.magma,
            'turbo': plt.cm.turbo,
            'tab10': plt.cm.tab10,
            'tab20': plt.cm.tab20,
            'tab20b': plt.cm.tab20b,
            'tab20c': plt.cm.tab20c,
            'Set2': plt.cm.Set2,
            'Set3': plt.cm.Set3,
            'rainbow': plt.cm.rainbow,
            'hsv': plt.cm.hsv
        }
        
        if palette_name in palette_options:
            return palette_options[palette_name](np.linspace(0, 1, n_colors))
        else:
            # Default a plasma si no se encuentra la paleta
            return plt.cm.plasma(np.linspace(0, 1, n_colors))

    @abstractmethod
    def _initialize_analysis_attributes(self):
        """
        Inicializar atributos específicos del análisis.
        Debe ser implementado por cada clase hija.
        """
        pass

    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Cargar y preparar los datos.

        Args:
            file_path (str): Ruta al archivo CSV. Si es None, usa self.dataset_path

        Returns:
            pd.DataFrame: Dataset cargado y filtrado
        """
        if file_path is None:
            file_path = self.dataset_path

        self.logger.info(f"Cargando datos desde: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No se encontró el dataset en: {file_path}")

        df = pd.read_csv(file_path)
        self.logger.info(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

        return df

    def get_valid_features(self, df: pd.DataFrame, exclude_targets: bool = True) -> list:
        """
        Obtiene las características válidas del dataset excluyendo las variables no relevantes.

        Args:
            df (pd.DataFrame): Dataset
            exclude_targets (bool): Si excluir las variables objetivo

        Returns:
            list: Lista de características válidas
        """
        excluded = set(self.EXCLUDED_FEATURES)
        excluded = excluded.union(set(self.EXCLUDED_FEATURES_NULL_VARIANCE))

        if exclude_targets:
            excluded.update([self.TARGET_CATEGORICAL, self.TARGET_NUMERIC])

        valid_features = [col for col in df.columns if col not in excluded]

        self.logger.info(f"Características válidas identificadas: {len(valid_features)}")
        return valid_features

    def identify_variable_types(self, df: pd.DataFrame, features: list = None) -> dict:
        """
        Identificar tipos de variables (categóricas vs continuas).

        Args:
            df (pd.DataFrame): Dataset
            features (list): Lista de características a analizar. Si es None, usa todas las válidas

        Returns:
            dict: Diccionario con listas de variables categóricas y continuas
        """
        if features is None:
            features = self.get_valid_features(df)

        categorical_vars = []
        continuous_vars = []

        for col in features:
            if col not in df.columns:
                continue

            # Determinar si es categórica o continua
            if col in self.KNOWN_CONTINUOUS:
                continuous_vars.append(col)
            elif col in self.KNOWN_CATEGORICAL:
                categorical_vars.append(col)
            elif df[col].dtype == 'object':
                categorical_vars.append(col)
            elif (df[col].dtype in ['int64', 'float64'] and 
                  df[col].nunique() <= 15 and  # Máximo 15 categorías
                  col not in self.NUMERIC_COLUMNS):  # No está en la lista de numéricas conocidas
                categorical_vars.append(col)
            else:
                continuous_vars.append(col)

        self.logger.info(f"Variables categóricas identificadas: {len(categorical_vars)}")
        self.logger.info(f"Variables continuas identificadas: {len(continuous_vars)}")

        return {
            'categorical': categorical_vars,
            'continuous': continuous_vars
        }

    def create_results_directory(self):
        """Crear directorio de resultados si no existe."""
        os.makedirs(self.results_path, exist_ok=True)
        self.logger.info(f"Directorio de resultados: {self.results_path}")

    def validate_target_variables(self, df: pd.DataFrame):
        """
        Validar que las variables objetivo existan en el dataset.

        Args:
            df (pd.DataFrame): Dataset a validar

        Raises:
            ValueError: Si no se encuentran las variables objetivo necesarias
        """
        missing_targets = []

        if self.TARGET_CATEGORICAL not in df.columns:
            missing_targets.append(self.TARGET_CATEGORICAL)

        if self.TARGET_NUMERIC not in df.columns:
            missing_targets.append(self.TARGET_NUMERIC)

        if missing_targets:
            raise ValueError(f"Variables objetivo faltantes en el dataset: {missing_targets}")

    def get_numeric_columns_available(self, df: pd.DataFrame) -> list:
        """
        Obtener columnas numéricas disponibles en el dataset.

        Args:
            df (pd.DataFrame): Dataset

        Returns:
            list: Lista de columnas numéricas disponibles
        """
        available_numeric = [col for col in self.NUMERIC_COLUMNS if col in df.columns]
        self.logger.info(f"Variables numéricas disponibles: {len(available_numeric)}")
        return available_numeric

    def filter_valid_data(self, df: pd.DataFrame, columns: list, min_observations: int = 10) -> pd.DataFrame:
        """
        Filtrar datos válidos (sin valores nulos) para las columnas especificadas.

        Args:
            df (pd.DataFrame): Dataset
            columns (list): Columnas a considerar
            min_observations (int): Número mínimo de observaciones válidas requeridas

        Returns:
            pd.DataFrame: Dataset filtrado

        Raises:
            ValueError: Si no hay suficientes observaciones válidas
        """
        mask = df[columns].notna().all(axis=1)
        df_filtered = df.loc[mask].copy()

        if len(df_filtered) < min_observations:
            raise ValueError(f"Insuficientes observaciones válidas: {len(df_filtered)} < {min_observations}")

        return df_filtered

    # === MÉTODOS DE INTERPRETACIÓN ESTADÍSTICA ===

    def interpret_cohens_d(self, d: float) -> str:
        """Interpretar Cohen's d (tamaño del efecto)."""
        d = abs(d)
        if d < 0.2:
            return "Muy pequeño"
        elif d < 0.5:
            return "Pequeño"
        elif d < 0.8:
            return "Moderado"
        else:
            return "Grande"

    def interpret_correlation(self, r: float) -> str:
        """Interpretar coeficiente de correlación."""
        r = abs(r)
        if r < 0.1:
            return "Muy débil"
        elif r < 0.3:
            return "Débil"
        elif r < 0.5:
            return "Moderado"
        elif r < 0.7:
            return "Fuerte"
        else:
            return "Muy fuerte"

    def interpret_cramers_v(self, v: float) -> str:
        """Interpretar V de Cramér (asociación categórica)."""
        if v < 0.1:
            return "Débil"
        elif v < 0.3:
            return "Moderado"
        elif v < 0.5:
            return "Fuerte"
        else:
            return "Muy fuerte"

    def interpret_eta_squared(self, eta_sq: float) -> str:
        """Interpretar eta cuadrado (tamaño del efecto ANOVA)."""
        if eta_sq < 0.01:
            return "Muy pequeño"
        elif eta_sq < 0.06:
            return "Pequeño"
        elif eta_sq < 0.14:
            return "Moderado"
        else:
            return "Grande"

    def interpret_r_effect(self, r: float) -> str:
        """Interpretar tamaño del efecto r (Mann-Whitney)."""
        r = abs(r)
        if r < 0.1:
            return "Muy pequeño"
        elif r < 0.3:
            return "Pequeño"
        elif r < 0.5:
            return "Moderado"
        else:
            return "Grande"

    def interpret_epsilon_squared(self, eps_sq: float) -> str:
        """Interpretar epsilon cuadrado (Kruskal-Wallis)."""
        if eps_sq < 0.01:
            return "Muy pequeño"
        elif eps_sq < 0.08:
            return "Pequeño"
        elif eps_sq < 0.26:
            return "Moderado"
        else:
            return "Grande"

    # === MÉTODOS ABSTRACTOS ===

    @abstractmethod
    def run_analysis(self):
        """
        Ejecutar el análisis completo.
        Debe ser implementado por cada clase hija.
        """
        pass

    def log_analysis_summary(self, **kwargs):
        """
        Registrar resumen del análisis.

        Args:
            **kwargs: Parámetros específicos del análisis para el resumen
        """
        self.logger.info("=== RESUMEN DEL ANÁLISIS ===")
        self.logger.info(f"Dataset: {self.dataset_path}")
        self.logger.info(f"Resultados guardados en: {self.results_path}")

        for key, value in kwargs.items():
            self.logger.info(f"{key}: {value}")

        self.logger.info("Análisis completado exitosamente!")

