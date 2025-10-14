"""Script para análisis de comportamiento de estudiantes en Moodle."""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para mejor rendimiento en paralelo
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing
from datetime import datetime, timedelta

# Configuración de warnings y logging
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)

# Silenciar mensajes de debug adicionales
matplotlib.set_loglevel("WARNING")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.eda_analysis_base import EDAAnalysisBase


class MoodleBehaviorAnalysis(EDAAnalysisBase):
    """Analizador de comportamiento de estudiantes en Moodle."""

    def _initialize_analysis_attributes(self):
        """Inicializar atributos específicos del análisis de Moodle."""
        self.student_interactions_df = None
        self.student_logs_df = None
        self.modules_active_df = None
        self.sequence_features_df = None
        self.activity_sequences_df = None
        self.results = {}
        self.max_workers = max(1, multiprocessing.cpu_count() - 1)

    def load_moodle_data(self):
        """Carga todos los datasets de Moodle."""
        self.logger.info("Cargando datasets de Moodle...")

        # Rutas de los datasets
        base_path = os.path.dirname(self.dataset_path)
        datasets = {
            'student_interactions': f'{base_path}/student_course_interactions.csv',
            'student_logs': f'{base_path}/student_logs.csv',
            'modules_active': f'{base_path}/modules_active.csv',
            'sequence_features': f'{base_path}/sequence_analysis_features.csv',
            'activity_sequences': f'{base_path}/student_activity_sequences.csv'
        }

        # Verificar que existan los archivos
        for name, path in datasets.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"No se encontró el dataset {name} en: {path}")

        # Cargar datasets con optimizaciones de memoria
        self.logger.info("Cargando student_course_interactions.csv...")
        self.student_interactions_df = pd.read_csv(
            datasets['student_interactions'], 
            dtype={'year': 'int16', 'period': 'int8', 'id_grado': 'int8'},
            low_memory=False
        )

        self.logger.info("Cargando student_logs.csv...")
        # Para logs, solo cargar columnas necesarias para reducir memoria
        logs_columns = ['year', 'documento_identificación', 'courseid', 'timecreated', 'eventname', 'component', 'id_asignatura', 'id_grado', 'sede']
        self.student_logs_df = pd.read_csv(
            datasets['student_logs'], 
            dtype={'year': 'int16'},
            usecols=logs_columns,
            low_memory=False
        )

        self.logger.info("Cargando modules_active.csv...")
        self.modules_active_df = pd.read_csv(
            datasets['modules_active'],
            dtype={'year': 'int16', 'period': 'int8', 'id_grado': 'int8'},
            low_memory=False
        )

        self.logger.info("Cargando sequence_analysis_features.csv...")
        self.sequence_features_df = pd.read_csv(
            datasets['sequence_features'],
            dtype={'year': 'int16', 'period': 'float32', 'id_grado': 'int8'},
            low_memory=False
        )

        self.logger.info("Cargando student_activity_sequences.csv...")
        self.activity_sequences_df = pd.read_csv(
            datasets['activity_sequences'],
            dtype={'year': 'int16', 'period': 'float32', 'id_grado': 'int8'},
            low_memory=False
        )

        # Cargar datos de inscripciones
        enrollments_path = f'{base_path}/../estudiantes/enrollments.csv'
        if os.path.exists(enrollments_path):
            self.logger.info("Cargando enrollments.csv...")
            self.enrollments_df = pd.read_csv(
                enrollments_path,
                dtype={'year': 'int16', 'id_grado': 'int8'},
                low_memory=False
            )
            self.logger.info(f"Enrollments: {self.enrollments_df.shape[0]:,} registros")
        else:
            self.logger.warning(f"No se encontró el archivo de inscripciones: {enrollments_path}")
            self.enrollments_df = None

        # Cargar mapeo de nombres de asignaturas
        asignaturas_path = 'data/raw/tablas_maestras/asignaturas.csv'
        if os.path.exists(asignaturas_path):
            self.logger.info("Cargando mapeo de asignaturas...")
            asignaturas_df = pd.read_csv(asignaturas_path)
            # Crear diccionario id -> nombre
            self.asignaturas_map = dict(zip(asignaturas_df['id_asignatura'], asignaturas_df['nombre']))
            self.logger.info(f"Mapeo de asignaturas cargado: {len(self.asignaturas_map)} asignaturas")
        else:
            self.logger.warning(f"No se encontró el archivo de asignaturas: {asignaturas_path}")
            self.asignaturas_map = {}

        # Log de información básica
        self.logger.info(f"Student interactions: {self.student_interactions_df.shape[0]:,} registros")
        self.logger.info(f"Student logs: {self.student_logs_df.shape[0]:,} registros")
        self.logger.info(f"Modules active: {self.modules_active_df.shape[0]:,} registros")
        self.logger.info(f"Sequence features: {self.sequence_features_df.shape[0]:,} registros")
        self.logger.info(f"Activity sequences: {self.activity_sequences_df.shape[0]:,} registros")

        return True

    def get_asignatura_name(self, id_asignatura):
        """Obtiene el nombre de la asignatura dado su ID."""
        return self.asignaturas_map.get(id_asignatura, f'Asig. {id_asignatura}')

    def prepare_temporal_data(self):
        """Prepara los datos temporales para análisis de accesos diarios."""
        self.logger.info("Preparando datos temporales...")

        # Convertir timestamp a datetime (UTC)
        self.student_logs_df['datetime'] = pd.to_datetime(
            self.student_logs_df['timecreated'], 
            unit='s', 
            errors='coerce',
            utc=True
        )

        # Convertir a hora de Bogotá (UTC-5)
        self.logger.info("Convirtiendo timestamps de UTC a hora de Bogotá (UTC-5)...")
        self.student_logs_df['datetime'] = self.student_logs_df['datetime'].dt.tz_convert('America/Bogota')

        # Remover información de zona horaria para facilitar el procesamiento
        self.student_logs_df['datetime'] = self.student_logs_df['datetime'].dt.tz_localize(None)

        # Crear columnas de fecha y hora (ya en hora de Bogotá)
        self.student_logs_df['date'] = self.student_logs_df['datetime'].dt.date
        self.student_logs_df['hour'] = self.student_logs_df['datetime'].dt.hour
        self.student_logs_df['day_of_week'] = self.student_logs_df['datetime'].dt.day_name()

        # Filtrar datos válidos
        valid_logs = self.student_logs_df.dropna(subset=['datetime'])
        self.logger.info(f"Registros con fecha válida: {len(valid_logs):,}")

        self.student_logs_df = valid_logs
        return True

    def analyze_daily_accesses(self):
        """Analiza los accesos diarios a Moodle."""
        self.logger.info("Analizando accesos diarios...")

        # Accesos por día
        daily_accesses = (
            self.student_logs_df.groupby('date', as_index=False)
            .agg({
                'documento_identificación': 'nunique',
                'timecreated': 'count'
            })
            .rename(columns={
                'documento_identificación': 'estudiantes_unicos',
                'timecreated': 'total_accesos'
            })
        )

        # Convertir date a datetime para gráficos
        daily_accesses['date'] = pd.to_datetime(daily_accesses['date'])
        daily_accesses = daily_accesses.sort_values('date')

        # Estadísticas básicas
        stats = {
            'promedio_accesos_dia': daily_accesses['total_accesos'].mean(),
            'maximo_accesos_dia': daily_accesses['total_accesos'].max(),
            'minimo_accesos_dia': daily_accesses['total_accesos'].min(),
            'promedio_estudiantes_dia': daily_accesses['estudiantes_unicos'].mean(),
            'total_dias_actividad': len(daily_accesses)
        }

        self.logger.info(f"Promedio de accesos por día: {stats['promedio_accesos_dia']:.1f}")
        self.logger.info(f"Promedio de estudiantes únicos por día: {stats['promedio_estudiantes_dia']:.1f}")

        self.results['daily_accesses'] = daily_accesses
        self.results['daily_stats'] = stats

        return daily_accesses

    def analyze_monthly_accesses_by_sede(self):
        """Analiza los accesos mensuales por sede con tasa por estudiante inscrito."""
        self.logger.info("Analizando accesos mensuales por sede...")

        # Filtrar solo registros con sede conocida
        logs_with_sede = self.student_logs_df.dropna(subset=['sede'])

        # Crear columna año-mes y año
        logs_with_sede['year_month'] = logs_with_sede['datetime'].dt.to_period('M')
        logs_with_sede['year'] = logs_with_sede['datetime'].dt.year

        # Accesos por mes y sede
        monthly_accesses = (
            logs_with_sede.groupby(['year_month', 'sede', 'year'], as_index=False)
            .agg({
                'timecreated': 'count'
            })
            .rename(columns={
                'timecreated': 'total_accesos'
            })
        )

        # Convertir year_month a string para mejor manejo
        monthly_accesses['year_month_str'] = monthly_accesses['year_month'].astype(str)

        # Calcular tasa por estudiante inscrito si tenemos los datos
        if self.enrollments_df is not None:
            self.logger.info("Calculando tasa de accesos por cada 10 estudiantes inscritos...")

            # Inscripciones por año y sede
            enrollments_by_year_sede = (
                self.enrollments_df.groupby(['year', 'sede'], as_index=False)['documento_identificación']
                .nunique()
                .rename(columns={'documento_identificación': 'estudiantes_inscritos'})
            )

            # Hacer merge con los datos de accesos mensuales
            monthly_accesses = monthly_accesses.merge(
                enrollments_by_year_sede, 
                on=['year', 'sede'], 
                how='left'
            )

            # Calcular tasa por cada 10 estudiantes
            monthly_accesses['accesos_por_10_estudiantes'] = (
                monthly_accesses['total_accesos'] / monthly_accesses['estudiantes_inscritos'] * 10
            ).fillna(0)

            self.logger.info("Resumen de inscripciones por año y sede:")
            for _, row in enrollments_by_year_sede.iterrows():
                self.logger.info(f"  {row['year']} - {row['sede']}: {row['estudiantes_inscritos']} inscritos")
        else:
            self.logger.warning("No se encontraron datos de inscripciones, no se calcularán tasas")
            monthly_accesses['estudiantes_inscritos'] = 0
            monthly_accesses['accesos_por_10_estudiantes'] = 0

        # Obtener lista de sedes únicas
        sedes_list = sorted(monthly_accesses['sede'].unique().tolist())

        self.logger.info(f"Sedes encontradas: {sedes_list}")
        self.logger.info(f"Períodos analizados: {sorted(monthly_accesses['year_month_str'].unique())}")

        self.results['monthly_accesses'] = monthly_accesses
        self.results['sedes_list_monthly'] = sedes_list

        return monthly_accesses

    def analyze_unique_students_by_month_sede(self):
        """Analiza los estudiantes únicos por año-mes y sede con porcentajes de inscripciones."""
        self.logger.info("Analizando estudiantes únicos por año-mes y sede...")

        # Filtrar solo registros con sede conocida
        logs_with_sede = self.student_logs_df.dropna(subset=['sede'])

        # Crear columna año-mes (ej: "2024-01", "2024-02")
        logs_with_sede['year_month'] = logs_with_sede['datetime'].dt.to_period('M')
        logs_with_sede['year_month_str'] = logs_with_sede['year_month'].astype(str)
        logs_with_sede['year'] = logs_with_sede['datetime'].dt.year

        # Estudiantes únicos por año-mes y sede
        unique_students = (
            logs_with_sede.groupby(['year_month_str', 'sede', 'year'], as_index=False)['documento_identificación']
            .nunique()
            .rename(columns={'documento_identificación': 'estudiantes_unicos'})
        )

        # Calcular inscripciones por año y sede si tenemos los datos
        if self.enrollments_df is not None:
            self.logger.info("Calculando porcentajes basados en inscripciones...")

            # Inscripciones por año y sede
            enrollments_by_year_sede = (
                self.enrollments_df.groupby(['year', 'sede'], as_index=False)['documento_identificación']
                .nunique()
                .rename(columns={'documento_identificación': 'estudiantes_inscritos'})
            )

            # Hacer merge con los datos de estudiantes únicos
            unique_students = unique_students.merge(
                enrollments_by_year_sede, 
                on=['year', 'sede'], 
                how='left'
            )

            # Calcular porcentaje de acceso
            unique_students['porcentaje_acceso'] = (
                unique_students['estudiantes_unicos'] / unique_students['estudiantes_inscritos'] * 100
            ).fillna(0)

            self.logger.info("Resumen de inscripciones por año y sede:")
            for _, row in enrollments_by_year_sede.iterrows():
                self.logger.info(f"  {row['year']} - {row['sede']}: {row['estudiantes_inscritos']} inscritos")
        else:
            self.logger.warning("No se encontraron datos de inscripciones, no se calcularán porcentajes")
            unique_students['estudiantes_inscritos'] = 0
            unique_students['porcentaje_acceso'] = 0

        # Obtener todos los meses únicos ordenados cronológicamente
        all_months = sorted(logs_with_sede['year_month'].unique())
        months_list = [str(month) for month in all_months]

        # Filtrar solo los meses que tienen datos
        months_with_data = sorted(unique_students['year_month_str'].unique())

        sedes_list = sorted(unique_students['sede'].unique())

        self.logger.info(f"Meses analizados: {months_with_data}")
        self.logger.info(f"Sedes encontradas: {sedes_list}")
        self.logger.info(f"Total combinaciones año-mes encontradas: {len(unique_students)}")

        # Mostrar resumen por mes
        month_summary = unique_students.groupby('year_month_str').agg({
            'estudiantes_unicos': 'sum',
            'porcentaje_acceso': 'mean'
        }).sort_index()

        self.logger.info("Resumen por mes:")
        for month, row in month_summary.iterrows():
            self.logger.info(f"  {month}: {row['estudiantes_unicos']} estudiantes únicos ({row['porcentaje_acceso']:.1f}% promedio)")

        self.results['unique_students_by_month'] = unique_students
        self.results['months_list'] = months_with_data
        self.results['sedes_list_months'] = sedes_list

        return unique_students

    def analyze_hourly_patterns(self):
        """Analiza los patrones de acceso por hora del día."""
        self.logger.info("Analizando patrones por hora...")

        hourly_accesses = (
            self.student_logs_df.groupby('hour', as_index=False)
            .agg({
                'documento_identificación': 'nunique',
                'timecreated': 'count'
            })
            .rename(columns={
                'documento_identificación': 'estudiantes_unicos',
                'timecreated': 'total_accesos'
            })
        )

        # Asegurar que tenemos todas las horas (0-23)
        all_hours = pd.DataFrame({'hour': range(24)})
        hourly_accesses = all_hours.merge(hourly_accesses, on='hour', how='left')
        hourly_accesses = hourly_accesses.fillna(0)

        self.results['hourly_accesses'] = hourly_accesses
        return hourly_accesses

    def analyze_course_participation_rate(self):
        """Analiza la tasa de vistas y participación usando mediana (métrica robusta)."""
        self.logger.info("Analizando tasa de participación de cursos por sede...")

        # Calcular métricas usando MEDIANA (más robusta que promedio)
        # 1. Mediana de % de módulos vistos
        # 2. Mediana de % de módulos con participación
        course_participation = (
            self.student_interactions_df.groupby(['id_asignatura', 'sede'], as_index=False)
            .agg({
                'documento_identificación': 'nunique',  # Total de estudiantes inscritos
                'percent_modules_viewed': 'median',  # Mediana de % de módulos vistos
                'percent_modules_participated': 'median',  # Mediana de % de participación
                'modules_viewed': 'sum',  # Total de vistas de módulos
                'total_modules': 'mean'  # Promedio de módulos totales
            })
            .rename(columns={
                'documento_identificación': 'estudiantes_inscritos',
                'percent_modules_viewed': 'mediana_vistas',
                'percent_modules_participated': 'mediana_participacion'
            })
        )

        # Convertir de decimal a porcentaje (0.75 -> 75.0)
        course_participation['mediana_vistas'] = (
            course_participation['mediana_vistas'] * 100
        ).round(1)

        course_participation['mediana_participacion'] = (
            course_participation['mediana_participacion'] * 100
        ).round(1)

        # Calcular estudiantes que accedieron (al menos 1 módulo visto)
        students_with_access = (
            self.student_interactions_df[
                self.student_interactions_df['modules_viewed'] > 0
            ]
            .groupby(['id_asignatura', 'sede'], as_index=False)['documento_identificación']
            .nunique()
            .rename(columns={'documento_identificación': 'estudiantes_con_acceso'})
        )

        # Calcular estudiantes que participaron (al menos 1 módulo con participación)
        students_with_participation = (
            self.student_interactions_df[
                self.student_interactions_df['modules_participated'] > 0
            ]
            .groupby(['id_asignatura', 'sede'], as_index=False)['documento_identificación']
            .nunique()
            .rename(columns={'documento_identificación': 'estudiantes_con_participacion'})
        )

        # Agregar información de estudiantes con acceso y participación
        course_participation = course_participation.merge(
            students_with_access,
            on=['id_asignatura', 'sede'],
            how='left'
        ).merge(
            students_with_participation,
            on=['id_asignatura', 'sede'],
            how='left'
        )

        course_participation['estudiantes_con_acceso'] = course_participation['estudiantes_con_acceso'].fillna(0)
        course_participation['estudiantes_con_participacion'] = course_participation['estudiantes_con_participacion'].fillna(0)



        # Filtrar cursos con al menos 5 estudiantes inscritos para evitar sesgos
        course_participation = course_participation[
            course_participation['estudiantes_inscritos'] >= 5
        ]

        # Ordenar por mediana de vistas
        course_participation = course_participation.sort_values('mediana_vistas', ascending=False)

        # Top y bottom 15 para tener variedad por sede
        top_participation = course_participation.head(15)
        bottom_participation = course_participation.tail(15)

        self.logger.info(f"Total de cursos-sede analizados: {len(course_participation)}")
        self.logger.info("Nota: Usando MEDIANA (métrica robusta, no afectada por outliers)")
        for sede in course_participation['sede'].unique():
            sede_data = course_participation[course_participation['sede'] == sede]
            if len(sede_data) > 0:
                median_views = sede_data['mediana_vistas'].median()
                median_participation = sede_data['mediana_participacion'].median()
                total_students = sede_data['estudiantes_inscritos'].sum()
                students_with_access = sede_data['estudiantes_con_acceso'].sum()
                students_with_part = sede_data['estudiantes_con_participacion'].sum()
                self.logger.info(
                    f"Sede {sede}: {len(sede_data)} cursos | "
                    f"Mediana vistas: {median_views:.1f}% | "
                    f"Mediana participación: {median_participation:.1f}% | "
                    f"Estudiantes: {int(students_with_access)}/{int(total_students)} con acceso, "
                    f"{int(students_with_part)}/{int(total_students)} con participación"
                )

        self.results['course_participation'] = course_participation
        self.results['top_participation'] = top_participation
        self.results['bottom_participation'] = bottom_participation

        return course_participation

    def analyze_behavior_by_grade(self):
        """Analiza el comportamiento por grado usando student_interactions."""
        self.logger.info("Analizando comportamiento por grado...")

        # Usar student_interactions que tiene información de grado
        grade_behavior = (
            self.student_interactions_df.groupby('id_grado', as_index=False)
            .agg({
                'documento_identificación': 'nunique',
                'total_views': 'mean',
                'total_interactions': 'mean',
                'percent_modules_viewed': 'mean',
                'percent_modules_participated': 'mean',
                'total_course_time_hours': 'mean'
            })
            .rename(columns={
                'documento_identificación': 'estudiantes_unicos',
                'total_views': 'promedio_vistas',
                'total_interactions': 'promedio_interacciones',
                'percent_modules_viewed': 'porcentaje_modulos_vistos',
                'percent_modules_participated': 'porcentaje_participacion',
                'total_course_time_hours': 'promedio_horas_curso'
            })
        )

        # Redondear valores para mejor presentación
        numeric_columns = ['promedio_vistas', 'promedio_interacciones', 
                          'porcentaje_modulos_vistos', 'porcentaje_participacion', 
                          'promedio_horas_curso']
        for col in numeric_columns:
            if col in grade_behavior.columns:
                grade_behavior[col] = grade_behavior[col].round(2)

        self.logger.info(f"Grados analizados: {sorted(grade_behavior['id_grado'].tolist())}")

        self.results['grade_behavior'] = grade_behavior
        return grade_behavior

    def analyze_sequence_patterns(self):
        """Analiza los patrones de navegación por secuencias."""
        self.logger.info("Analizando patrones de secuencia de navegación...")

        # Filtrar solo asignaturas 1, 2, 3, 4
        asignaturas_filtradas = [1, 2, 3, 4]

        sequence_data = self.sequence_features_df[
            self.sequence_features_df['id_asignatura'].isin(asignaturas_filtradas)
        ].copy()

        # Calcular métricas por asignatura, grado y sede
        sequence_metrics = (
            sequence_data.groupby(['id_asignatura', 'id_grado', 'sede'], as_index=False)
            .agg({
                'sequence_match_ratio': 'median',  # Mediana de adherencia a la secuencia
                'levenshtein_normalized': 'median',  # Mediana de distancia normalizada
                'correct_order_ratio': 'median',  # Mediana de orden correcto
                'missing_activities': 'median',  # Mediana de actividades faltantes
                'extra_activities': 'median',  # Mediana de actividades extras
                'documento_identificación': 'nunique'  # Total de estudiantes
            })
            .rename(columns={
                'documento_identificación': 'total_estudiantes'
            })
        )

        # Convertir de decimal a porcentaje (0-100)
        sequence_metrics['sequence_match_ratio'] = (sequence_metrics['sequence_match_ratio'] * 100).round(1)
        sequence_metrics['levenshtein_normalized'] = (sequence_metrics['levenshtein_normalized'] * 100).round(1)
        sequence_metrics['correct_order_ratio'] = (sequence_metrics['correct_order_ratio'] * 100).round(1)

        self.logger.info(f"Total de combinaciones asignatura-grado-sede analizadas: {len(sequence_metrics)}")

        # Log por sede
        for sede in sequence_metrics['sede'].unique():
            sede_data = sequence_metrics[sequence_metrics['sede'] == sede]
            avg_match = sede_data['sequence_match_ratio'].mean()
            avg_distance = sede_data['levenshtein_normalized'].mean()
            total_students = sede_data['total_estudiantes'].sum()

            self.logger.info(
                f"Sede {sede}: {len(sede_data)} combinaciones | "
                f"Adherencia promedio: {avg_match:.1f}% | "
                f"Distancia promedio: {avg_distance:.1f}% | "
                f"Total estudiantes: {int(total_students)}"
            )

        self.results['sequence_metrics'] = sequence_metrics
        return sequence_metrics

    def plot_monthly_accesses_by_sede(self):
        """Genera gráfico de barras de accesos mensuales por sede."""
        self.logger.info("Generando gráfico de accesos mensuales por sede...")

        monthly_accesses = self.results['monthly_accesses']
        sedes_list = self.results['sedes_list_monthly']

        # Crear pivot tables para tasa y totales
        pivot_rate = monthly_accesses.pivot(index='year_month_str', columns='sede', values='accesos_por_10_estudiantes')
        pivot_total = monthly_accesses.pivot(index='year_month_str', columns='sede', values='total_accesos')

        pivot_rate = pivot_rate.fillna(0)
        pivot_total = pivot_total.fillna(0)

        # Configurar el gráfico
        fig, ax = plt.subplots(figsize=(16, 8))

        # Obtener colores para cada sede
        colors = self.get_beautiful_palette(len(sedes_list), palette_name='Set2')

        # Crear gráfico de barras agrupadas
        x_pos = np.arange(len(pivot_rate.index))
        bar_width = 0.8 / len(sedes_list)

        for i, sede in enumerate(sedes_list):
            if sede in pivot_rate.columns:
                rate_values = pivot_rate[sede].values
                total_values = pivot_total[sede].values
                positions = x_pos + (i - len(sedes_list)/2 + 0.5) * bar_width

                bars = ax.bar(positions, rate_values, bar_width, 
                            label=sede, color=colors[i], alpha=0.8, 
                            edgecolor='black', linewidth=0.5)

                # Agregar valores en las barras (tasa y total)
                for bar, rate, total in zip(bars, rate_values, total_values):
                    if rate > 0:
                        height = bar.get_height()
                        # Mostrar tasa y total de accesos
                        label_text = f'{rate:.1f}\n({int(total):,})'
                        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                               label_text, ha='center', va='bottom', 
                               fontsize=8, fontweight='bold')

        # Configuración del gráfico
        ax.set_xlabel('Año - Mes', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accesos por cada 10 Estudiantes', fontsize=14, fontweight='bold')
        ax.set_title('Tasa de Accesos Mensuales a Moodle por cada 10 Estudiantes por Sede', 
                    fontsize=16, fontweight='bold', pad=20)

        # Configurar eje X
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pivot_rate.index, rotation=45, ha='right')

        # Leyenda
        ax.legend(title='Sede', fontsize=11, title_fontsize=12, 
                 loc='upper left', framealpha=0.9)

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Agregar nota explicativa
        ax.text(0.98, 0.90, 'Formato: Tasa por 10 estudiantes\n(Total de accesos)', 
                transform=ax.transAxes, fontsize=9, 
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7, pad=0.5))

        # Ajustar layout
        plt.tight_layout()

        # Guardar
        output_path = f'{self.results_path}/monthly_accesses_rate_by_sede.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✅ Gráfico guardado: {output_path}")
        return output_path

    def plot_unique_students_by_month_sede(self):
        """Genera gráfico de barras de porcentaje de estudiantes únicos por año-mes y sede."""
        self.logger.info("Generando gráfico de porcentaje de estudiantes únicos por año-mes y sede...")

        unique_students = self.results['unique_students_by_month']
        months_list = self.results['months_list']
        sedes_list = self.results['sedes_list_months']

        self.logger.info(f"Meses a graficar: {months_list}")
        self.logger.info(f"Sedes a graficar: {sedes_list}")

        # Crear pivot tables para porcentajes y conteos
        pivot_percentage = unique_students.pivot(index='year_month_str', columns='sede', values='porcentaje_acceso')
        pivot_count = unique_students.pivot(index='year_month_str', columns='sede', values='estudiantes_unicos')

        pivot_percentage = pivot_percentage.fillna(0)
        pivot_count = pivot_count.fillna(0)

        # Asegurar que tenemos todos los meses y sedes
        for month in months_list:
            if month not in pivot_percentage.index:
                pivot_percentage.loc[month] = 0
                pivot_count.loc[month] = 0

        for sede in sedes_list:
            if sede not in pivot_percentage.columns:
                pivot_percentage[sede] = 0
                pivot_count[sede] = 0

        # Reordenar por meses y sedes
        pivot_percentage = pivot_percentage.reindex(months_list)
        pivot_percentage = pivot_percentage[sedes_list]
        pivot_count = pivot_count.reindex(months_list)
        pivot_count = pivot_count[sedes_list]

        self.logger.info(f"Dimensiones del pivot: {pivot_percentage.shape}")
        self.logger.info(f"Meses en pivot: {list(pivot_percentage.index)}")

        # Configurar el gráfico - más ancho para acomodar la leyenda
        fig, ax = plt.subplots(figsize=(max(18, len(months_list) * 1.2), 8))

        # Obtener colores para cada sede
        colors = self.get_beautiful_palette(len(sedes_list), palette_name='Set1')

        # Crear gráfico de barras agrupadas
        x_pos = np.arange(len(months_list))
        bar_width = 0.8 / len(sedes_list) if len(sedes_list) > 0 else 0.8

        for i, sede in enumerate(sedes_list):
            percentage_values = pivot_percentage[sede].values
            count_values = pivot_count[sede].values
            positions = x_pos + (i - len(sedes_list)/2 + 0.5) * bar_width

            bars = ax.bar(positions, percentage_values, bar_width, 
                        label=sede, color=colors[i], alpha=0.8, 
                        edgecolor='black', linewidth=0.5)

            # Agregar valores en las barras (porcentaje y conteo)
            for bar, percentage, count in zip(bars, percentage_values, count_values):
                if percentage > 0:
                    height = bar.get_height()
                    # Mostrar porcentaje y conteo sobre la barra
                    label_text = f'{percentage:.1f}%\n({int(count)})'

                    # Posicionar siempre sobre la barra
                    y_pos = height + 1

                    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                           label_text, ha='center', va='bottom', 
                           fontsize=8, fontweight='bold', color='black')

        # Configuración del gráfico
        ax.set_xlabel('Año - Mes', fontsize=14, fontweight='bold')
        ax.set_ylabel('Porcentaje de Acceso (%)', fontsize=14, fontweight='bold')
        ax.set_title('Porcentaje de Estudiantes Únicos que Acceden a Moodle por Año-Mes y Sede', 
                    fontsize=16, fontweight='bold', pad=20)

        # Configurar eje X
        ax.set_xticks(x_pos)
        ax.set_xticklabels(months_list, fontsize=10, rotation=45, ha='right')

        # Leyenda - colocar más abajo para evitar traslapes
        ax.legend(title='Sede', fontsize=11, title_fontsize=12, 
                 loc='lower center', bbox_to_anchor=(0.5, -0.35), 
                 ncol=len(sedes_list), framealpha=0.9)

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Ajustar límites del eje Y (0-100% con espacio adicional para etiquetas)
        max_percentage = pivot_percentage.max().max() if not pivot_percentage.empty else 100
        ax.set_ylim(0, min(120, max_percentage + 15))  # Más espacio arriba para las etiquetas

        # Ajustar layout con espacio para la leyenda
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # Dar espacio para la leyenda

        # Guardar
        output_path = f'{self.results_path}/unique_students_percentage_by_month_sede.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✅ Gráfico guardado: {output_path}")
        return output_path

    def plot_hourly_patterns(self):
        """Genera gráfico de patrones de acceso por hora."""
        self.logger.info("Generando gráfico de patrones por hora...")

        hourly_accesses = self.results['hourly_accesses']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Gráfico 1: Barras de accesos por hora
        bars1 = ax1.bar(hourly_accesses['hour'], hourly_accesses['total_accesos'], 
                       color='#F18F01', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_title('Distribución de Accesos por Hora del Día', 
                     fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Hora del Día', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Total de Accesos', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(0, 24, 2))
        ax1.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Gráfico 2: Estudiantes únicos por hora
        bars2 = ax2.bar(hourly_accesses['hour'], hourly_accesses['estudiantes_unicos'], 
                       color='#C73E1D', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax2.set_title('Estudiantes Únicos por Hora del Día', 
                     fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Hora del Día', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Estudiantes Únicos', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(0, 24, 2))
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

        plt.tight_layout()

        # Guardar
        output_path = f'{self.results_path}/hourly_access_patterns.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✅ Gráfico guardado: {output_path}")
        return output_path

    def plot_course_participation_rate(self):
        """Genera dos mapas de calor por sede: % vistas y % participación (usando mediana)."""
        self.logger.info("Generando mapas de calor de participación por sede...")

        course_participation = self.results['course_participation']

        # Filtrar solo asignaturas 1, 2, 3, 4
        asignaturas_filtradas = [1, 2, 3, 4]

        # Necesitamos agregar id_grado al análisis
        # Obtener id_grado desde student_interactions_df
        grado_mapping = self.student_interactions_df[['id_asignatura', 'id_grado', 'sede']].drop_duplicates()

        # Hacer merge para agregar id_grado
        course_participation_with_grade = course_participation.merge(
            grado_mapping,
            on=['id_asignatura', 'sede'],
            how='left'
        )

        # Filtrar por asignaturas 1-4
        course_participation_filtered = course_participation_with_grade[
            course_participation_with_grade['id_asignatura'].isin(asignaturas_filtradas)
        ]

        # Obtener lista de sedes únicas
        sedes = sorted(course_participation_filtered['sede'].unique())

        # Crear dos mapas de calor por cada sede
        output_paths = []

        for sede in sedes:
            self.logger.info(f"Generando mapas de calor para sede: {sede}")

            # Filtrar datos de la sede
            sede_data = course_participation_filtered[
                course_participation_filtered['sede'] == sede
            ].copy()

            # === MAPA 1: % DE MÓDULOS VISTOS ===
            pivot_vistas = sede_data.pivot_table(
                index='id_grado',
                columns='id_asignatura',
                values='mediana_vistas',
                aggfunc='median'  # Mediana si hay múltiples períodos
            )

            # Asegurar que tenemos las 4 asignaturas en el orden correcto
            for asig in asignaturas_filtradas:
                if asig not in pivot_vistas.columns:
                    pivot_vistas[asig] = np.nan

            pivot_vistas = pivot_vistas[asignaturas_filtradas]
            pivot_vistas = pivot_vistas.sort_index()

            # Crear figura para vistas
            fig, ax = plt.subplots(figsize=(14, 10))

            sns.heatmap(
                pivot_vistas,
                annot=True,
                fmt='.1f',
                cmap='RdYlGn',
                vmin=0,
                vmax=100,
                cbar_kws={'label': 'Mediana % Módulos Vistos', 'shrink': 0.8},
                linewidths=0.5,
                linecolor='gray',
                ax=ax,
                square=True
            )

            ax.set_title(f'Mediana de % de Módulos Vistos en Moodle\nAsignatura por Grado - Sede: {sede}',
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Asignatura', fontsize=14, fontweight='bold')
            ax.set_ylabel('Grado', fontsize=14, fontweight='bold')

            ax.set_xticklabels([self.get_asignatura_name(int(x)) for x in pivot_vistas.columns], rotation=45, ha='right')
            ax.set_yticklabels([f'Grado {int(y)}' for y in pivot_vistas.index], rotation=0)

            plt.tight_layout(rect=[0, 0.03, 1, 1])

            sede_safe = sede.replace(' ', '_').replace('/', '_')
            output_path_vistas = f'{self.results_path}/heatmap_vistas_{sede_safe}.png'
            plt.savefig(output_path_vistas, dpi=300, bbox_inches='tight')
            plt.close()

            output_paths.append(output_path_vistas)
            self.logger.info(f"✅ Mapa de vistas guardado: {output_path_vistas}")

            # === MAPA 2: % DE PARTICIPACIÓN EN ACTIVIDADES ===
            pivot_participacion = sede_data.pivot_table(
                index='id_grado',
                columns='id_asignatura',
                values='mediana_participacion',
                aggfunc='median'  # Mediana si hay múltiples períodos
            )

            # Asegurar que tenemos las 4 asignaturas en el orden correcto
            for asig in asignaturas_filtradas:
                if asig not in pivot_participacion.columns:
                    pivot_participacion[asig] = np.nan

            pivot_participacion = pivot_participacion[asignaturas_filtradas]
            pivot_participacion = pivot_participacion.sort_index()

            # Crear figura para participación
            fig, ax = plt.subplots(figsize=(14, 10))

            sns.heatmap(
                pivot_participacion,
                annot=True,
                fmt='.1f',
                cmap='RdYlGn',
                vmin=0,
                vmax=100,
                cbar_kws={'label': 'Mediana % Módulos con Participación', 'shrink': 0.8},
                linewidths=0.5,
                linecolor='gray',
                ax=ax,
                square=True
            )

            ax.set_title(f'Mediana de % de Módulos con Participación Activa en Moodle\nAsignatura por Grado - Sede: {sede}',
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Asignatura', fontsize=14, fontweight='bold')
            ax.set_ylabel('Grado', fontsize=14, fontweight='bold')

            ax.set_xticklabels([self.get_asignatura_name(int(x)) for x in pivot_participacion.columns], rotation=45, ha='right')
            ax.set_yticklabels([f'Grado {int(y)}' for y in pivot_participacion.index], rotation=0)

            plt.tight_layout(rect=[0, 0.03, 1, 1])

            output_path_participacion = f'{self.results_path}/heatmap_participacion_{sede_safe}.png'
            plt.savefig(output_path_participacion, dpi=300, bbox_inches='tight')
            plt.close()

            output_paths.append(output_path_participacion)
            self.logger.info(f"✅ Mapa de participación guardado: {output_path_participacion}")

        return output_paths

    def plot_participation_diagnostic(self):
        """Genera gráficas de diagnóstico: Embudo + Pérdida de estudiantes por asignatura."""
        self.logger.info("Generando gráficas de diagnóstico de participación...")

        course_participation = self.results['course_participation']

        # Filtrar solo asignaturas 1, 2, 3, 4
        asignaturas_filtradas = [1, 2, 3, 4]

        # Agregar id_grado
        grado_mapping = self.student_interactions_df[['id_asignatura', 'id_grado', 'sede']].drop_duplicates()
        course_participation_with_grade = course_participation.merge(
            grado_mapping,
            on=['id_asignatura', 'sede'],
            how='left'
        )

        # Filtrar por asignaturas 1-4
        course_participation_filtered = course_participation_with_grade[
            course_participation_with_grade['id_asignatura'].isin(asignaturas_filtradas)
        ]

        # Obtener sedes únicas
        sedes = sorted(course_participation_filtered['sede'].unique())

        output_paths = []

        for sede in sedes:
            self.logger.info(f"Generando diagnóstico para sede: {sede}")

            # Filtrar datos de la sede
            sede_data = course_participation_filtered[
                course_participation_filtered['sede'] == sede
            ].copy()

            # Calcular métricas detalladas por asignatura
            diagnostico_data = []

            for asignatura in asignaturas_filtradas:
                # Obtener todos los estudiantes de esta asignatura en esta sede
                estudiantes_asig = self.student_interactions_df[
                    (self.student_interactions_df['id_asignatura'] == asignatura) &
                    (self.student_interactions_df['sede'] == sede)
                ].copy()

                if len(estudiantes_asig) == 0:
                    continue

                total_inscritos = len(estudiantes_asig)

                # Embudo de participación
                con_acceso = len(estudiantes_asig[estudiantes_asig['modules_viewed'] > 0])
                participaron = len(estudiantes_asig[estudiantes_asig['modules_participated'] > 0])
                completaron = len(estudiantes_asig[estudiantes_asig['has_viewed_all_modules'] == 1])

                # Segmentación para pérdida de estudiantes
                sin_acceso = len(estudiantes_asig[estudiantes_asig['modules_viewed'] == 0])
                acceso_minimo = len(estudiantes_asig[
                    (estudiantes_asig['percent_modules_viewed'] > 0) & 
                    (estudiantes_asig['percent_modules_viewed'] < 0.3)
                ])
                parcialmente_activos = len(estudiantes_asig[
                    (estudiantes_asig['percent_modules_viewed'] >= 0.3) & 
                    (estudiantes_asig['percent_modules_viewed'] < 0.7)
                ])
                muy_activos = len(estudiantes_asig[estudiantes_asig['percent_modules_viewed'] >= 0.7])

                diagnostico_data.append({
                    'asignatura': asignatura,
                    'total_inscritos': total_inscritos,
                    'con_acceso': con_acceso,
                    'participaron': participaron,
                    'completaron': completaron,
                    'sin_acceso': sin_acceso,
                    'acceso_minimo': acceso_minimo,
                    'parcialmente_activos': parcialmente_activos,
                    'muy_activos': muy_activos
                })

            if not diagnostico_data:
                continue

            # === GRÁFICA 1: EMBUDO DE PARTICIPACIÓN ===
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

            # Preparar datos del embudo
            n_asignaturas = len(diagnostico_data)
            bar_width = 0.15
            x_pos = np.arange(n_asignaturas)

            colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']

            for idx, asig_data in enumerate(diagnostico_data):
                total = asig_data['total_inscritos']

                # Datos del embudo (porcentajes) - solo 4 etapas
                embudo = [
                    100,  # Inscritos
                    (asig_data['con_acceso'] / total * 100) if total > 0 else 0,
                    (asig_data['participaron'] / total * 100) if total > 0 else 0,
                    (asig_data['completaron'] / total * 100) if total > 0 else 0
                ]

                etapas = ['Inscritos', 'Accedieron', 'Participaron', 'Completaron']
                y_pos = np.arange(len(etapas))

                bars = ax1.barh(y_pos + idx * bar_width, embudo, bar_width,
                               label=self.get_asignatura_name(asig_data["asignatura"]),
                               color=colors[idx % len(colors)], alpha=0.8)

                # Agregar valores en las barras
                for bar, value in zip(bars, embudo):
                    width = bar.get_width()
                    if width > 5:  # Solo mostrar si hay espacio
                        ax1.text(width - 3, bar.get_y() + bar.get_height()/2, 
                                f'{value:.0f}%', ha='right', va='center', 
                                fontweight='bold', fontsize=9, color='white')

            ax1.set_yticks(y_pos + bar_width * (n_asignaturas - 1) / 2)
            ax1.set_yticklabels(etapas)
            ax1.set_xlabel('Porcentaje de Estudiantes', fontsize=12, fontweight='bold')
            ax1.set_title(f'Embudo de Participación en Moodle\nSede: {sede}', 
                         fontsize=14, fontweight='bold', pad=15)
            ax1.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=9, framealpha=0.95)
            ax1.set_xlim(0, 105)
            ax1.grid(True, alpha=0.3, axis='x')

            # === GRÁFICA 2: PÉRDIDA DE ESTUDIANTES (BARRAS APILADAS) ===

            categorias = ['Sin acceso\n(0%)', 'Acceso mínimo\n(1-30%)', 
                         'Parcialmente\nactivos\n(30-70%)', 'Muy activos\n(>70%)']

            sin_acceso_vals = [d['sin_acceso'] for d in diagnostico_data]
            acceso_minimo_vals = [d['acceso_minimo'] for d in diagnostico_data]
            parciales_vals = [d['parcialmente_activos'] for d in diagnostico_data]
            muy_activos_vals = [d['muy_activos'] for d in diagnostico_data]

            asignaturas_labels = [self.get_asignatura_name(d['asignatura']) for d in diagnostico_data]
            x_pos2 = np.arange(len(asignaturas_labels))

            # Crear barras apiladas
            p1 = ax2.bar(x_pos2, sin_acceso_vals, label='Sin acceso (0%)', 
                        color='#e74c3c', alpha=0.9)
            p2 = ax2.bar(x_pos2, acceso_minimo_vals, bottom=sin_acceso_vals,
                        label='Acceso mínimo (1-30%)', color='#f39c12', alpha=0.9)
            p3 = ax2.bar(x_pos2, parciales_vals, 
                        bottom=np.array(sin_acceso_vals) + np.array(acceso_minimo_vals),
                        label='Parcialmente activos (30-70%)', color='#f1c40f', alpha=0.9)
            p4 = ax2.bar(x_pos2, muy_activos_vals,
                        bottom=np.array(sin_acceso_vals) + np.array(acceso_minimo_vals) + np.array(parciales_vals),
                        label='Muy activos (>70%)', color='#2ecc71', alpha=0.9)

            # Agregar números en las secciones
            for i, (d, p_sin, p_min, p_par, p_act) in enumerate(zip(
                diagnostico_data, p1, p2, p3, p4)):

                total = d['total_inscritos']

                # Sin acceso
                if d['sin_acceso'] > 0:
                    height = p_sin.get_height()
                    if height > total * 0.05:
                        ax2.text(p_sin.get_x() + p_sin.get_width()/2., height/2,
                                f"{d['sin_acceso']}\n({d['sin_acceso']/total*100:.0f}%)",
                                ha='center', va='center', fontweight='bold', fontsize=9)

                # Acceso mínimo
                if d['acceso_minimo'] > 0:
                    y_base = d['sin_acceso']
                    height = p_min.get_height()
                    if height > total * 0.05:
                        ax2.text(p_min.get_x() + p_min.get_width()/2., y_base + height/2,
                                f"{d['acceso_minimo']}\n({d['acceso_minimo']/total*100:.0f}%)",
                                ha='center', va='center', fontweight='bold', fontsize=9)

                # Parcialmente activos
                if d['parcialmente_activos'] > 0:
                    y_base = d['sin_acceso'] + d['acceso_minimo']
                    height = p_par.get_height()
                    if height > total * 0.05:
                        ax2.text(p_par.get_x() + p_par.get_width()/2., y_base + height/2,
                                f"{d['parcialmente_activos']}\n({d['parcialmente_activos']/total*100:.0f}%)",
                                ha='center', va='center', fontweight='bold', fontsize=9)

                # Muy activos
                if d['muy_activos'] > 0:
                    y_base = d['sin_acceso'] + d['acceso_minimo'] + d['parcialmente_activos']
                    height = p_act.get_height()
                    if height > total * 0.05:
                        ax2.text(p_act.get_x() + p_act.get_width()/2., y_base + height/2,
                                f"{d['muy_activos']}\n({d['muy_activos']/total*100:.0f}%)",
                                ha='center', va='center', fontweight='bold', fontsize=9, color='white')

            ax2.set_xticks(x_pos2)
            ax2.set_xticklabels(asignaturas_labels, rotation=20, ha='right', fontsize=10)
            ax2.set_ylabel('Número de Estudiantes', fontsize=12, fontweight='bold')
            ax2.set_title(f'Distribución de Estudiantes por Nivel de Actividad\nSede: {sede}',
                         fontsize=14, fontweight='bold', pad=15)
            ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), fontsize=9,
                      framealpha=0.95, ncol=2)
            ax2.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)

            # Guardar
            sede_safe = sede.replace(' ', '_').replace('/', '_')
            output_path = f'{self.results_path}/diagnostic_participacion_{sede_safe}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            output_paths.append(output_path)
            self.logger.info(f"✅ Diagnóstico guardado: {output_path}")

        return output_paths

    def _create_heatmap(self, sede_data, column, title, cbar_label, fmt='.1f', 
                       cmap='RdYlGn', vmin=0, vmax=100, ax=None):
        """Método auxiliar para crear un heatmap reutilizable."""
        asignaturas_filtradas = [1, 2, 3, 4]

        # Crear pivot table
        pivot = sede_data.pivot_table(
            index='id_grado',
            columns='id_asignatura',
            values=column,
            aggfunc='median'
        )

        # Asegurar que tenemos las 4 asignaturas
        for asig in asignaturas_filtradas:
            if asig not in pivot.columns:
                pivot[asig] = np.nan

        pivot = pivot[asignaturas_filtradas].sort_index()

        # Configurar heatmap
        heatmap_kwargs = {
            'annot': True,
            'fmt': fmt,
            'cmap': cmap,
            'cbar_kws': {'label': cbar_label},
            'linewidths': 0.5,
            'linecolor': 'gray',
            'ax': ax,
            'square': True
        }

        # Solo agregar vmin/vmax si están definidos
        if vmin is not None and vmax is not None:
            heatmap_kwargs['vmin'] = vmin
            heatmap_kwargs['vmax'] = vmax

        sns.heatmap(pivot, **heatmap_kwargs)

        # Configurar ejes
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Asignatura', fontsize=10, fontweight='bold')
        ax.set_ylabel('Grado', fontsize=10, fontweight='bold')
        ax.set_xticklabels([self.get_asignatura_name(int(x)) for x in pivot.columns], 
                          rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels([f'Grado {int(y)}' for y in pivot.index], 
                          rotation=0, fontsize=9)

        return pivot

    def plot_sequence_patterns_heatmaps(self):
        """Genera heatmaps de patrones de secuencia por sede."""
        self.logger.info("Generando heatmaps de patrones de secuencia...")

        sequence_metrics = self.results['sequence_metrics']
        sedes = sorted(sequence_metrics['sede'].unique())
        output_paths = []

        # Configuración de los 3 heatmaps
        heatmap_configs = [
            {
                'column': 'sequence_match_ratio',
                'title': 'Adherencia a la Secuencia Ideal\n(Sequence Match Ratio)',
                'cbar_label': '% Adherencia',
                'cmap': 'RdYlGn',
                'fmt': '.1f',
                'vmin': 0,
                'vmax': 100
            },
            {
                'column': 'levenshtein_normalized',
                'title': 'Distancia de la Secuencia Ideal\n(Levenshtein Normalizado)',
                'cbar_label': '% Distancia',
                'cmap': 'RdYlGn_r',  # Invertido
                'fmt': '.1f',
                'vmin': 0,
                'vmax': 100
            },
            {
                'column': 'missing_activities',
                'title': 'Actividades Faltantes\n(Missing Activities)',
                'cbar_label': 'N° Actividades',
                'cmap': 'YlOrRd',
                'fmt': '.0f',
                'vmin': None,  # Escala automática
                'vmax': None
            }
        ]

        for sede in sedes:
            self.logger.info(f"Generando heatmaps de secuencia para sede: {sede}")

            sede_data = sequence_metrics[sequence_metrics['sede'] == sede].copy()
            if len(sede_data) == 0:
                continue

            # Crear figura con 3 heatmaps
            fig, axes = plt.subplots(1, 3, figsize=(20, 8))

            # Generar cada heatmap usando el método auxiliar
            for ax, config in zip(axes, heatmap_configs):
                self._create_heatmap(sede_data, ax=ax, **config)

            # Título general
            fig.suptitle(f'Patrones de Navegación en Secuencias de Actividades - Sede: {sede}',
                        fontsize=14, fontweight='bold', y=0.98)

            plt.tight_layout(rect=[0, 0.02, 1, 0.96])

            # Guardar
            sede_safe = sede.replace(' ', '_').replace('/', '_')
            output_path = f'{self.results_path}/sequence_patterns_{sede_safe}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            output_paths.append(output_path)
            self.logger.info(f"✅ Heatmaps de secuencia guardados: {output_path}")

        return output_paths

    def plot_grades_without_moodle_access(self):
        """Analiza las calificaciones de estudiantes que NO accedieron a Moodle vs los que SÍ accedieron."""
        self.logger.info("Analizando calificaciones de estudiantes sin acceso a Moodle...")

        # Cargar calificaciones reales
        grades_path = 'data/interim/calificaciones/calificaciones_2024_2025_short.csv'
        self.logger.info(f"Cargando calificaciones desde {grades_path}...")

        try:
            grades_df = pd.read_csv(
                grades_path,
                dtype={'year': 'int16', 'id_grado': 'int8', 'id_asignatura': 'int8', 
                      'period': 'int8', 'sede': 'str'},  # Forzar sede como string
                low_memory=False
            )
            self.logger.info(f"Calificaciones cargadas: {len(grades_df):,} registros")
        except Exception as e:
            self.logger.error(f"Error cargando calificaciones: {e}")
            return []

        # Asegurar que sede sea string en ambos dataframes
        grades_df['sede'] = grades_df['sede'].astype(str)

        # Obtener información de acceso a Moodle desde student_interactions
        # Agregamos modules_viewed por estudiante-asignatura-sede-year-period
        moodle_access = self.student_interactions_df.copy()
        moodle_access['sede'] = moodle_access['sede'].astype(str)

        moodle_access = moodle_access.groupby(
            ['documento_identificación', 'id_asignatura', 'sede', 'year', 'period'],
            as_index=False
        )['modules_viewed'].sum()

        # Hacer merge de calificaciones con acceso a Moodle
        grades_with_moodle = grades_df.merge(
            moodle_access,
            on=['documento_identificación', 'id_asignatura', 'sede', 'year', 'period'],
            how='left'
        )

        # Llenar NaN en modules_viewed con 0 (estudiantes sin acceso a Moodle)
        grades_with_moodle['modules_viewed'] = grades_with_moodle['modules_viewed'].fillna(0)

        self.logger.info(f"Total estudiantes con calificaciones: {len(grades_with_moodle):,}")
        self.logger.info(f"Estudiantes SIN acceso a Moodle: {(grades_with_moodle['modules_viewed'] == 0).sum():,}")
        self.logger.info(f"Estudiantes CON acceso a Moodle: {(grades_with_moodle['modules_viewed'] > 0).sum():,}")

        # Filtrar solo asignaturas 1, 2, 3, 4
        asignaturas_filtradas = [1, 2, 3, 4]

        # Obtener sedes únicas
        sedes = sorted(grades_with_moodle['sede'].unique())

        output_paths = []

        for sede in sedes:
            # Filtrar sedes inválidas (nan, None, etc.)
            if pd.isna(sede) or str(sede).lower() in ['nan', 'none', '']:
                self.logger.warning(f"Saltando sede inválida: {sede}")
                continue

            self.logger.info(f"Analizando calificaciones para sede: {sede}")

            # Datos de la sede
            sede_data = grades_with_moodle[
                (grades_with_moodle['sede'] == sede) &
                (grades_with_moodle['id_asignatura'].isin(asignaturas_filtradas))
            ].copy()

            if len(sede_data) == 0:
                continue

            # Crear figura con subplots para cada asignatura
            fig, axes = plt.subplots(2, 2, figsize=(18, 14))
            axes = axes.flatten()

            for idx, asignatura in enumerate(asignaturas_filtradas):
                ax = axes[idx]

                # Filtrar datos de la asignatura
                asig_data = sede_data[sede_data['id_asignatura'] == asignatura].copy()

                if len(asig_data) == 0:
                    ax.text(0.5, 0.5, f'Sin datos para\nAsignatura {asignatura}', 
                           ha='center', va='center', fontsize=14, transform=ax.transAxes)
                    ax.set_xlim(0, 100)
                    continue

                # Separar estudiantes según acceso a Moodle
                sin_acceso = asig_data[asig_data['modules_viewed'] == 0].copy()
                con_acceso = asig_data[asig_data['modules_viewed'] > 0].copy()

                # Usar nota_final (0-100)
                grade_col = 'nota_final'

                # Filtrar valores válidos (no NaN) y > 0 para evitar registros sin calificación
                sin_acceso_grades = sin_acceso[
                    (sin_acceso[grade_col].notna()) & (sin_acceso[grade_col] > 0)
                ][grade_col]

                con_acceso_grades = con_acceso[
                    (con_acceso[grade_col].notna()) & (con_acceso[grade_col] > 0)
                ][grade_col]

                # Log para debug
                self.logger.info(f"  {self.get_asignatura_name(asignatura)}: "
                               f"Sin acceso con notas={len(sin_acceso_grades)}, "
                               f"Con acceso con notas={len(con_acceso_grades)}, "
                               f"Sin acceso total={len(sin_acceso)}, "
                               f"Con acceso total={len(con_acceso)}")

                # Crear histogramas superpuestos (escala 0-100)
                bins = np.linspace(0, 100, 21)  # Bins de 5 puntos en escala 0-100

                if len(sin_acceso_grades) > 0:
                    ax.hist(sin_acceso_grades, bins=bins, alpha=0.6, 
                           label=f'Sin acceso Moodle (n={len(sin_acceso_grades)})',
                           color='#e74c3c', edgecolor='black', linewidth=0.5)

                if len(con_acceso_grades) > 0:
                    ax.hist(con_acceso_grades, bins=bins, alpha=0.6,
                           label=f'Con acceso Moodle (n={len(con_acceso_grades)})',
                           color='#2ecc71', edgecolor='black', linewidth=0.5)

                # Línea vertical para el umbral de aprobación (60)
                ax.axvline(60, color='gray', linestyle=':', linewidth=1.5, 
                          label='Umbral aprobación (60)', alpha=0.7)

                # Líneas verticales para las medianas
                if len(sin_acceso_grades) > 0:
                    median_sin = sin_acceso_grades.median()
                    ax.axvline(median_sin, color='#c0392b', linestyle='--', 
                              linewidth=2, label=f'Mediana sin acceso: {median_sin:.1f}')

                if len(con_acceso_grades) > 0:
                    median_con = con_acceso_grades.median()
                    ax.axvline(median_con, color='#27ae60', linestyle='--', 
                              linewidth=2, label=f'Mediana con acceso: {median_con:.1f}')

                # Configurar gráfica
                ax.set_xlabel('Calificación (0-100)', fontsize=11, fontweight='bold')
                ax.set_ylabel('Número de Estudiantes', fontsize=11, fontweight='bold')
                ax.set_title(self.get_asignatura_name(asignatura), fontsize=13, fontweight='bold')
                ax.legend(fontsize=7, loc='upper left', framealpha=0.9, bbox_to_anchor=(0.01, 0.99))
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_xlim(0, 100)

                # Agregar estadísticas en texto
                stats_text = []
                if len(sin_acceso_grades) > 0:
                    stats_text.append(f"Sin acceso: Media={sin_acceso_grades.mean():.1f}, "
                                    f"Aprobados={(sin_acceso_grades >= 60).sum()} "
                                    f"({(sin_acceso_grades >= 60).sum()/len(sin_acceso_grades)*100:.1f}%)")
                if len(con_acceso_grades) > 0:
                    stats_text.append(f"Con acceso: Media={con_acceso_grades.mean():.1f}, "
                                    f"Aprobados={(con_acceso_grades >= 60).sum()} "
                                    f"({(con_acceso_grades >= 60).sum()/len(con_acceso_grades)*100:.1f}%)")

                if stats_text:
                    ax.text(0.37, 0.80, '\n'.join(stats_text),
                           transform=ax.transAxes, fontsize=8,
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Título general
            fig.suptitle(f'Distribución de Calificaciones: CON vs SIN Acceso a Moodle\nSede: {sede} | Nota: n = registros estudiante-período (un estudiante puede aparecer en múltiples períodos)',
                        fontsize=15, fontweight='bold', y=0.995)

            plt.tight_layout(rect=[0, 0, 1, 0.99])

            # Guardar
            sede_safe = sede.replace(' ', '_').replace('/', '_')
            output_path = f'{self.results_path}/grades_without_moodle_{sede_safe}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            output_paths.append(output_path)
            self.logger.info(f"✅ Análisis de calificaciones sin Moodle guardado: {output_path}")

            # Log resumen estadístico
            for asignatura in asignaturas_filtradas:
                asig_data = sede_data[sede_data['id_asignatura'] == asignatura]
                if len(asig_data) > 0:
                    sin_acceso = asig_data[asig_data['modules_viewed'] == 0]
                    con_acceso = asig_data[asig_data['modules_viewed'] > 0]

                    if len(sin_acceso) > 0:
                        sin_grades = sin_acceso['nota_final'].dropna()
                        if len(sin_grades) > 0:
                            self.logger.info(
                                f"  {self.get_asignatura_name(asignatura)} SIN acceso: {len(sin_grades)} estudiantes, "
                                f"Media={sin_grades.mean():.1f}, "
                                f"Aprobados={(sin_grades >= 60).sum()} ({(sin_grades >= 60).sum()/len(sin_grades)*100:.1f}%)"
                            )

                    if len(con_acceso) > 0:
                        con_grades = con_acceso['nota_final'].dropna()
                        if len(con_grades) > 0:
                            self.logger.info(
                                f"  {self.get_asignatura_name(asignatura)} CON acceso: {len(con_grades)} estudiantes, "
                                f"Media={con_grades.mean():.1f}, "
                                f"Aprobados={(con_grades >= 60).sum()} ({(con_grades >= 60).sum()/len(con_grades)*100:.1f}%)"
                            )

        return output_paths

    def generate_summary_statistics(self):
        """Genera estadísticas resumen del análisis."""
        self.logger.info("Generando estadísticas resumen...")

        # Estadísticas generales
        total_students = self.student_logs_df['documento_identificación'].nunique()
        total_courses = self.student_logs_df['courseid'].nunique()
        total_accesses = len(self.student_logs_df)

        daily_stats = self.results['daily_stats']

        # Escribir resumen
        summary_path = f'{self.results_path}/moodle_behavior_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ANÁLISIS DE COMPORTAMIENTO EN MOODLE\n")
            f.write("=" * 80 + "\n\n")

            f.write("ESTADÍSTICAS GENERALES:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total de estudiantes únicos: {total_students:,}\n")
            f.write(f"Total de cursos: {total_courses:,}\n")
            f.write(f"Total de accesos registrados: {total_accesses:,}\n")
            f.write(f"Días con actividad: {daily_stats['total_dias_actividad']}\n")
            f.write(f"Promedio de accesos por día: {daily_stats['promedio_accesos_dia']:.1f}\n")
            f.write(f"Promedio de estudiantes activos por día: {daily_stats['promedio_estudiantes_dia']:.1f}\n\n")

            # Estadísticas por grado
            if 'grade_behavior' in self.results:
                grade_behavior = self.results['grade_behavior']
                f.write("COMPORTAMIENTO POR GRADO:\n")
                f.write("-" * 40 + "\n")
                f.write(grade_behavior.to_string(index=False))
                f.write("\n\n")

            # Top cursos
            if 'top_courses' in self.results:
                top_courses = self.results['top_courses'].head(5)
                f.write("TOP 5 CURSOS MÁS POPULARES:\n")
                f.write("-" * 40 + "\n")
                for _, row in top_courses.iterrows():
                    # Usar id_asignatura si courseid no está disponible
                    course_id = row.get('courseid', row.get('id_asignatura', 'N/A'))
                    f.write(f"Curso {course_id}: {int(row['total_accesos']):,} accesos, "
                           f"{int(row['estudiantes_unicos'])} estudiantes únicos\n")

        self.logger.info(f"✅ Resumen guardado: {summary_path}")
        return summary_path

    def run_analysis(self):
        """Ejecuta el pipeline completo de análisis."""
        self.logger.info("=" * 60)
        self.logger.info("INICIANDO ANÁLISIS DE COMPORTAMIENTO EN MOODLE")
        self.logger.info("=" * 60)

        try:
            # Crear directorio de resultados
            self.create_results_directory()

            # 1. Cargar datos
            self.load_moodle_data()

            # 2. Preparar datos temporales
            self.prepare_temporal_data()

            # 3. Análisis de accesos diarios
            self.analyze_daily_accesses()

            # 4. Análisis de accesos mensuales por sede
            self.analyze_monthly_accesses_by_sede()

            # 5. Análisis de estudiantes únicos por mes y sede
            self.analyze_unique_students_by_month_sede()

            # 6. Análisis de patrones por hora
            self.analyze_hourly_patterns()

            # 7. Análisis de tasa de participación de cursos
            self.analyze_course_participation_rate()

            # 8. Análisis por grado
            self.analyze_behavior_by_grade()

            # 9. Análisis de patrones de secuencia
            self.analyze_sequence_patterns()

            # 10. Generar gráficos
            self.logger.info("Generando gráficos...")
            self.plot_monthly_accesses_by_sede()
            self.plot_unique_students_by_month_sede()
            self.plot_hourly_patterns()
            self.plot_course_participation_rate()
            self.plot_participation_diagnostic()  # Gráfica de diagnóstico
            self.plot_sequence_patterns_heatmaps()  # Patrones de secuencia
            self.plot_grades_without_moodle_access()  # Calificaciones sin acceso a Moodle

            # 11. Generar resumen
            self.generate_summary_statistics()

            self.logger.info("=" * 60)
            self.logger.info("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
            self.logger.info("=" * 60)

            return self.results

        except Exception as e:
            self.logger.error(f"❌ Error en análisis: {e}")
            raise


def main():
    """Función principal."""
    import argparse

    parser = argparse.ArgumentParser(description='Análisis de comportamiento en Moodle')
    parser.add_argument('--dataset', '-d', type=str, 
                       default='data/interim/moodle',
                       help='Ruta al directorio de datasets de Moodle')
    parser.add_argument('--results', '-r', type=str, 
                       default='moodle_behavior_analysis',
                       help='Nombre del folder para guardar resultados')

    args = parser.parse_args()

    # Crear y ejecutar analizador
    # Usar cualquier archivo del directorio como dataset_path (se ajustará internamente)
    dataset_path = f"{args.dataset}/student_course_interactions.csv"
    analyzer = MoodleBehaviorAnalysis(dataset_path, args.results)

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
