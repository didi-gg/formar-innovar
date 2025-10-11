import pandas as pd
import numpy as np
import os
import sys
import time
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.base_script import BaseScript

class StudentCourseInteractionsProcessor(BaseScript):

    def __init__(self):
        super().__init__()
        self.student_modules = None
        self.modules_featured = None

    def load_data(self):
        """Cargar los datos necesarios"""
        load_start_time = time.time()
        self.logger.info("Cargando datos...")
        
        # Cargar student_modules
        self.logger.info("Cargando student_modules.csv...")
        file_start = time.time()
        self.student_modules = pd.read_csv("data/interim/moodle/student_modules.csv")
        file_time = time.time() - file_start
        self.logger.info(f"student_modules.csv cargado en {file_time:.2f}s - Shape: {self.student_modules.shape}")
        
        # Cargar modules_featured
        self.logger.info("Cargando modules_featured.csv...")
        file_start = time.time()
        self.modules_featured = pd.read_csv("data/interim/moodle/modules_featured.csv")
        file_time = time.time() - file_start
        self.logger.info(f"modules_featured.csv cargado en {file_time:.2f}s - Shape: {self.modules_featured.shape}")

        # Log información de columnas
        self.logger.info(f"Columnas en student_modules: {list(self.student_modules.columns)}")
        self.logger.info(f"Columnas en modules_featured: {list(self.modules_featured.columns)}")

        # Convertir columnas numéricas
        conversion_start = time.time()
        self.logger.info("Convirtiendo columnas numéricas...")
        numeric_cols = ['num_views', 'num_interactions', 'days_from_planned_start', 'days_after_end']
        for col in numeric_cols:
            if col in self.student_modules.columns:
                col_start = time.time()
                original_nulls = self.student_modules[col].isna().sum()
                self.student_modules[col] = pd.to_numeric(self.student_modules[col], errors='coerce').fillna(0)
                new_nulls = self.student_modules[col].isna().sum()
                col_time = time.time() - col_start
                self.logger.info(f"  {col}: {col_time:.3f}s - NULLs originales: {original_nulls}, NULLs nuevos: {new_nulls}")

        # Convertir columnas booleanas
        self.logger.info("Convirtiendo columnas booleanas...")
        bool_cols = ['has_viewed', 'has_participated', 'was_on_time']
        for col in bool_cols:
            if col in self.student_modules.columns:
                col_start = time.time()
                original_unique = len(self.student_modules[col].unique())
                self.student_modules[col] = self.student_modules[col].astype(int)
                col_time = time.time() - col_start
                self.logger.info(f"  {col}: {col_time:.3f}s - Valores únicos originales: {original_unique}")
        
        conversion_time = time.time() - conversion_start
        total_load_time = time.time() - load_start_time
        
        # Log estadísticas de memoria
        memory_usage_mb = (self.student_modules.memory_usage(deep=True).sum() + 
                          self.modules_featured.memory_usage(deep=True).sum()) / 1024 / 1024
        
        self.logger.info(f"=== RESUMEN DE CARGA DE DATOS ===")
        self.logger.info(f"Tiempo total de carga: {total_load_time:.2f}s")
        self.logger.info(f"Tiempo de conversión de tipos: {conversion_time:.2f}s")
        self.logger.info(f"Uso de memoria: {memory_usage_mb:.2f} MB")
        self.logger.info(f"Registros de student_modules: {len(self.student_modules):,}")
        self.logger.info(f"Registros de modules_featured: {len(self.modules_featured):,}")
        
        # Estadísticas de los datos
        unique_students = self.student_modules['documento_identificación'].nunique()
        unique_courses = self.student_modules.groupby(['year', 'id_grado', 'sede', 'id_asignatura', 'period']).ngroups
        self.logger.info(f"Estudiantes únicos: {unique_students:,}")
        self.logger.info(f"Cursos únicos (combinaciones año-grado-sede-asignatura-período): {unique_courses:,}")

    def calculate_basic_engagement_metrics(self, group_data):
        """Calcular métricas básicas de engagement y participación"""
        metrics = {}

        # Métricas básicas
        metrics['total_modules'] = len(group_data)
        metrics['modules_viewed'] = group_data['has_viewed'].sum()
        metrics['modules_participated'] = group_data['has_participated'].sum()

        # Porcentajes
        if metrics['total_modules'] > 0:
            metrics['percent_modules_viewed'] = round(metrics['modules_viewed'] / metrics['total_modules'], 4)
            metrics['percent_modules_participated'] = round(metrics['modules_participated'] / metrics['total_modules'], 4)
        else:
            metrics['percent_modules_viewed'] = 0
            metrics['percent_modules_participated'] = 0

        # Indicadores binarios
        metrics['has_viewed_all_modules'] = 1 if metrics['modules_viewed'] == metrics['total_modules'] else 0
        metrics['has_participated_all_modules'] = 1 if metrics['modules_participated'] == metrics['total_modules'] else 0

        return metrics

    def calculate_activity_metrics(self, group_data):
        """Calcular métricas de actividad"""
        metrics = {}

        # Totales
        metrics['total_views'] = group_data['num_views'].sum()
        metrics['total_interactions'] = group_data['num_interactions'].sum()

        # Promedios
        total_modules = len(group_data)
        if total_modules > 0:
            metrics['avg_views_per_module'] = round(metrics['total_views'] / total_modules, 2)
            metrics['avg_interactions_per_module'] = round(metrics['total_interactions'] / total_modules, 2)
        else:
            metrics['avg_views_per_module'] = 0
            metrics['avg_interactions_per_module'] = 0

        # Medianas
        metrics['median_views_per_module'] = round(group_data['num_views'].median(), 2)
        metrics['median_interactions_per_module'] = round(group_data['num_interactions'].median(), 2)

        # Mínimos
        metrics['min_views_per_module'] = group_data['num_views'].min()
        metrics['min_interactions_per_module'] = group_data['num_interactions'].min()

        # Máximos
        metrics['max_views_in_a_module'] = group_data['num_views'].max()
        metrics['max_interactions_in_a_module'] = group_data['num_interactions'].max()

        # Desviaciones estándar
        metrics['std_views_per_module'] = round(group_data['num_views'].std(), 2) if len(group_data) > 1 else 0
        metrics['std_interactions_per_module'] = round(group_data['num_interactions'].std(), 2) if len(group_data) > 1 else 0

        # Ratio interacción/vista
        if metrics['total_views'] > 0:
            metrics['interaction_to_view_ratio'] = round(metrics['total_interactions'] / metrics['total_views'], 4)
        else:
            metrics['interaction_to_view_ratio'] = 0

        # Logaritmos para normalización
        metrics['log_total_views'] = round(np.log1p(metrics['total_views']), 4)
        metrics['log_total_interactions'] = round(np.log1p(metrics['total_interactions']), 4)

        return metrics

    def calculate_temporal_metrics(self, group_data):
        """Calcular métricas temporales y de disciplina"""
        metrics = {}

        # Métricas de días antes/después
        metrics['avg_days_from_planned_start'] = round(group_data['days_from_planned_start'].mean(), 2)
        metrics['avg_days_after_end'] = round(group_data['days_after_end'].mean(), 2)
        metrics['std_days_from_planned_start'] = round(group_data['days_from_planned_start'].std(), 2) if len(group_data) > 1 else 0
        metrics['std_days_after_end'] = round(group_data['days_after_end'].std(), 2) if len(group_data) > 1 else 0
        metrics['min_days_from_planned_start'] = group_data['days_from_planned_start'].min()
        metrics['min_days_after_end'] = group_data['days_after_end'].min()
        metrics['max_days_after_end'] = group_data['days_after_end'].max()
        metrics['max_days_from_planned_start'] = group_data['days_from_planned_start'].max()
        metrics['median_days_from_planned_start'] = round(group_data['days_from_planned_start'].median(), 2)
        metrics['median_days_after_end'] = round(group_data['days_after_end'].median(), 2)

        # Métricas de puntualidad
        total_modules = len(group_data)
        if total_modules > 0:
            metrics['on_time_rate'] = round(group_data['was_on_time'].mean(), 4)
        else:
            metrics['on_time_rate'] = 0
        metrics['late_rate'] = round(1 - metrics['on_time_rate'], 4)

        # Conteos de acceso temprano y tardío
        metrics['early_access_count'] = (group_data['days_from_planned_start'] > 0).sum()
        metrics['late_access_count'] = (group_data['days_after_end'] > 0).sum()

        return metrics

    def calculate_dispersion_metrics(self, group_data):
        """Calcular métricas de dispersión"""
        metrics = {}
        views = group_data['num_views']

        if len(views) > 0:
            # IQR de vistas
            q75, q25 = np.percentile(views, [75, 25])
            metrics['iqr_views'] = round(q75 - q25, 2)

            # Asimetría (skewness)
            skew_value = stats.skew(views)
            metrics['skew_views'] = round(skew_value, 4) if not np.isnan(skew_value) else 0

            # Curtosis
            kurtosis_value = stats.kurtosis(views)
            metrics['kurtosis_views'] = round(kurtosis_value, 4) if not np.isnan(kurtosis_value) else 0
        else:
            metrics['iqr_views'] = 0
            metrics['skew_views'] = 0
            metrics['kurtosis_views'] = 0
        return metrics

    def calculate_relative_engagement(self, student_data, course_data):
        """Calcular métricas de engagement relativo"""
        metrics = {}

        # Obtener métricas del estudiante
        student_views = student_data['num_views'].sum()
        student_interactions = student_data['num_interactions'].sum()

        # Obtener métricas del curso
        course_views = course_data['num_views']
        course_interactions = course_data['num_interactions']

        # Calcular percentiles
        if len(course_views) > 0:
            # Optimización: calcular groupby una sola vez y reutilizar
            student_grouped = course_data.groupby('documento_identificación').agg({
                'num_views': 'sum',
                'num_interactions': 'sum'
            })
            
            student_views_per_student = student_grouped['num_views']
            student_interactions_per_student = student_grouped['num_interactions']

            # Verificar que hay suficientes datos para estadísticas
            if len(student_views_per_student) > 1:
                metrics['relative_views_percentile'] = round(stats.percentileofscore(student_views_per_student, student_views), 2)
                metrics['relative_interaction_percentile'] = round(stats.percentileofscore(student_interactions_per_student, student_interactions), 2)

                # Z-scores
                views_mean = student_views_per_student.mean()
                views_std = student_views_per_student.std()
                if views_std > 0:
                    metrics['zscore_views'] = round((student_views - views_mean) / views_std, 4)
                else:
                    metrics['zscore_views'] = 0

                interactions_mean = student_interactions_per_student.mean()
                interactions_std = student_interactions_per_student.std()
                if interactions_std > 0:
                    metrics['zscore_interactions'] = round((student_interactions - interactions_mean) / interactions_std, 4)
                else:
                    metrics['zscore_interactions'] = 0
            else:
                # Solo hay un estudiante, no se pueden calcular estadísticas relativas
                metrics['relative_views_percentile'] = 50.0  # Neutral
                metrics['relative_interaction_percentile'] = 50.0  # Neutral
                metrics['zscore_views'] = 0
                metrics['zscore_interactions'] = 0
        else:
            metrics['relative_views_percentile'] = 0
            metrics['relative_interaction_percentile'] = 0
            metrics['zscore_views'] = 0
            metrics['zscore_interactions'] = 0

        return metrics
    
    def calculate_mid_week_engagement(self, student_data, modules_data):
        """Calcular engagement de mitad de semana"""
        try:
            # Verificar si tenemos los datos necesarios
            if 'course_module_id' not in student_data.columns:
                return {'mid_week_engagement': 0}
            
            # Verificar si modules_data tiene las columnas necesarias
            required_cols = ['course_module_id', 'week', 'planned_start_date', 'planned_end_date']
            available_cols = [col for col in required_cols if col in modules_data.columns]
            
            if not available_cols or 'course_module_id' not in available_cols:
                return {'mid_week_engagement': 0}

            # Unir con modules_featured para obtener información de semana (solo columnas disponibles)
            student_modules_with_week = student_data.merge(
                modules_data[available_cols], 
                on='course_module_id', 
                how='left'
            )

            # Convertir fechas si están disponibles
            if 'first_view' in student_modules_with_week.columns and 'planned_start_date' in student_modules_with_week.columns:
                # Evitar conversiones costosas si ya están en datetime
                if not pd.api.types.is_datetime64_any_dtype(student_modules_with_week['first_view']):
                    student_modules_with_week['first_view'] = pd.to_datetime(student_modules_with_week['first_view'], errors='coerce')
                if not pd.api.types.is_datetime64_any_dtype(student_modules_with_week['planned_start_date']):
                    student_modules_with_week['planned_start_date'] = pd.to_datetime(student_modules_with_week['planned_start_date'], errors='coerce')

                # Calcular día de la semana del primer acceso (0=Lunes, 6=Domingo)
                valid_first_view = student_modules_with_week['first_view'].notna()
                student_modules_with_week.loc[valid_first_view, 'access_weekday'] = student_modules_with_week.loc[valid_first_view, 'first_view'].dt.dayofweek

                # Considerar lunes a miércoles como mitad de semana (0, 1, 2)
                mid_week_mask = (
                    student_modules_with_week['access_weekday'].isin([0, 1, 2]) & 
                    (student_modules_with_week['has_viewed'] == 1)
                )
                mid_week_accessed = student_modules_with_week[mid_week_mask]

                total_viewed = student_modules_with_week[student_modules_with_week['has_viewed'] == 1]

                if len(total_viewed) > 0:
                    mid_week_engagement = len(mid_week_accessed) / len(total_viewed)
                else:
                    mid_week_engagement = 0
            else:
                mid_week_engagement = 0

            return {'mid_week_engagement': round(mid_week_engagement, 4)}
        except Exception as e:
            # Log error silencioso para debugging
            return {'mid_week_engagement': 0}

    def calculate_grade_metrics(self, group_data):
        """Calcular métricas relacionadas con calificaciones"""
        metrics = {}
        # Verificar si existe la columna finalgrade
        if 'finalgrade' in group_data.columns:
            # Filtrar solo calificaciones válidas (no nulas y > 0)
            valid_grades = group_data[
                (group_data['finalgrade'].notna()) & 
                (group_data['finalgrade'] > 0)
            ]['finalgrade']
            if len(valid_grades) > 0:
                # Máxima calificación obtenida en la asignatura
                metrics['max_activity_grade'] = round(valid_grades.max(), 2)
                
                # Métricas adicionales de calificaciones
                metrics['avg_activity_grade'] = round(valid_grades.mean(), 2)
                metrics['min_activity_grade'] = round(valid_grades.min(), 2)
                metrics['std_activity_grade'] = round(valid_grades.std(), 2) if len(valid_grades) > 1 else 0
                
                # Conteo de actividades calificadas
                metrics['graded_activities_count'] = len(valid_grades)
                
                # Porcentaje de actividades calificadas
                total_activities = len(group_data)
                if total_activities > 0:
                    metrics['percent_graded_activities'] = round(len(valid_grades) / total_activities, 4)
                else:
                    metrics['percent_graded_activities'] = 0
            else:
                # No hay calificaciones válidas
                metrics['max_activity_grade'] = 0
                metrics['avg_activity_grade'] = 0
                metrics['min_activity_grade'] = 0
                metrics['std_activity_grade'] = 0
                metrics['graded_activities_count'] = 0
                metrics['percent_graded_activities'] = 0
        else:
            # No existe la columna finalgrade
            metrics['max_activity_grade'] = 0
            metrics['avg_activity_grade'] = 0
            metrics['min_activity_grade'] = 0
            metrics['std_activity_grade'] = 0
            metrics['graded_activities_count'] = 0
            metrics['percent_graded_activities'] = 0
        
        return metrics

    def calculate_time_metrics(self, group_data):
        """
        Calcular métricas de tiempo total que el estudiante pasa en el curso.
        Args:
            group_data (pd.DataFrame): Datos del grupo estudiante-curso
        Returns:
            dict: Métricas de tiempo calculadas
        """
        metrics = {}
        # Verificar si existe la columna total_time_minutes
        if 'total_time_minutes' in group_data.columns:
            # Filtrar solo registros con tiempo calculado
            valid_times = group_data[group_data['total_time_minutes'].notna()]['total_time_minutes']

            if len(valid_times) > 0:
                # Tiempo total en el curso (suma de todos los módulos)
                metrics['total_course_time_minutes'] = round(valid_times.sum(), 2)

                # Tiempo promedio por módulo accedido
                metrics['avg_time_per_module'] = round(valid_times.mean(), 2)

                # Tiempo mediano por módulo
                metrics['median_time_per_module'] = round(valid_times.median(), 2)

                # Tiempo mínimo y máximo por módulo
                metrics['min_time_per_module'] = round(valid_times.min(), 2)
                metrics['max_time_per_module'] = round(valid_times.max(), 2)

                # Desviación estándar del tiempo por módulo
                metrics['std_time_per_module'] = round(valid_times.std(), 2) if len(valid_times) > 1 else 0

                # Número de módulos con tiempo calculado
                metrics['modules_with_time'] = len(valid_times)

                # Porcentaje de módulos con tiempo calculado
                total_modules = len(group_data)
                if total_modules > 0:
                    metrics['percent_modules_with_time'] = round(len(valid_times) / total_modules, 4)
                else:
                    metrics['percent_modules_with_time'] = 0

                # Métricas de distribución de tiempo
                if len(valid_times) > 0:
                    # Tiempo total en horas
                    metrics['total_course_time_hours'] = round(metrics['total_course_time_minutes'] / 60, 2)

                    # Clasificación de engagement por tiempo
                    if metrics['total_course_time_minutes'] < 30:
                        metrics['time_engagement_level'] = 'bajo'
                    elif metrics['total_course_time_minutes'] < 120:
                        metrics['time_engagement_level'] = 'moderado'
                    elif metrics['total_course_time_minutes'] < 300:
                        metrics['time_engagement_level'] = 'alto'
                    else:
                        metrics['time_engagement_level'] = 'muy_alto'
                else:
                    metrics['total_course_time_hours'] = 0
                    metrics['time_engagement_level'] = 'sin_datos'
            else:
                # No hay datos de tiempo válidos
                metrics['total_course_time_minutes'] = 0
                metrics['avg_time_per_module'] = 0
                metrics['median_time_per_module'] = 0
                metrics['min_time_per_module'] = 0
                metrics['max_time_per_module'] = 0
                metrics['std_time_per_module'] = 0
                metrics['modules_with_time'] = 0
                metrics['percent_modules_with_time'] = 0
                metrics['total_course_time_hours'] = 0
                metrics['time_engagement_level'] = 'sin_datos'
        else:
            # No existe la columna total_time_minutes
            metrics['total_course_time_minutes'] = 0
            metrics['avg_time_per_module'] = 0
            metrics['median_time_per_module'] = 0
            metrics['min_time_per_module'] = 0
            metrics['max_time_per_module'] = 0
            metrics['std_time_per_module'] = 0
            metrics['modules_with_time'] = 0
            metrics['percent_modules_with_time'] = 0
            metrics['total_course_time_hours'] = 0
            metrics['time_engagement_level'] = 'sin_datos'
        return metrics
    
    def process_student_engagement_metrics(self):
        """Procesar todas las métricas de engagement para cada estudiante"""
        start_time = time.time()
        self.logger.info("Iniciando cálculo de métricas de engagement...")

        # Agrupar por la clave especificada
        groupby_cols = ['documento_identificación', 'year', 'id_grado', 'sede', 'id_asignatura', 'period']
        
        # Obtener información sobre los grupos
        grouped = self.student_modules.groupby(groupby_cols)
        total_groups = len(grouped)
        self.logger.info(f"Total de grupos estudiante-curso a procesar: {total_groups}")

        results = []
        processed_groups = 0
        checkpoint_interval = max(1, total_groups // 20)  # Log cada 5% del progreso
        
        # Inicializar contadores de tiempo para cada sección
        time_metrics = {
            'basic': 0,
            'activity': 0,
            'temporal': 0,
            'dispersion': 0,
            'relative': 0,
            'mid_week': 0,
            'grade': 0,
            'time': 0,
            'filtering': 0
        }

        # Procesar cada grupo de estudiante-curso
        for group_key, group_data in grouped:
            group_start_time = time.time()
            try:
                # Crear diccionario con las claves del grupo
                result = dict(zip(groupby_cols, group_key))
                processed_groups += 1

                # Log de progreso cada checkpoint_interval grupos
                if processed_groups % checkpoint_interval == 0 or processed_groups == 1:
                    elapsed_time = time.time() - start_time
                    progress_percent = (processed_groups / total_groups) * 100
                    avg_time_per_group = elapsed_time / processed_groups
                    estimated_remaining = (total_groups - processed_groups) * avg_time_per_group
                    
                    self.logger.info(f"Progreso: {processed_groups}/{total_groups} ({progress_percent:.1f}%) - "
                                   f"Tiempo transcurrido: {elapsed_time:.1f}s - "
                                   f"Tiempo estimado restante: {estimated_remaining:.1f}s - "
                                   f"Promedio por grupo: {avg_time_per_group:.3f}s")

                # Calcular métricas básicas
                metric_start = time.time()
                basic_metrics = self.calculate_basic_engagement_metrics(group_data)
                result.update(basic_metrics)
                time_metrics['basic'] += time.time() - metric_start

                # Calcular métricas de actividad
                metric_start = time.time()
                activity_metrics = self.calculate_activity_metrics(group_data)
                result.update(activity_metrics)
                time_metrics['activity'] += time.time() - metric_start

                # Calcular métricas temporales
                metric_start = time.time()
                temporal_metrics = self.calculate_temporal_metrics(group_data)
                result.update(temporal_metrics)
                time_metrics['temporal'] += time.time() - metric_start

                # Calcular métricas de dispersión
                metric_start = time.time()
                dispersion_metrics = self.calculate_dispersion_metrics(group_data)
                result.update(dispersion_metrics)
                time_metrics['dispersion'] += time.time() - metric_start

                # Calcular métricas de engagement relativo
                metric_start = time.time()
                # Filtrar datos del curso para comparación
                course_filter = (
                    (self.student_modules['year'] == group_key[1]) &
                    (self.student_modules['id_grado'] == group_key[2]) &
                    (self.student_modules['sede'] == group_key[3]) &
                    (self.student_modules['id_asignatura'] == group_key[4]) &
                    (self.student_modules['period'] == group_key[5])
                )
                course_data = self.student_modules[course_filter]
                time_metrics['filtering'] += time.time() - metric_start

                metric_start = time.time()
                relative_metrics = self.calculate_relative_engagement(group_data, course_data)
                result.update(relative_metrics)
                time_metrics['relative'] += time.time() - metric_start

                # Calcular engagement de mitad de semana
                metric_start = time.time()
                mid_week_metrics = self.calculate_mid_week_engagement(group_data, self.modules_featured)
                result.update(mid_week_metrics)
                time_metrics['mid_week'] += time.time() - metric_start

                # Calcular métricas de calificaciones
                metric_start = time.time()
                grade_metrics = self.calculate_grade_metrics(group_data)
                result.update(grade_metrics)
                time_metrics['grade'] += time.time() - metric_start

                # Calcular métricas de tiempo
                metric_start = time.time()
                time_metrics_result = self.calculate_time_metrics(group_data)
                result.update(time_metrics_result)
                time_metrics['time'] += time.time() - metric_start
                
                results.append(result)

                # Log detallado para grupos que toman mucho tiempo
                group_time = time.time() - group_start_time
                if group_time > 1.0:  # Log si un grupo toma más de 1 segundo
                    self.logger.warning(f"Grupo lento detectado: {group_key} - "
                                      f"Tiempo: {group_time:.3f}s - "
                                      f"Tamaño: {len(group_data)} registros")

            except Exception as e:
                self.logger.error(f"Error procesando grupo {group_key}: {str(e)}")
                continue

        # Log final con estadísticas de tiempo
        total_time = time.time() - start_time
        self.logger.info(f"Procesamiento completado en {total_time:.2f} segundos")
        self.logger.info(f"Tiempo promedio por grupo: {total_time/total_groups:.4f} segundos")
        
        # Log de tiempo por cada tipo de métrica
        self.logger.info("=== TIEMPO POR TIPO DE MÉTRICA ===")
        for metric_type, metric_time in time_metrics.items():
            percentage = (metric_time / total_time) * 100 if total_time > 0 else 0
            self.logger.info(f"{metric_type.capitalize()}: {metric_time:.2f}s ({percentage:.1f}%)")

        # Convertir a DataFrame
        self.logger.info("Convirtiendo resultados a DataFrame...")
        df_start = time.time()
        engagement_df = pd.DataFrame(results)
        df_time = time.time() - df_start
        
        self.logger.info(f"DataFrame creado en {df_time:.2f} segundos")
        self.logger.info(f"Métricas calculadas para {len(engagement_df)} grupos estudiante-curso")
        return engagement_df
    
    def run(self):
        """Ejecutar el proceso completo"""
        try:
            # Cargar datos
            self.load_data()

            # Calcular métricas
            engagement_metrics = self.process_student_engagement_metrics()

            # Guardar resultados
            output_path = "data/interim/moodle/student_course_interactions.csv"
            self.save_to_csv(engagement_metrics, output_path)

            self.logger.info(f"Métricas de engagement guardadas en: {output_path}")
            self.logger.info(f"Total de registros: {len(engagement_metrics)}")

            # Mostrar estadísticas básicas
            self.logger.info("\n=== ESTADÍSTICAS BÁSICAS ===")
            self.logger.info(f"Promedio de módulos totales: {engagement_metrics['total_modules'].mean():.2f}")
            self.logger.info(f"Promedio de módulos vistos: {engagement_metrics['modules_viewed'].mean():.2f}")
            self.logger.info(f"Promedio de módulos participados: {engagement_metrics['modules_participated'].mean():.2f}")
            self.logger.info(f"Promedio de vistas totales: {engagement_metrics['total_views'].mean():.2f}")
            self.logger.info(f"Promedio de interacciones totales: {engagement_metrics['total_interactions'].mean():.2f}")
            self.logger.info(f"Promedio de puntualidad: {engagement_metrics['on_time_rate'].mean():.4f}")

            # Estadísticas de calificaciones
            if 'max_activity_grade' in engagement_metrics.columns:
                self.logger.info(f"Promedio de calificación máxima: {engagement_metrics['max_activity_grade'].mean():.2f}")
                self.logger.info(f"Promedio de actividades calificadas: {engagement_metrics['graded_activities_count'].mean():.2f}")
                self.logger.info(f"Porcentaje promedio de actividades calificadas: {engagement_metrics['percent_graded_activities'].mean():.4f}")

            # Estadísticas de tiempo
            if 'total_course_time_minutes' in engagement_metrics.columns:
                self.logger.info(f"\n=== ESTADÍSTICAS DE TIEMPO ===")
                self.logger.info(f"Promedio de tiempo total por curso: {engagement_metrics['total_course_time_minutes'].mean():.2f} minutos ({engagement_metrics['total_course_time_minutes'].mean()/60:.2f} horas)")
                self.logger.info(f"Mediana de tiempo total por curso: {engagement_metrics['total_course_time_minutes'].median():.2f} minutos")
                self.logger.info(f"Promedio de tiempo por módulo: {engagement_metrics['avg_time_per_module'].mean():.2f} minutos")
                self.logger.info(f"Promedio de módulos con tiempo: {engagement_metrics['modules_with_time'].mean():.2f}")
                self.logger.info(f"Porcentaje promedio de módulos con tiempo: {engagement_metrics['percent_modules_with_time'].mean():.4f}")
                # Distribución por nivel de engagement
                if 'time_engagement_level' in engagement_metrics.columns:
                    engagement_dist = engagement_metrics['time_engagement_level'].value_counts()
                    self.logger.info(f"Distribución por nivel de engagement temporal:")
                    for level, count in engagement_dist.items():
                        percentage = (count / len(engagement_metrics)) * 100
                        self.logger.info(f"  - {level}: {count} ({percentage:.1f}%)")

            return engagement_metrics

        except Exception as e:
            self.logger.error(f"Error en el proceso: {str(e)}")
            raise

if __name__ == "__main__":
    processor = StudentCourseInteractionsProcessor()
    try:
        result = processor.run()
        processor.logger.info("Proceso de métricas de engagement completado exitosamente.")
    except Exception as e:
        processor.logger.error(f"Error en el proceso: {str(e)}")
    finally:
        processor.close() 