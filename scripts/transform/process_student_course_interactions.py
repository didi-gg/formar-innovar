import pandas as pd
import numpy as np
import os
import sys
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
        self.logger.info("Cargando datos...")
        self.student_modules = pd.read_csv("data/interim/moodle/student_modules.csv")
        self.modules_featured = pd.read_csv("data/interim/moodle/modules_featured.csv")

        # Convertir columnas numéricas
        numeric_cols = ['num_views', 'num_interactions', 'days_before_start', 'days_after_end']
        for col in numeric_cols:
            if col in self.student_modules.columns:
                self.student_modules[col] = pd.to_numeric(self.student_modules[col], errors='coerce').fillna(0)

        # Convertir columnas booleanas
        bool_cols = ['has_viewed', 'has_participated', 'was_on_time']
        for col in bool_cols:
            if col in self.student_modules.columns:
                self.student_modules[col] = self.student_modules[col].astype(int)
        
        self.logger.info(f"Datos cargados: {len(self.student_modules)} registros de student_modules")
        self.logger.info(f"Datos cargados: {len(self.modules_featured)} registros de modules_featured")

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
        metrics['avg_days_before_start'] = round(group_data['days_before_start'].mean(), 2)
        metrics['avg_days_after_end'] = round(group_data['days_after_end'].mean(), 2)
        metrics['std_days_before_start'] = round(group_data['days_before_start'].std(), 2) if len(group_data) > 1 else 0
        metrics['std_days_after_end'] = round(group_data['days_after_end'].std(), 2) if len(group_data) > 1 else 0
        metrics['min_days_before_start'] = group_data['days_before_start'].min()
        metrics['min_days_after_end'] = group_data['days_after_end'].min()
        metrics['max_days_after_end'] = group_data['days_after_end'].max()
        metrics['max_days_before_start'] = group_data['days_before_start'].max()
        metrics['median_days_before_start'] = round(group_data['days_before_start'].median(), 2)
        metrics['median_days_after_end'] = round(group_data['days_after_end'].median(), 2)

        # Métricas de puntualidad
        total_modules = len(group_data)
        if total_modules > 0:
            metrics['on_time_rate'] = round(group_data['was_on_time'].mean(), 4)
        else:
            metrics['on_time_rate'] = 0
        metrics['late_rate'] = round(1 - metrics['on_time_rate'], 4)

        # Conteos de acceso temprano y tardío
        metrics['early_access_count'] = (group_data['days_before_start'] > 0).sum()
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
            metrics['skew_views'] = round(stats.skew(views), 4)

            # Curtosis
            metrics['kurtosis_views'] = round(stats.kurtosis(views), 4)
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
            student_views_per_student = course_data.groupby('documento_identificación')['num_views'].sum()
            student_interactions_per_student = course_data.groupby('documento_identificación')['num_interactions'].sum()

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
            metrics['relative_views_percentile'] = 0
            metrics['relative_interaction_percentile'] = 0
            metrics['zscore_views'] = 0
            metrics['zscore_interactions'] = 0

        return metrics
    
    def calculate_mid_week_engagement(self, student_data, modules_data):
        """Calcular engagement de mitad de semana"""
        try:
            # Unir con modules_featured para obtener información de semana
            student_modules_with_week = student_data.merge(
                modules_data[['course_module_id', 'week', 'planned_start_date', 'planned_end_date']], 
                on='course_module_id', 
                how='left'
            )

            # Convertir fechas si están disponibles
            if 'first_view' in student_modules_with_week.columns and 'planned_start_date' in student_modules_with_week.columns:
                student_modules_with_week['first_view'] = pd.to_datetime(student_modules_with_week['first_view'], errors='coerce')
                student_modules_with_week['planned_start_date'] = pd.to_datetime(student_modules_with_week['planned_start_date'], errors='coerce')

                # Calcular día de la semana del primer acceso (0=Lunes, 6=Domingo)
                student_modules_with_week['access_weekday'] = student_modules_with_week['first_view'].dt.dayofweek

                # Considerar lunes a miércoles como mitad de semana (0, 1, 2)
                mid_week_accessed = student_modules_with_week[
                    (student_modules_with_week['access_weekday'].isin([0, 1, 2])) & 
                    (student_modules_with_week['has_viewed'] == 1)
                ]

                total_viewed = student_modules_with_week[student_modules_with_week['has_viewed'] == 1]

                if len(total_viewed) > 0:
                    mid_week_engagement = len(mid_week_accessed) / len(total_viewed)
                else:
                    mid_week_engagement = 0
            else:
                mid_week_engagement = 0

            return {'mid_week_engagement': round(mid_week_engagement, 4)}
        except Exception as e:
            return {'mid_week_engagement': 0}
    
    def process_student_engagement_metrics(self):
        """Procesar todas las métricas de engagement para cada estudiante"""
        self.logger.info("Iniciando cálculo de métricas de engagement...")

        # Agrupar por la clave especificada
        groupby_cols = ['documento_identificación', 'year', 'id_grado', 'sede', 'id_asignatura', 'period']

        results = []

        # Procesar cada grupo de estudiante-curso
        for group_key, group_data in self.student_modules.groupby(groupby_cols):
            try:
                # Crear diccionario con las claves del grupo
                result = dict(zip(groupby_cols, group_key))

                # Calcular métricas básicas
                basic_metrics = self.calculate_basic_engagement_metrics(group_data)
                result.update(basic_metrics)

                # Calcular métricas de actividad
                activity_metrics = self.calculate_activity_metrics(group_data)
                result.update(activity_metrics)

                # Calcular métricas temporales
                temporal_metrics = self.calculate_temporal_metrics(group_data)
                result.update(temporal_metrics)

                # Calcular métricas de dispersión
                dispersion_metrics = self.calculate_dispersion_metrics(group_data)
                result.update(dispersion_metrics)

                # Calcular métricas de engagement relativo
                # Filtrar datos del curso para comparación
                course_filter = (
                    (self.student_modules['year'] == group_key[1]) &
                    (self.student_modules['id_grado'] == group_key[2]) &
                    (self.student_modules['sede'] == group_key[3]) &
                    (self.student_modules['id_asignatura'] == group_key[4]) &
                    (self.student_modules['period'] == group_key[5])
                )
                course_data = self.student_modules[course_filter]

                relative_metrics = self.calculate_relative_engagement(group_data, course_data)
                result.update(relative_metrics)

                # Calcular engagement de mitad de semana (placeholder)
                mid_week_metrics = self.calculate_mid_week_engagement(group_data, self.modules_featured)
                result.update(mid_week_metrics)
                
                results.append(result)

            except Exception as e:
                self.logger.error(f"Error procesando grupo {group_key}: {str(e)}")
                continue

        # Convertir a DataFrame
        engagement_df = pd.DataFrame(results)
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