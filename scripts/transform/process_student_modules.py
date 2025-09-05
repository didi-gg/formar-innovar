import pandas as pd
import os
import sys
import duckdb
import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.base_script import BaseScript
from utils.moodle_path_resolver import MoodlePathResolver


class StudentModulesProcessor(BaseScript):
    @staticmethod
    def _classify_event(eventname):
        eventname = str(eventname).lower()
        if "view" in eventname:
            return "view"
        else:
            return "interaction"

    @staticmethod
    def _merge_modules_students(students_df, modules_df):
        students_sel = students_df[["documento_identificación","year","id_grado","sede"]]
        return students_sel.merge(modules_df, on=["id_grado", "year", "sede"], how="inner")

    @staticmethod
    def _get_student_logs_summary(logs_df):
        """
        Genera un resumen de la actividad de los estudiantes por módulo.

        Métricas base calculadas:
        - num_views: Número total de visualizaciones del módulo por estudiante
        - num_interactions: Número total de interacciones del módulo por estudiante
        - first_view: Fecha y hora del primer acceso al módulo
        - last_view: Fecha y hora del último acceso al módulo

        Args:
            logs_df (pd.DataFrame): DataFrame con logs de actividad de estudiantes

        Returns:
            pd.DataFrame: Resumen de actividad por estudiante y módulo
        """
        student_logs_summary = (
            logs_df.groupby(["documento_identificación", "contextinstanceid", "year", "platform"])
            .agg(
                num_views=("event_type", lambda x: (x == "view").sum()),
                num_interactions=("event_type", lambda x: (x == "interaction").sum()),
                first_view=("timecreated", "min"),
                last_view=("timecreated", "max"),
            )
            .reset_index()
            .rename(columns={"contextinstanceid": "course_module_id"})
        )
        return student_logs_summary

    @staticmethod
    def _get_student_grades(enrollments_file, year, platform):
        """
        Obtiene las calificaciones finales de los estudiantes para los módulos usando SQL.
        
        Args:
            enrollments_file (str): Ruta al archivo de enrollments
            year (int): Año para el cual obtener las calificaciones
            platform (str): Plataforma ('moodle' o 'edukrea')
            
        Returns:
            pd.DataFrame: DataFrame con calificaciones finales por estudiante y módulo
        """
        try:
            # Determinar el folder y la columna de join según la plataforma
            if platform.lower() == 'moodle':
                folder = year
                user_id_column = 'moodle_user_id'
            elif platform.lower() == 'edukrea':
                folder = "Edukrea"
                user_id_column = 'edukrea_user_id'
            else:
                raise ValueError(f"Platform '{platform}' no es válida. Usa 'moodle' o 'edukrea'.")
            
            # Obtener rutas de las tablas
            tables = ["grade_grades", "grade_items"]
            grades_file, grades_items_file = MoodlePathResolver.get_paths(folder, *tables)
            
            # Query SQL para obtener las calificaciones
            con = duckdb.connect()
            sql = f"""
            SELECT 
                e.documento_identificación,
                gi.iteminstance AS course_module_id,
                g.finalgrade
            FROM '{grades_file}' g
            JOIN '{grades_items_file}' gi ON g.itemid = gi.id
            JOIN '{enrollments_file}' e ON g.userid = e.{user_id_column}
            WHERE e.year = {year}
            """
            
            result = con.execute(sql).df()
            con.close()
            
            return result
            
        except Exception as e:
            print(f"Error obteniendo calificaciones para {platform} {year}: {e}")
            return pd.DataFrame(columns=['documento_identificación', 'course_module_id', 'finalgrade'])

    @staticmethod
    def _calculate_metrics(df):
        """
        Calcula métricas de engagement y temporalidad para los módulos de estudiantes.
        
        Métricas calculadas:
        - has_viewed: Indica si el estudiante visualizó el módulo al menos una vez (1=sí, 0=no)
        - has_participated: Indica si el estudiante interactuó con el módulo al menos una vez (1=sí, 0=no)
        - days_from_planned_start: Días antes de la fecha de inicio planificada en que el estudiante 
          accedió por primera vez al módulo (valores negativos indican acceso temprano)
        - days_after_end: Días después de la fecha de fin planificada en que el estudiante 
          accedió por última vez al módulo (valores negativos indican acceso dentro del período)
        - was_on_time: Indica si el primer acceso del estudiante fue dentro del período 
          planificado del módulo (1=sí, 0=no)

        Args:
            df (pd.DataFrame): DataFrame con datos de módulos y logs de estudiantes

        Returns:
            pd.DataFrame: DataFrame con las métricas calculadas agregadas
        """
        df["first_view"] = pd.to_datetime(df["first_view"], errors="coerce")
        df["last_view"] = pd.to_datetime(df["last_view"], errors="coerce")
        df["planned_start_date"] = pd.to_datetime(df["planned_start_date"], errors="coerce")
        df["planned_end_date"] = pd.to_datetime(df["planned_end_date"], errors="coerce")

        # Métricas de engagement
        df["has_viewed"] = (df["num_views"].fillna(0) > 0).astype(int)
        df["has_participated"] = (df["num_interactions"].fillna(0) > 0).astype(int)
        
        # Métricas temporales
        df["days_from_planned_start"] = (df["first_view"] - df["planned_start_date"]).dt.days
        df["days_after_end"] = (df["last_view"] - df["planned_end_date"]).dt.days
        df["was_on_time"] = ((df["first_view"] >= df["planned_start_date"]) & (df["first_view"] <= df["planned_end_date"])).astype(int)

        return df

    @staticmethod
    def _calculate_total_time_in_modules(logs_df, max_duration_minutes=60):
        """
        Calcula el tiempo total que cada estudiante pasa en cada módulo siguiendo los lineamientos:
        1. Ordena los logs por estudiante y tiempo
        2. Calcula la duración de cada acceso (diferencia con el siguiente acceso)
        3. Suma todas las duraciones para la misma actividad
        4. Maneja duraciones dudosas usando estadísticas robustas (IQR, mediana) y k-NN
        5. Considera casos especiales para el último acceso

        Mejoras implementadas para evitar duraciones irreales:
        - Límite absoluto inmediato: 45 min máximo por acceso individual
        - Detección de outliers muy conservadora (1× std, 1× IQR)
        - Umbrales dinámicos con límites absolutos (20-30 min máximo)
        - Uso de mediana en lugar de media para mayor robustez
        - Límite final: 1 hora máximo de tiempo total por módulo
        - Valores por defecto muy conservadores (2-5 min)

        Args:
            logs_df (pd.DataFrame): DataFrame con logs de actividad de estudiantes
            max_duration_minutes (int): Umbral máximo para duraciones válidas en minutos (default: 60)

        Returns:
            pd.DataFrame: DataFrame con tiempo total por estudiante y módulo
        """
        # Convertir timestamp a datetime
        logs_df = logs_df.copy()
        logs_df['timecreated'] = pd.to_datetime(logs_df['timecreated'], unit='s')

        # Filtrar solo logs de módulos (contextinstanceid no nulo)
        module_logs = logs_df[logs_df['contextinstanceid'].notna()].copy()

        # Ordenar por estudiante y tiempo
        module_logs = module_logs.sort_values(['documento_identificación', 'timecreated'])
        
        # Calcular duración de cada acceso
        module_logs['next_access_time'] = module_logs.groupby('documento_identificación')['timecreated'].shift(-1)
        module_logs['duration_minutes'] = (
            module_logs['next_access_time'] - module_logs['timecreated']
        ).dt.total_seconds() / 60

        # Aplicar límite absoluto inmediato para evitar duraciones extremas
        # Cualquier duración individual mayor a 45 minutos se considera una sesión abandonada
        module_logs['duration_minutes'] = module_logs['duration_minutes'].clip(upper=45.0)

        # Filtrar accesos a módulos específicos
        module_access_logs = module_logs[module_logs['contextinstanceid'].notna()].copy()
        
        if module_access_logs.empty:
            return pd.DataFrame(columns=['documento_identificación', 'contextinstanceid', 'year', 'platform', 'total_time_minutes'])

        # Identificar duraciones dudosas por módulo
        def clean_durations_by_module(group):
            valid_durations = group['duration_minutes'].dropna()
            valid_durations = valid_durations[valid_durations > 0]
            valid_durations = valid_durations[valid_durations <= max_duration_minutes]

            if len(valid_durations) == 0:
                # Si no hay duraciones válidas, usar un valor por defecto
                group['duration_minutes_clean'] = 3.0  # 3 minutos por defecto (más conservador)
                return group

            # Calcular estadísticas para detectar outliers de manera más agresiva
            mean_duration = valid_durations.mean()
            std_duration = valid_durations.std()
            median_duration = valid_durations.median()
            q75 = valid_durations.quantile(0.75)
            q25 = valid_durations.quantile(0.25)
            iqr = q75 - q25

            # Usar múltiples criterios para detectar outliers de manera muy conservadora
            if pd.isna(std_duration) or std_duration == 0:
                # Si no hay variabilidad, usar un threshold muy conservador
                threshold = min(mean_duration * 1.2, 20.0)  # Máximo 20 minutos
            else:
                # Usar el criterio más restrictivo entre:
                # 1. Media + 1 * desviación estándar (muy conservador)
                # 2. Q75 + 1 * IQR (método de outliers por cuartiles conservador)
                # 3. Umbral máximo absoluto muy bajo
                threshold_std = mean_duration + 1.0 * std_duration
                threshold_iqr = q75 + 1.0 * iqr
                threshold = min(threshold_std, threshold_iqr, max_duration_minutes, 30.0)  # Máximo absoluto de 30 min

            # Identificar duraciones dudosas
            group['is_valid_duration'] = (
                (group['duration_minutes'] > 0) & 
                (group['duration_minutes'] <= threshold) & 
                (group['duration_minutes'].notna())
            )

            # Para duraciones dudosas, usar k-NN si hay suficientes datos válidos
            if len(valid_durations) >= 3:
                try:
                    # Preparar datos para k-NN (usar características del acceso)
                    features = []
                    for _, row in group.iterrows():
                        # Usar hora del día y día de la semana como características
                        hour = row['timecreated'].hour
                        day_of_week = row['timecreated'].dayofweek
                        features.append([hour, day_of_week])

                    features = np.array(features)
                    valid_indices = group['is_valid_duration'].values

                    if np.sum(valid_indices) >= 2:
                        # Entrenar k-NN con duraciones válidas
                        knn = NearestNeighbors(n_neighbors=min(3, np.sum(valid_indices)))
                        knn.fit(features[valid_indices])

                        # Predecir duraciones para casos dudosos
                        invalid_indices = ~valid_indices
                        if np.sum(invalid_indices) > 0:
                            distances, indices = knn.kneighbors(features[invalid_indices])

                            # Calcular duración promedio de los vecinos más cercanos
                            valid_durations_array = group.loc[group['is_valid_duration'], 'duration_minutes'].values
                            
                            predicted_durations = []
                            for neighbor_indices in indices:
                                neighbor_durations = valid_durations_array[neighbor_indices]
                                predicted_durations.append(np.mean(neighbor_durations))

                            group.loc[invalid_indices, 'duration_minutes_clean'] = predicted_durations

                        # Mantener duraciones válidas
                        group.loc[valid_indices, 'duration_minutes_clean'] = group.loc[valid_indices, 'duration_minutes']
                    else:
                        # Si no hay suficientes datos válidos, usar la mediana (más robusta que la media)
                        group['duration_minutes_clean'] = median_duration

                except Exception:
                    # En caso de error, usar la mediana
                    group['duration_minutes_clean'] = median_duration
            else:
                # Si hay muy pocos datos, usar la mediana o valor por defecto
                if len(valid_durations) > 0:
                    group['duration_minutes_clean'] = median_duration
                else:
                    group['duration_minutes_clean'] = 3.0  # 3 minutos por defecto

            # Para el último acceso de cada estudiante (sin siguiente acceso), usar valor típico muy conservador
            last_access_mask = group['duration_minutes'].isna()
            if last_access_mask.any():
                if len(valid_durations) > 0:
                    # Usar el mínimo entre mediana y 5 minutos para ser muy conservador
                    typical_duration = min(median_duration, 5.0)
                else:
                    typical_duration = 2.0
                group.loc[last_access_mask, 'duration_minutes_clean'] = typical_duration
            return group

        # Aplicar limpieza por módulo
        # Usar transform en lugar de apply para evitar problemas con las columnas de agrupación
        def apply_cleaning_by_module(df):
            result_list = []
            for name, group in df.groupby('contextinstanceid'):
                cleaned_group = clean_durations_by_module(group)
                result_list.append(cleaned_group)
            return pd.concat(result_list, ignore_index=True)

        module_access_logs = apply_cleaning_by_module(module_access_logs)

        # Sumar duraciones por estudiante y módulo
        time_summary = (
            module_access_logs.groupby(['documento_identificación', 'contextinstanceid', 'year', 'platform'])
            .agg(
                total_time_minutes=('duration_minutes_clean', 'sum'),
                num_accesses=('duration_minutes_clean', 'count')
            )
            .reset_index()
            .rename(columns={'contextinstanceid': 'course_module_id'})
        )

        # Aplicar límite final al tiempo total por módulo (segunda capa de protección)
        # Un estudiante no debería pasar más de 1 hora en total en un módulo específico
        max_total_time = 60.0  # 1 hora máximo por módulo (muy conservador)
        time_summary['total_time_minutes'] = time_summary['total_time_minutes'].clip(upper=max_total_time)

        return time_summary

    @staticmethod
    def _print_time_summary(df):
        """
        Imprime un resumen estadístico de la nueva variable de tiempo total en módulos.
        
        Args:
            df (pd.DataFrame): DataFrame con la variable total_time_minutes
        """
        print("\n" + "="*80)
        print("📊 RESUMEN DE LA NUEVA VARIABLE: TIEMPO TOTAL EN MÓDULOS")
        print("="*80)

        # Filtrar solo registros con tiempo calculado
        time_data = df[df['total_time_minutes'].notna()]['total_time_minutes']

        if len(time_data) == 0:
            print("⚠️  No se encontraron datos de tiempo calculados.")
            return

        print(f"📈 Estadísticas Descriptivas:")
        print(f"   • Total de registros con tiempo: {len(time_data):,}")
        print(f"   • Promedio: {time_data.mean():.2f} minutos ({time_data.mean()/60:.2f} horas)")
        print(f"   • Mediana: {time_data.median():.2f} minutos ({time_data.median()/60:.2f} horas)")
        print(f"   • Desviación estándar: {time_data.std():.2f} minutos")
        print(f"   • Mínimo: {time_data.min():.2f} minutos")
        print(f"   • Máximo: {time_data.max():.2f} minutos ({time_data.max()/60:.2f} horas)")
        print(f"\n📊 Distribución por Rangos:")

        # Definir rangos de tiempo más conservadores
        ranges = [
            (0, 3, "Muy corto (0-3 min)"),
            (3, 10, "Corto (3-10 min)"),
            (10, 20, "Moderado (10-20 min)"),
            (20, 45, "Largo (20-45 min)"),
            (45, 60, "Muy largo (45-60 min)"),
            (60, float('inf'), "Límite máximo (exactamente 1 hora)")
        ]

        for min_val, max_val, label in ranges:
            if max_val == float('inf'):
                count = len(time_data[time_data >= min_val])
            else:
                count = len(time_data[(time_data >= min_val) & (time_data < max_val)])
            
            percentage = (count / len(time_data)) * 100
            print(f"   • {label}: {count:,} registros ({percentage:.1f}%)")

        print(f"\n📋 Percentiles:")
        percentiles = [25, 50, 75, 90, 95, 99]
        for p in percentiles:
            value = time_data.quantile(p/100)
            print(f"   • P{p}: {value:.2f} minutos ({value/60:.2f} horas)")

        # Información adicional sobre accesos
        access_data = df[df['num_accesses'].notna()]['num_accesses']
        if len(access_data) > 0:
            print(f"\n🔄 Información de Accesos:")
            print(f"   • Promedio de accesos por módulo: {access_data.mean():.2f}")
            print(f"   • Mediana de accesos: {access_data.median():.0f}")
            print(f"   • Máximo de accesos: {access_data.max():.0f}")

        print("="*80)
        print("✅ La nueva variable 'total_time_minutes' ha sido calculada exitosamente")
        print("   siguiendo los lineamientos de cálculo de tiempo en actividades.")
        print("="*80 + "\n")

    @staticmethod
    def _process_df(modules_df, students_df, logs_df):
        """
        Procesa y combina los datos de módulos, estudiantes y logs para generar 
        un dataset con métricas de engagement y temporalidad.
        
        El dataset final incluye:
        - Información básica: year, course_id, platform, course_module_id, sede, id_grado
        - Datos del estudiante: documento_identificación
        - Métricas de actividad: num_views, num_interactions, first_view, last_view
        - Métricas de engagement: has_viewed, has_participated
        - Métricas temporales: days_from_planned_start, days_after_end, was_on_time
        - Métricas de tiempo: total_time_minutes, num_accesses
        - Calificaciones: finalgrade
        
        Args:
            modules_df (pd.DataFrame): DataFrame con información de módulos
            students_df (pd.DataFrame): DataFrame con información de estudiantes
            logs_df (pd.DataFrame): DataFrame con logs de actividad
            
        Returns:
            pd.DataFrame: Dataset procesado con todas las métricas calculadas
        """
        modules_df = modules_df[
            [
                "year",
                "sede",
                "id_grado",
                "id_asignatura",
                "period",
                "course_id",
                "platform",
                "course_module_id",
                "planned_start_date",
                "planned_end_date",
            ]
        ].copy()

        df_base = StudentModulesProcessor._merge_modules_students(students_df, modules_df)

        logs_df["event_type"] = logs_df["eventname"].apply(StudentModulesProcessor._classify_event)

        student_logs_summary = StudentModulesProcessor._get_student_logs_summary(logs_df)

        # Calcular tiempo total en módulos
        time_summary = StudentModulesProcessor._calculate_total_time_in_modules(logs_df)

        df_full = df_base.merge(student_logs_summary, on=["documento_identificación", "course_module_id", "year", "platform"], how="left")
        
        # Agregar métricas de tiempo
        df_full = df_full.merge(time_summary, on=["documento_identificación", "course_module_id", "year", "platform"], how="left")

        df_full = StudentModulesProcessor._calculate_metrics(df_full)

        # Obtener calificaciones para cada combinación de año y plataforma
        enrollments_file = "data/interim/estudiantes/enrollments.csv"
        year_platform_combinations = df_full[['year', 'platform']].drop_duplicates()
        
        all_grades = []
        for _, row in year_platform_combinations.iterrows():
            year = row['year']
            platform = row['platform']
            grades = StudentModulesProcessor._get_student_grades(enrollments_file, year, platform)
            if not grades.empty:
                grades['year'] = year
                grades['platform'] = platform
                all_grades.append(grades)
        
        # Combinar calificaciones si hay datos
        if all_grades:
            grades_df = pd.concat(all_grades, ignore_index=True)
            df_full = df_full.merge(
                grades_df[['documento_identificación', 'course_module_id', 'finalgrade', 'year', 'platform']], 
                on=['documento_identificación', 'course_module_id', 'year', 'platform'], 
                how='left'
            )
        else:
            df_full['finalgrade'] = None

        # Remove the date columns used only for calculations
        df_full = df_full.drop(columns=["planned_start_date", "planned_end_date"])

        return df_full

    def process_course_data(self):
        modules_df = pd.read_csv("data/interim/moodle/modules_featured.csv")
        students_df = pd.read_csv("data/interim/estudiantes/enrollments.csv")
        logs_df = pd.read_csv("data/interim/moodle/student_logs.csv")

        student_modules_moodle = StudentModulesProcessor._process_df(modules_df, students_df, logs_df)

        self.save_to_csv(student_modules_moodle, "data/interim/moodle/student_modules.csv")
        self.logger.info("Courses data processed and saved successfully.")

        # Mostrar resumen de la nueva variable de tiempo
        StudentModulesProcessor._print_time_summary(student_modules_moodle)

if __name__ == "__main__":
    processor = StudentModulesProcessor()
    processor.process_course_data()
    processor.logger.info("Student Modules processed successfully.")
    processor.close()
