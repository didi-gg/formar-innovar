import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.base_script import BaseScript
from utils.academic_period_utils import AcademicPeriodUtils


class DatasetProcessorBase(BaseScript):

    # data/interim/estudiantes/enrollments.csv
    COLUMNS_ENROLLMENTS = [
        'documento_identificación',
        'moodle_user_id',
        'year',
        'edukrea_user_id',
        'id_grado',
        'sede'
    ]

    # data/interim/estudiantes/estudiantes_clean.csv
    COLUMNS_STUDENTS = [
        'sede',
        'año_ingreso',
        'antigüedad',
        'género',
        'documento_identificación',
        'fecha_nacimiento',
        'demuestra_confianza',
        'país_origen',
        'estrato',
        'tipo_vivienda',
        'zona_vivienda',
        'horas_semana_estudio_casa',
        'interés_estudios_superiores',
        'medio_transporte',
        'apoyo_familiar',
        'total_hermanos',
        'familia',
        'actividades_extracurriculares',
        'enfermedades',
        'proyección_vocacional',
        'participación_clase',
        'nee',
        'valoración_emocional',
        'nivel_motivación'
    ]

    # data/interim/moodle/student_login_moodle.csv
    COLUMNS_STUDENTS_LOGINS = [
        "documento_identificación",
        "year",
        "periodo", # Rename to period
        "count_login",
        "max_inactividad",
        "count_login_mon",
        "count_login_tue",
        "count_login_wed",
        "count_login_thu",
        "count_login_fri",
        "count_login_sat",
        "count_login_sun",
        "count_jornada_madrugada",
        "count_jornada_mañana",
        "count_jornada_tarde",
        "count_jornada_noche",
        "login_consistency",
        "dia_preferido",
        "jornada_preferida",
        "login_regularity_score",
        "consecutive_days_max",
        "gaps_between_sessions_avg",
        "engagement_decay",
        "activity_percentile",
        "longest_inactivity_streak"
    ]

    # data/interim/moodle/courses_base.csv
    COLUMNS_COURSES_BASE = [
        "sede",
        "year",
        "id_grado",
        "id_asignatura",
        "period",
        "intensidad",
        "id_docente",
        "rol_adicional",
        "nivel_educativo",
        "total_subjects",
        "total_hours",
        "unique_students_count",
        "update_events_count",
        "years_experience_ficc",
        "years_experience_total",
        "age",
    ]

    # data/interim/moodle/courses.csv
    COLUMNS_COURSES = [
        "id_asignatura",
        "id_grado",
        "year",
        "period",
        "sede",
        "count_evaluation",
        "count_collaboration",
        "count_content",
        "count_in_english",
        "count_interactive",
        "num_modules",
        "num_modules_updated",
        "num_teacher_views_before_planned_start_date",
        "teacher_total_updates",
        "teacher_total_views",
        "student_total_views",
        "student_total_interactions",
        "min_days_since_creation",
        "max_days_since_creation",
        "avg_days_since_creation",
        "median_days_since_creation",
        "avg_days_since_last_update",
        "median_days_since_last_update",
        "percent_evaluation",
        "percent_collaboration",
        "percent_content",
        "percent_in_english",
        "percent_interactive",
        "percent_updated",
        "num_students",
        "num_students_viewed",
        "num_students_interacted",
        "num_modules_viewed",
        "avg_views_per_student",
        "median_views_per_student",
        "avg_interactions_per_student",
        "median_interactions_per_student",
        "id_least_viewed_module",
        "students_viewed_least_module",
        "id_most_late_opened_module",
        "percent_modules_out_of_date",
        "percent_students_viewed",
        "percent_students_interacted",
        "percent_modules_viewed"
    ]

    # data/interim/moodle/student_course_interactions.csv
    COLUMNS_STUDENT_COURSE_INTERACTIONS = [
        "documento_identificación",
        "year",
        "id_grado",
        "sede",
        "id_asignatura",
        "period",
        "total_modules",
        "modules_viewed",
        "modules_participated",
        "percent_modules_viewed",
        "percent_modules_participated",
        "has_viewed_all_modules",
        "has_participated_all_modules",
        "total_views",
        "total_interactions",
        "avg_views_per_module",
        "avg_interactions_per_module",
        "median_views_per_module",
        "median_interactions_per_module",
        "min_views_per_module",
        "min_interactions_per_module",
        "max_views_in_a_module",
        "max_interactions_in_a_module",
        "std_views_per_module",
        "std_interactions_per_module",
        "interaction_to_view_ratio",
        "log_total_views",
        "log_total_interactions",
        "avg_days_from_planned_start",
        "avg_days_after_end",
        "std_days_from_planned_start",
        "std_days_after_end",
        "min_days_from_planned_start",
        "min_days_after_end",
        "max_days_after_end",
        "max_days_from_planned_start",
        "median_days_from_planned_start",
        "median_days_after_end",
        "on_time_rate",
        "late_rate",
        "early_access_count",
        "late_access_count",
        "iqr_views",
        "skew_views",
        "kurtosis_views",
        "relative_views_percentile",
        "relative_interaction_percentile",
        "zscore_views",
        "zscore_interactions",
        "mid_week_engagement",
        "max_activity_grade",
        "avg_activity_grade",
        "min_activity_grade",
        "std_activity_grade",
        "graded_activities_count",
        "percent_graded_activities",
        "total_course_time_minutes",
        "avg_time_per_module",
        "median_time_per_module",
        "min_time_per_module",
        "max_time_per_module",
        "std_time_per_module",
        "total_course_time_hours",
        "time_engagement_level"
    ]

    # data/interim/teachers_merged_final.csv
    # KEY teacher_id_docente - id_docente
    COLUMNS_TEACHERS_FEATURED = [
        'teacher_experiencia_ficc_percentil',
        'teacher_experiencia_total_percentil',
        'teacher_nivel_educativo_num',
        'teacher_nivel_educativo_percentil',
        'teacher_experiencia_nivel',
        'teacher_experiencia_nivel_ficc',
    ]

    # data/interim/moodle/sequence_analysis_features.csv
    COLUMNS_SEQUENCE_ANALYSIS = [
        "real_sequence_length",
        "total_accesses",
        "levenshtein_distance",
        "levenshtein_normalized",
        "substring_similarity",
        "common_bigrams",
        "common_trigrams",
        "sequence_match_ratio",
        "extra_activities",
        "missing_activities",
        "correct_order_count",
        "correct_order_ratio",
    ]

    def _load_and_prepare_enrollments(self):
        """Carga y prepara el dataset base de enrollments"""
        enrollments_df = pd.read_csv("data/interim/estudiantes/enrollments.csv")
        enrollments_df = enrollments_df[self.COLUMNS_ENROLLMENTS]
        enrollments_df['documento_identificación'] = enrollments_df['documento_identificación'].astype(str).str.strip()
        self.logger.info(f"Dataset de enrollments cargado: {enrollments_df.shape}")
        return enrollments_df

    def _load_and_prepare_students(self):
        """Carga y prepara el dataset de estudiantes, incluyendo el manejo de ID duplicado"""
        students_df = pd.read_csv("data/interim/estudiantes/estudiantes_clean.csv")
        students_df = students_df[self.COLUMNS_STUDENTS]
        students_df['documento_identificación'] = students_df['documento_identificación'].astype(str).str.strip()
        self.logger.info(f"Dataset de estudiantes cargado: {students_df.shape}")

        # Manejar caso específico de ID duplicado
        original_id = 'b590e0984547ece1eda24fa512647a538cac6f79b4b3d59c0972638354830fe5'
        alternative_id = 'c5fcc9a2b1087f1c855cc8c85fa4446f2ad80e0e4734c0afbe854332acc1c22f'

        original_row = students_df[students_df['documento_identificación'] == original_id]
        if len(original_row) > 0:
            new_row = original_row.copy()
            new_row['documento_identificación'] = alternative_id
            students_df = pd.concat([students_df, new_row], ignore_index=True)
            self.logger.info(f"Agregada fila adicional para ID {alternative_id}")
            self.logger.info(f"Nuevo tamaño del dataset de estudiantes: {students_df.shape}")
        return students_df

    def _load_and_prepare_student_logins(self):
        """Carga y prepara el dataset de logins de estudiantes"""
        student_logins_df = pd.read_csv("data/interim/moodle/student_login_moodle.csv")
        student_logins_df = student_logins_df[self.COLUMNS_STUDENTS_LOGINS]
        student_logins_df = student_logins_df.rename(columns={'periodo': 'period'})
        self.logger.info(f"Dataset de logins de estudiantes cargado: {student_logins_df.shape}")
        return student_logins_df

    def _load_and_prepare_courses_base(self):
        """Carga y prepara el dataset de cursos base"""
        courses_base_df = pd.read_csv("data/interim/moodle/courses_base.csv")
        courses_base_df = courses_base_df[self.COLUMNS_COURSES_BASE]
        self.logger.info(f"Dataset de cursos base cargado: {courses_base_df.shape}")
        return courses_base_df

    def _load_and_prepare_courses(self):
        """Carga y prepara el dataset de cursos"""
        courses_df = pd.read_csv("data/interim/moodle/courses.csv")
        courses_df = courses_df[self.COLUMNS_COURSES]
        self.logger.info(f"Dataset de cursos cargado: {courses_df.shape}")
        return courses_df

    def _load_and_prepare_student_interactions(self):
        """Carga y prepara el dataset de interacciones de estudiantes con cursos"""
        student_interactions_df = pd.read_csv("data/interim/moodle/student_course_interactions.csv")
        student_interactions_df = student_interactions_df[self.COLUMNS_STUDENT_COURSE_INTERACTIONS]
        self.logger.info(f"Dataset de interacciones de estudiantes cargado: {student_interactions_df.shape}")
        return student_interactions_df

    def _load_and_prepare_teachers_featured(self):
        """Carga y prepara el dataset de características de teachers"""
        teachers_df = pd.read_csv("data/interim/teachers_merged_final.csv")
        # Seleccionar solo las columnas necesarias incluyendo las llaves de unión
        columns_to_select = ['teacher_id_docente', 'teacher_year'] + self.COLUMNS_TEACHERS_FEATURED
        teachers_df = teachers_df[columns_to_select]
        # Renombrar las llaves para hacer match con courses_base
        teachers_df = teachers_df.rename(columns={
            'teacher_id_docente': 'id_docente',
            'teacher_year': 'year'
        })
        self.logger.info(f"Dataset de características de teachers cargado: {teachers_df.shape}")
        return teachers_df

    def _load_and_prepare_sequence_analysis(self):
        """Carga y prepara el dataset de análisis de secuencias"""
        sequence_analysis_df = pd.read_csv("data/interim/moodle/sequence_analysis_features.csv")
        # Seleccionar solo las columnas necesarias incluyendo las llaves de unión
        columns_to_select = ['documento_identificación', 'id_asignatura', 'id_grado', 'sede', 'year', 'period'] + self.COLUMNS_SEQUENCE_ANALYSIS
        sequence_analysis_df = sequence_analysis_df[columns_to_select]
        self.logger.info(f"Dataset de análisis de secuencias cargado: {sequence_analysis_df.shape}")
        return sequence_analysis_df

    def _merge_with_analysis(self, left_df, right_df, merge_keys, dataset_name, how='inner'):
        """Realiza un merge con análisis detallado de los resultados"""
        self.logger.info(f"=== UNIÓN CON {dataset_name.upper()} ===")

        merged_df = left_df.merge(
            right_df, 
            on=merge_keys, 
            how=how,
            suffixes=('', f'_{dataset_name.lower().replace(" ", "_")}'),
            indicator=True
        )

        # Analizar matches
        match_counts = merged_df['_merge'].value_counts()
        self.logger.info(f"Resultados del {how} join con {dataset_name}:")
        self.logger.info(f"  - Registros con datos en ambos datasets: {match_counts.get('both', 0)}")
        
        if how == 'left':
            left_only_count = match_counts.get('left_only', 0)
            self.logger.info(f"  - Registros solo en dataset izquierdo (sin datos de {dataset_name}): {left_only_count}")
        elif how == 'inner':
            left_only_count = match_counts.get('left_only', 0)
            right_only_count = match_counts.get('right_only', 0)
            if left_only_count > 0 or right_only_count > 0:
                self.logger.warning(f"  - Registros inesperados left_only: {left_only_count}")
                self.logger.warning(f"  - Registros inesperados right_only: {right_only_count}")

        # Eliminar columna indicator
        merged_df = merged_df.drop('_merge', axis=1)
        self.logger.info(f"Después de unir {dataset_name}: {merged_df.shape}")

        return merged_df

    def _detailed_course_analysis(self, combined_df, courses_df, merge_keys):
        """Realiza análisis detallado de cursos cuando hay registros sin match"""
        no_match_courses = combined_df[combined_df['_merge'] == 'left_only']
        if len(no_match_courses) > 0:
            sample_keys = no_match_courses[merge_keys].drop_duplicates().head(10)
            first_no_match = sample_keys.iloc[0]

            self.logger.info("=== ANÁLISIS DETALLADO DEL PRIMER CASO SIN MATCH ===")
            self.logger.info(f"Buscando en courses.csv: sede={first_no_match['sede']}, year={first_no_match['year']}, id_grado={first_no_match['id_grado']}, id_asignatura={first_no_match['id_asignatura']}, period={first_no_match['period']}")

            # Verificar qué combinaciones existen en courses.csv para esa sede/año/grado
            courses_subset = courses_df[
                (courses_df['sede'] == first_no_match['sede']) & 
                (courses_df['year'] == first_no_match['year']) & 
                (courses_df['id_grado'] == first_no_match['id_grado'])
            ]
            self.logger.info(f"Cursos disponibles para sede={first_no_match['sede']}, year={first_no_match['year']}, grado={first_no_match['id_grado']}: {len(courses_subset)}")

            if len(courses_subset) > 0:
                unique_asignaturas = courses_subset['id_asignatura'].unique()
                unique_periods = courses_subset['period'].unique()
                self.logger.info(f"Asignaturas disponibles: {sorted(unique_asignaturas)}")
                self.logger.info(f"Períodos disponibles: {sorted(unique_periods)}")
                
                asignatura_exists = first_no_match['id_asignatura'] in unique_asignaturas
                period_exists = first_no_match['period'] in unique_periods
                self.logger.info(f"¿Existe asignatura {first_no_match['id_asignatura']}? {asignatura_exists}")
                self.logger.info(f"¿Existe período {first_no_match['period']}? {period_exists}")
            else:
                self.logger.warning(f"No hay NINGÚN curso para sede={first_no_match['sede']}, year={first_no_match['year']}, grado={first_no_match['id_grado']}")

    def _merge_courses_with_analysis(self, combined_df, courses_df, merge_keys, how='inner'):
        """Realiza merge con cursos incluyendo análisis detallado"""
        self.logger.info("=== UNIÓN CON CURSOS ===")

        merged_df = combined_df.merge(
            courses_df, 
            on=merge_keys, 
            how=how,
            suffixes=('', '_course'),
            indicator=True
        )

        # Analizar matches
        match_counts = merged_df['_merge'].value_counts()
        self.logger.info(f"Resultados del {how} join con cursos:")
        self.logger.info(f"  - Registros con datos en ambos datasets: {match_counts.get('both', 0)}")
        
        if how == 'left':
            left_only_count = match_counts.get('left_only', 0)
            self.logger.info(f"  - Registros solo en dataset izquierdo (sin datos de cursos): {left_only_count}")
        elif how == 'inner':
            left_only_count = match_counts.get('left_only', 0)
            if left_only_count > 0:
                self.logger.warning(f"  - Registros inesperados sin match: {left_only_count}")
                # Análisis detallado si hay registros sin match (no debería ocurrir con inner join)
                self._detailed_course_analysis(merged_df, courses_df, merge_keys)

        # Eliminar columna indicator
        merged_df = merged_df.drop('_merge', axis=1)
        self.logger.info(f"Después de unir cursos: {merged_df.shape}")

        return merged_df

    def _remove_duplicate_columns(self, combined_df):
        """Elimina columnas duplicadas que pueden haberse creado durante los merges"""
        self.logger.info("=== LIMPIEZA DE COLUMNAS DUPLICADAS ===")
        
        # Lista de columnas duplicadas conocidas a eliminar
        columns_to_remove = []
        
        # Buscar columnas con sufijos que indican duplicación
        for col in combined_df.columns:
            if col.endswith('_estudiantes') or col.endswith('_calificaciones'):
                base_col = col.split('_')[0]
                if base_col in combined_df.columns:
                    columns_to_remove.append(col)
                    self.logger.info(f"Columna duplicada encontrada: {col} (manteniendo {base_col})")
        
        # Eliminar las columnas duplicadas
        if columns_to_remove:
            combined_df = combined_df.drop(columns=columns_to_remove)
            self.logger.info(f"Eliminadas {len(columns_to_remove)} columnas duplicadas: {columns_to_remove}")
        else:
            self.logger.info("No se encontraron columnas duplicadas para eliminar")
            
        return combined_df

    def _calculate_student_age_in_months(self, combined_df):
        """Calcula la edad del estudiante en meses basado en fecha_nacimiento y period"""
        self.logger.info("=== CÁLCULO DE EDAD DE ESTUDIANTES EN MESES ===")
        
        # Inicializar utilidades académicas
        academic_utils = AcademicPeriodUtils()

        # Convertir fecha_nacimiento a datetime (formato YYYY-MM-DD)
        combined_df['fecha_nacimiento'] = pd.to_datetime(combined_df['fecha_nacimiento'], format='%Y-%m-%d', errors='coerce')

        # Función para calcular edad en meses
        def calculate_age_months(row):
            try:
                if pd.isna(row['fecha_nacimiento']) or pd.isna(row['year']) or pd.isna(row['period']):
                    raise Exception(f"Error calculando edad para estudiante: {row}. Las columnas fecha_nacimiento, year y period deben ser no nulas.")
                # Obtener fecha de inicio del periodo usando las utilidades académicas
                period_start_date = academic_utils.get_period_start_date(row)
                
                if pd.isna(period_start_date):
                    raise Exception(f"No se pudo obtener fecha de inicio para año {row['year']}, periodo {row['period']}")
                
                # Calcular diferencia en meses
                months_diff = (period_start_date.year - row['fecha_nacimiento'].year) * 12 + \
                             (period_start_date.month - row['fecha_nacimiento'].month)
                
                # Ajustar si el día del periodo es anterior al día de nacimiento
                if period_start_date.day < row['fecha_nacimiento'].day:
                    months_diff -= 1
                return months_diff

            except Exception as e:
                raise Exception(f"Error calculando edad para estudiante: {e}")

        # Aplicar el cálculo
        combined_df['edad_estudiante'] = combined_df.apply(calculate_age_months, axis=1)
        
        # Reportar estadísticas
        valid_ages = combined_df['edad_estudiante'].dropna()
        if len(valid_ages) > 0:
            self.logger.info(f"Edades calculadas para {len(valid_ages)} estudiantes")
            self.logger.info(f"Edad promedio: {valid_ages.mean():.1f} meses ({valid_ages.mean()/12:.1f} años)")
            self.logger.info(f"Edad mínima: {valid_ages.min()} meses ({valid_ages.min()/12:.1f} años)")
            self.logger.info(f"Edad máxima: {valid_ages.max()} meses ({valid_ages.max()/12:.1f} años)")
        else:
            self.logger.warning("No se pudieron calcular edades para ningún estudiante")
        
        invalid_ages = combined_df['edad_estudiante'].isna().sum()
        if invalid_ages > 0:
            self.logger.warning(f"No se pudo calcular edad para {invalid_ages} registros")
            
        return combined_df

    def _save_dataset(self, combined_df, output_path):
        """Guarda el dataset y muestra información del mismo"""
        # Guardar el dataset
        self.save_to_csv(combined_df, output_path)
        self.logger.info(f"Dataset combinado guardado exitosamente: {combined_df.shape}")

        # Mostrar información del dataset final
        self.logger.info("=== INFORMACIÓN DEL DATASET FINAL ===")
        self.logger.info(f"Forma del dataset: {combined_df.shape}")
        self.logger.info(f"Número de columnas: {len(combined_df.columns)}")
        self.logger.info(f"Número de filas: {len(combined_df)}")
        
        # Verificar valores nulos
        total_nulls = combined_df.isnull().sum().sum()
        self.logger.info(f"Total de valores nulos en el dataset final: {total_nulls}")

    def _get_key_columns_definitions(self):
        """Define las llaves de unión comunes para todos los datasets"""
        return {
            'enrollments': ["documento_identificación", "year", "id_grado", "sede"],
            'student': ["documento_identificación"],
            'student_period': ["documento_identificación", "year", "period"],
            'course': ["sede", "year", "id_grado", "id_asignatura", "period"],
            'student_course_interaction': ["period", "year", "sede", "id_asignatura", "id_grado", "documento_identificación"],
            'teachers': ["id_docente", "year"],
            'sequence_analysis': ["documento_identificación", "id_asignatura", "id_grado", "sede", "year", "period"]
        }

    def _get_datasets_analysis_base(self):
        """Define el análisis base de datasets común - puede ser extendido por subclases"""
        return [
            (self.COLUMNS_STUDENTS_LOGINS, ['documento_identificación', 'year', 'period'], "Logins"),
            (self.COLUMNS_COURSES_BASE, ['sede', 'year', 'id_grado', 'id_asignatura', 'period'], "Cursos base"),
            (self.COLUMNS_TEACHERS_FEATURED, ['id_docente', 'year'], "Teachers featured"),
            (self.COLUMNS_COURSES, ['sede', 'year', 'id_grado', 'id_asignatura', 'period'], "Cursos"),
            (self.COLUMNS_STUDENT_COURSE_INTERACTIONS, ['documento_identificación', 'year', 'id_grado', 'sede', 'id_asignatura', 'period'], "Interacciones"),
            (self.COLUMNS_SEQUENCE_ANALYSIS, ['documento_identificación', 'id_asignatura', 'id_grado', 'sede', 'year', 'period'], "Análisis de secuencias")
        ]

    def _analyze_null_values(self, combined_df):
        """Analiza valores nulos por cada dataset unido"""
        self.logger.info("=== ANÁLISIS DE VALORES NULOS POR DATASET ===")

        # Obtener análisis base y específico de la subclase
        datasets_analysis = self._get_datasets_analysis_specific()

        for all_cols, key_cols, name in datasets_analysis:
            cols_to_check = [col for col in combined_df.columns if col in all_cols and col not in key_cols]
            if cols_to_check:
                null_counts = combined_df[cols_to_check].isnull().sum().sum()
                total_cells = len(combined_df) * len(cols_to_check)
                percentage = null_counts/total_cells*100 if total_cells > 0 else 0
                self.logger.info(f"{name} - Valores nulos: {null_counts}/{total_cells} ({percentage:.2f}%)")

    def _get_datasets_analysis_specific(self):
        """Debe ser implementado por subclases para agregar su análisis específico"""
        raise NotImplementedError("Subclases deben implementar _get_datasets_analysis_specific")

    def _verify_duplicates(self, *datasets):
        """Verifica la presencia de duplicados en todos los datasets - debe ser implementado por subclases"""
        raise NotImplementedError("Subclases deben implementar _verify_duplicates")
