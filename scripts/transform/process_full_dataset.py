import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.base_script import BaseScript


class FullDatasetProcessor(BaseScript):

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
    # data/interim/calificaciones/calificaciones_2024_2025_long.csv
    COLUMNS_GRADES = [
        "documento_identificación",
        "id_asignatura",
        "id_grado",
        "period",
        "year",
        "sede",
        "dimensión",
        "resultado", # Variable de interés
        "nivel" # Variable de interés es una clasificación de la variable resultado
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
        "count_jornada_noche"
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
        "days_before_start",
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
        "avg_days_before_start",
        "avg_days_after_end",
        "std_days_before_start",
        "std_days_after_end",
        "min_days_before_start",
        "min_days_after_end",
        "max_days_after_end",
        "max_days_before_start",
        "median_days_before_start",
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
        "mid_week_engagement"
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

    def _load_and_prepare_grades(self):
        """Carga y prepara el dataset de calificaciones"""
        grades_df = pd.read_csv("data/interim/calificaciones/calificaciones_2024_2025_long.csv")
        grades_df = grades_df[self.COLUMNS_GRADES]
        self.logger.info(f"Dataset de calificaciones cargado: {grades_df.shape}")
        return grades_df

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

    def _verify_duplicates(self, enrollments_df, students_df, grades_df, student_logins_df, 
                          courses_base_df, courses_df, student_interactions_df):
        """Verifica la presencia de duplicados en todos los datasets"""
        self.logger.info("=== VERIFICACIÓN DE DUPLICADOS ===")

        # Definir las llaves de unión
        key_columns_enrollments = ["documento_identificación", "year", "id_grado", "sede"]
        key_columns_student = ["documento_identificación"]
        key_columns_grades = ["documento_identificación", "id_asignatura", "id_grado", "period", "year", "sede", "dimensión"]
        key_columns_student_period = ["documento_identificación", "year", "period"]
        key_columns_course = ["sede", "year", "id_grado", "id_asignatura", "period"]
        key_columns_full_renamed = ["period", "year", "sede", "id_asignatura", "id_grado", "documento_identificación"]

        # Lista de datasets y sus llaves correspondientes
        datasets_to_check = [
            (enrollments_df, key_columns_enrollments, "enrollments"),
            (students_df, key_columns_student, "estudiantes"),
            (student_logins_df, key_columns_student_period, "logins"),
            (courses_base_df, key_columns_course, "cursos base"),
            (courses_df, key_columns_course, "cursos"),
            (student_interactions_df, key_columns_full_renamed, "interacciones"),
            (grades_df, key_columns_grades, "calificaciones")
        ]

        # Verificar duplicados en cada dataset
        for df, key_cols, name in datasets_to_check:
            dups = df.duplicated(subset=key_cols).sum()
            if dups > 0:
                self.logger.error(f"Duplicados en {name} por {key_cols}: {dups}")
                raise ValueError(f"Se encontraron {dups} duplicados en el dataset de {name} usando las llaves: {key_cols}")

        self.logger.info("✓ No se encontraron duplicados en ningún dataset")

    def _merge_with_analysis(self, left_df, right_df, merge_keys, dataset_name):
        """Realiza un merge con análisis detallado de los resultados"""
        self.logger.info(f"=== UNIÓN CON {dataset_name.upper()} ===")

        merged_df = left_df.merge(
            right_df, 
            on=merge_keys, 
            how='left',
            suffixes=('', f'_{dataset_name.lower().replace(" ", "_")}'),
            indicator=True
        )

        # Analizar matches
        match_counts = merged_df['_merge'].value_counts()
        self.logger.info(f"Resultados del join con {dataset_name}:")
        self.logger.info(f"  - Registros que hicieron match: {match_counts.get('both', 0)}")
        self.logger.info(f"  - Registros sin match (solo en left): {match_counts.get('left_only', 0)}")

        # Mostrar ejemplos de registros sin match
        no_match = merged_df[merged_df['_merge'] == 'left_only']
        if len(no_match) > 0:
            self.logger.warning(f"Registros sin match en {dataset_name} (mostrando primeros 10):")
            sample_keys = no_match[merge_keys].drop_duplicates().head(10)
            for _, row in sample_keys.iterrows():
                self.logger.warning(f"  - {dict(row)}")

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

    def _merge_courses_with_analysis(self, combined_df, courses_df, merge_keys):
        """Realiza merge con cursos incluyendo análisis detallado"""
        self.logger.info("=== UNIÓN CON CURSOS ===")

        merged_df = combined_df.merge(
            courses_df, 
            on=merge_keys, 
            how='left',
            suffixes=('', '_course'),
            indicator=True
        )

        # Analizar matches
        match_counts = merged_df['_merge'].value_counts()
        self.logger.info(f"Resultados del join con cursos:")
        self.logger.info(f"  - Registros que hicieron match: {match_counts.get('both', 0)}")
        self.logger.info(f"  - Registros sin match (solo en combined): {match_counts.get('left_only', 0)}")

        # Análisis detallado si hay registros sin match
        self._detailed_course_analysis(merged_df, courses_df, merge_keys)

        # Eliminar columna indicator
        merged_df = merged_df.drop('_merge', axis=1)
        self.logger.info(f"Después de unir cursos: {merged_df.shape}")

        return merged_df

    def _analyze_null_values(self, combined_df):
        """Analiza valores nulos por cada dataset unido"""
        self.logger.info("=== ANÁLISIS DE VALORES NULOS POR DATASET ===")

        # Definir columnas por dataset (excluyendo llaves)
        datasets_analysis = [
            (self.COLUMNS_GRADES, ['documento_identificación', 'id_asignatura', 'id_grado', 'period', 'year', 'sede'], "Calificaciones"),
            (self.COLUMNS_STUDENTS_LOGINS, ['documento_identificación', 'year', 'period'], "Logins"),
            (self.COLUMNS_COURSES_BASE, ['sede', 'year', 'id_grado', 'id_asignatura', 'period'], "Cursos base"),
            (self.COLUMNS_COURSES, ['sede', 'year', 'id_grado', 'id_asignatura', 'period'], "Cursos"),
            (self.COLUMNS_STUDENT_COURSE_INTERACTIONS, ['documento_identificación', 'year', 'id_grado', 'sede', 'id_asignatura', 'period'], "Interacciones")
        ]

        for all_cols, key_cols, name in datasets_analysis:
            cols_to_check = [col for col in combined_df.columns if col in all_cols and col not in key_cols]
            if cols_to_check:
                null_counts = combined_df[cols_to_check].isnull().sum().sum()
                total_cells = len(combined_df) * len(cols_to_check)
                percentage = null_counts/total_cells*100 if total_cells > 0 else 0
                self.logger.info(f"{name} - Valores nulos: {null_counts}/{total_cells} ({percentage:.2f}%)")

    def _save_dataset(self, combined_df):
        """Guarda el dataset combinado y muestra información final"""
        output_path = "data/interim/full_dataset_combined.csv"

        # Guardar el dataset
        combined_df.to_csv(output_path, index=False)
        self.logger.info(f"Dataset combinado guardado exitosamente: {combined_df.shape}")

        # Mostrar información del dataset final
        self.logger.info("=== INFORMACIÓN DEL DATASET FINAL ===")
        self.logger.info(f"Forma del dataset: {combined_df.shape}")
        self.logger.info(f"Número de columnas: {len(combined_df.columns)}")
        self.logger.info(f"Número de filas: {len(combined_df)}")

    def process_full_dataset(self):
        """
        Combina todos los datasets usando enrollments.csv como base
        1. enrollments.csv (base)
        2. estudiantes_clean.csv (por documento_identificación + sede)
        3. calificaciones (por documento_identificación + sede + year + id_grado, expandirá por períodos)
        4. moodle datasets (por llaves correspondientes)
        """
        self.logger.info("Iniciando procesamiento del dataset completo...")

        # Cargar todos los datasets
        enrollments_df = self._load_and_prepare_enrollments()
        students_df = self._load_and_prepare_students()
        grades_df = self._load_and_prepare_grades()
        student_logins_df = self._load_and_prepare_student_logins()
        courses_base_df = self._load_and_prepare_courses_base()
        courses_df = self._load_and_prepare_courses()
        student_interactions_df = self._load_and_prepare_student_interactions()

        # Verificar duplicados
        self._verify_duplicates(enrollments_df, students_df, grades_df, student_logins_df, 
                               courses_base_df, courses_df, student_interactions_df)

        # Iniciar con el dataset base
        self.logger.info("Iniciando combinación de datasets...")
        combined_df = enrollments_df.copy()

        # Definir llaves de unión
        key_columns_student = ["documento_identificación"]
        key_columns_grades_join = ["documento_identificación", "sede", "year", "id_grado"]
        key_columns_student_period = ["documento_identificación", "year", "period"]
        key_columns_course = ["sede", "year", "id_grado", "id_asignatura", "period"]
        key_columns_full_renamed = ["period", "year", "sede", "id_asignatura", "id_grado", "documento_identificación"]

        # Realizar uniones secuenciales
        combined_df = self._merge_with_analysis(combined_df, students_df, key_columns_student, "estudiantes")
        combined_df = self._merge_with_analysis(combined_df, grades_df, key_columns_grades_join, "calificaciones")

        self.logger.info("=== UNIONES RESTANTES (SIN FALTANTES ESPERADOS) ===")
        combined_df = self._merge_with_analysis(combined_df, student_logins_df, key_columns_student_period, "logins de estudiantes")
        combined_df = self._merge_with_analysis(combined_df, courses_base_df, key_columns_course, "cursos base")
        combined_df = self._merge_courses_with_analysis(combined_df, courses_df, key_columns_course)
        combined_df = self._merge_with_analysis(combined_df, student_interactions_df, key_columns_full_renamed, "interacciones de estudiantes")

        # Analizar valores nulos
        self._analyze_null_values(combined_df)

        # Guardar dataset
        self._save_dataset(combined_df)

        return combined_df


if __name__ == "__main__":
    processor = FullDatasetProcessor()
    processor.process_full_dataset()
    processor.logger.info("Full dataset processed successfully.")
    processor.close()