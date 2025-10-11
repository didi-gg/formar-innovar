import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.dataset_processor_base import DatasetProcessorBase


class FullShortDatasetProcessor(DatasetProcessorBase):

    def _verify_duplicates(self, enrollments_df, students_df, grades_df, student_logins_df, 
                          courses_base_df, courses_df, student_interactions_df, teachers_df, sequence_analysis_df):
        """Verifica la presencia de duplicados en todos los datasets"""
        self.logger.info("=== VERIFICACIÓN DE DUPLICADOS ===")

        # Obtener definiciones de llaves comunes
        key_cols = self._get_key_columns_definitions()
        key_columns_grades = ["documento_identificación", "id_asignatura", "id_grado", "period", "year", "sede"]
        # Para teachers verificar duplicados usando todas las columnas
        key_columns_teachers_all = teachers_df.columns.tolist()

        # Lista de datasets y sus llaves correspondientes
        datasets_to_check = [
            (enrollments_df, key_cols['enrollments'], "enrollments"),
            (students_df, key_cols['student'], "estudiantes"),
            (student_logins_df, key_cols['student_period'], "logins"),
            (courses_base_df, key_cols['course'], "cursos base"),
            (courses_df, key_cols['course'], "cursos"),
            (student_interactions_df, key_cols['student_course_interaction'], "interacciones"),
            (grades_df, key_columns_grades, "calificaciones"),
            (teachers_df, key_columns_teachers_all, "teachers"),
            (sequence_analysis_df, key_cols['sequence_analysis'], "análisis de secuencias")
        ]

        # Verificar duplicados en cada dataset
        for df, key_cols, name in datasets_to_check:
            dups = df.duplicated(subset=key_cols).sum()
            if dups > 0:
                self.logger.error(f"Duplicados en {name} por {key_cols}: {dups}")
                raise ValueError(f"Se encontraron {dups} duplicados en el dataset de {name} usando las llaves: {key_cols}")

        self.logger.info("✓ No se encontraron duplicados en ningún dataset")

    def _get_datasets_analysis_specific(self):
        """Retorna el análisis específico para el dataset corto con calificaciones"""
        # Combinar el análisis base con el específico de calificaciones
        base_analysis = self._get_datasets_analysis_base()
        grades_analysis = [(self.COLUMNS_GRADES, ['documento_identificación', 'id_asignatura', 'id_grado', 'period', 'year', 'sede'], "Calificaciones")]
        return grades_analysis + base_analysis

    def _process_moodle_subjects(self, combined_df, student_logins_df, courses_base_df, courses_df, 
                                student_interactions_df, teachers_df, sequence_analysis_df):
        """Procesa asignaturas con Moodle usando LEFT JOIN para mantener todos los registros"""
        self.logger.info("=== PROCESANDO ASIGNATURAS CON MOODLE (LEFT JOINS) ===")

        # Filtrar solo asignaturas Moodle (1, 2, 3, 4)
        moodle_df = combined_df[combined_df['id_asignatura'].isin([1, 2, 3, 4])].copy()
        self.logger.info(f"Registros de asignaturas Moodle: {moodle_df.shape}")

        # Obtener definiciones de llaves comunes
        key_cols = self._get_key_columns_definitions()

        # Realizar uniones con LEFT JOIN
        moodle_df = self._merge_with_analysis(moodle_df, student_logins_df, key_cols['student_period'], "logins de estudiantes", how='left')
        moodle_df = self._merge_with_analysis(moodle_df, courses_base_df, key_cols['course'], "cursos base", how='left')
        moodle_df = self._merge_with_analysis(moodle_df, teachers_df, key_cols['teachers'], "teachers featured", how='left')
        moodle_df = self._merge_courses_with_analysis(moodle_df, courses_df, key_cols['course'], how='left')
        moodle_df = self._merge_with_analysis(moodle_df, student_interactions_df, key_cols['student_course_interaction'], "interacciones de estudiantes", how='left')
        moodle_df = self._merge_with_analysis(moodle_df, sequence_analysis_df, key_cols['sequence_analysis'], "análisis de secuencias", how='inner')

        # Guardar dataset Moodle
        output_path = "data/processed/full_short_dataset_moodle.csv"
        self._save_dataset(moodle_df, output_path)
        return moodle_df

    def _process_non_moodle_subjects(self, combined_df, courses_base_df, teachers_df):
        """Procesa asignaturas sin Moodle incluyendo información de teachers.
        Incluye: courses_base (para id_docente) y teachers_featured
        Excluye datos específicos de Moodle: student_logins, student_interactions, sequence_analysis, courses"""
        self.logger.info("=== PROCESANDO ASIGNATURAS SIN MOODLE (LEFT JOINS) ===")

        # Filtrar asignaturas no Moodle (todas excepto 1, 2, 3, 4)
        non_moodle_df = combined_df[~combined_df['id_asignatura'].isin([1, 2, 3, 4])].copy()
        self.logger.info(f"Registros de asignaturas no Moodle: {non_moodle_df.shape}")

        # Obtener definiciones de llaves comunes
        key_cols = self._get_key_columns_definitions()

        # Para asignaturas no Moodle, incluimos solo courses_base para obtener id_docente y luego teachers
        # Excluimos datasets específicos de Moodle: student_logins_df, student_interactions_df, 
        # sequence_analysis_df, courses_df
        non_moodle_df = self._merge_with_analysis(non_moodle_df, courses_base_df, key_cols['course'], "cursos base", how='left')
        non_moodle_df = self._merge_with_analysis(non_moodle_df, teachers_df, key_cols['teachers'], "teachers featured", how='left')

        self.logger.info("Nota: Para asignaturas no Moodle se incluyen courses_base y teachers, pero se excluyen logins, interacciones, análisis de secuencias y courses")

        # Guardar dataset no Moodle
        output_path = "data/processed/full_short_dataset_no_moodle.csv"
        self._save_dataset(non_moodle_df, output_path)
        return non_moodle_df

    def process_full_dataset(self):
        """
        Combina todos los datasets usando enrollments.csv como base
        Estrategia de uniones:
        - Dataset base: INNER JOINS (enrollments + estudiantes + calificaciones)
        - Datos adicionales: LEFT JOINS para mantener todos los registros y agregar información cuando exista

        1. enrollments.csv (base)
        2. estudiantes_clean.csv (por documento_identificación + sede) - INNER JOIN
        3. calificaciones (por documento_identificación + sede + year + id_grado, expandirá por períodos) - INNER JOIN
        4. Datos adicionales (logins, courses, teachers, interactions, etc.) - LEFT JOINS
        """
        self.logger.info("Iniciando procesamiento del dataset completo con LEFT JOINS para datos adicionales...")

        # Cargar todos los datasets
        enrollments_df = self._load_and_prepare_enrollments()
        students_df = self._load_and_prepare_students()
        grades_df = self._load_and_prepare_grades()
        student_logins_df = self._load_and_prepare_student_logins()
        courses_base_df = self._load_and_prepare_courses_base()
        courses_df = self._load_and_prepare_courses()
        student_interactions_df = self._load_and_prepare_student_interactions()
        teachers_df = self._load_and_prepare_teachers_featured()
        sequence_analysis_df = self._load_and_prepare_sequence_analysis()

        # Verificar duplicados
        self._verify_duplicates(enrollments_df, students_df, grades_df, student_logins_df, 
                               courses_base_df, courses_df, student_interactions_df, teachers_df, sequence_analysis_df)

        # Crear dataset base común (enrollments + estudiantes + calificaciones)
        self.logger.info("=== CREANDO DATASET BASE COMÚN ===")
        combined_df = enrollments_df.copy()

        # Obtener definiciones de llaves comunes
        key_cols = self._get_key_columns_definitions()
        key_columns_grades_join = ["documento_identificación", "sede", "year", "id_grado"]

        # Realizar uniones base con INNER JOIN
        combined_df = self._merge_with_analysis(combined_df, students_df, key_cols['student'], "estudiantes", how='inner')
        combined_df = self._merge_with_analysis(combined_df, grades_df, key_columns_grades_join, "calificaciones", how='inner')

        # Limpieza y cálculos adicionales después del merge con calificaciones
        combined_df = self._remove_duplicate_columns(combined_df)
        combined_df = self._calculate_student_age_in_months(combined_df)

        # Procesamiento diferenciado por tipo de asignatura
        df_moodle = self._process_moodle_subjects(
            combined_df, student_logins_df, courses_base_df, courses_df, 
            student_interactions_df, teachers_df, sequence_analysis_df
        )

        # Procesar asignaturas no Moodle con LEFT JOIN
        df_no_moodle = self._process_non_moodle_subjects(
            combined_df, courses_base_df, teachers_df
        )

        # Combinar ambos datasets
        self.logger.info("=== COMBINANDO DATASETS MOODLE Y NO MOODLE ===")
        combined_full_df = pd.concat([df_moodle, df_no_moodle], ignore_index=True)
        self.logger.info(f"Dataset completo combinado: {combined_full_df.shape}")

        # Guardar dataset completo combinado
        output_path = "data/processed/full_short_dataset.csv"
        self._save_dataset(combined_full_df, output_path)

        # Analizar valores nulos en ambos datasets individuales y el combinado
        self.logger.info("=== ANÁLISIS FINAL DE VALORES NULOS ===")
        self.logger.info("Dataset Moodle:")
        self._analyze_null_values(df_moodle)
        self.logger.info("Dataset no Moodle:")
        self._analyze_null_values(df_no_moodle)
        self.logger.info("Dataset completo combinado:")
        self._analyze_null_values(combined_full_df)

        # Retornar dataset combinado
        return combined_full_df


if __name__ == "__main__":
    processor = FullShortDatasetProcessor()
    processor.process_full_dataset()
    processor.logger.info("Full dataset processed successfully.")
    processor.close()