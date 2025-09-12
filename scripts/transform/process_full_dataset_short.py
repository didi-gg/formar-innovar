import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.dataset_processor_base import DatasetProcessorBase


class FullShortDatasetProcessor(DatasetProcessorBase):

    # data/interim/calificaciones/calificaciones_2024_2025_short.csv
    COLUMNS_GRADES = [
        "documento_identificación",
        "id_asignatura",
        "id_grado",
        "period",
        "year",
        "sede",
        "cog",
        "act",
        "axi",
        "proc",
        "nota_final", # Variable de interés
        "nivel" # Variable de interés es una clasificación de la variable resultado
    ]

    def _load_and_prepare_grades(self):
        """Carga y prepara el dataset de calificaciones"""
        grades_df = pd.read_csv("data/interim/calificaciones/calificaciones_2024_2025_short.csv")
        grades_df = grades_df[self.COLUMNS_GRADES]
        self.logger.info(f"Dataset de calificaciones cargado: {grades_df.shape}")
        return grades_df

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
        """Procesa asignaturas con Moodle usando INNER JOIN para dataset completo sin nulos"""
        self.logger.info("=== PROCESANDO ASIGNATURAS CON MOODLE (INNER JOINS) ===")

        # Filtrar solo asignaturas Moodle (1, 2, 3, 4)
        moodle_df = combined_df[combined_df['id_asignatura'].isin([1, 2, 3, 4])].copy()
        self.logger.info(f"Registros de asignaturas Moodle: {moodle_df.shape}")

        # Obtener definiciones de llaves comunes
        key_cols = self._get_key_columns_definitions()

        # Realizar uniones con INNER JOIN
        moodle_df = self._merge_with_analysis(moodle_df, student_logins_df, key_cols['student_period'], "logins de estudiantes", how='inner')
        moodle_df = self._merge_with_analysis(moodle_df, courses_base_df, key_cols['course'], "cursos base", how='inner')
        moodle_df = self._merge_with_analysis(moodle_df, teachers_df, key_cols['teachers'], "teachers featured", how='inner')
        moodle_df = self._merge_courses_with_analysis(moodle_df, courses_df, key_cols['course'], how='inner')
        moodle_df = self._merge_with_analysis(moodle_df, student_interactions_df, key_cols['student_course_interaction'], "interacciones de estudiantes", how='inner')
        moodle_df = self._merge_with_analysis(moodle_df, sequence_analysis_df, key_cols['sequence_analysis'], "análisis de secuencias", how='inner')

        # Guardar dataset Moodle
        output_path = "data/interim/full_short_dataset_moodle.csv"
        self._save_dataset(moodle_df, output_path)
        return moodle_df
    
    def _process_non_moodle_subjects(self, combined_df, student_logins_df, courses_base_df, courses_df, 
                                    student_interactions_df, teachers_df, sequence_analysis_df):
        """Procesa asignaturas sin Moodle usando LEFT JOIN para mantener todos los registros"""
        self.logger.info("=== PROCESANDO ASIGNATURAS SIN MOODLE (LEFT JOINS) ===")

        # Filtrar asignaturas no Moodle (todas excepto 1, 2, 3, 4)
        non_moodle_df = combined_df[~combined_df['id_asignatura'].isin([1, 2, 3, 4])].copy()
        self.logger.info(f"Registros de asignaturas no Moodle: {non_moodle_df.shape}")
        
        # Obtener definiciones de llaves comunes
        key_cols = self._get_key_columns_definitions()

        # Realizar uniones con LEFT JOIN para mantener todos los registros
        non_moodle_df = self._merge_with_analysis(non_moodle_df, student_logins_df, key_cols['student_period'], "logins de estudiantes", how='left')
        non_moodle_df = self._merge_with_analysis(non_moodle_df, courses_base_df, key_cols['course'], "cursos base", how='left')
        non_moodle_df = self._merge_with_analysis(non_moodle_df, teachers_df, key_cols['teachers'], "teachers featured", how='left')
        non_moodle_df = self._merge_courses_with_analysis(non_moodle_df, courses_df, key_cols['course'], how='left')
        non_moodle_df = self._merge_with_analysis(non_moodle_df, student_interactions_df, key_cols['student_course_interaction'], "interacciones de estudiantes", how='left')
        non_moodle_df = self._merge_with_analysis(non_moodle_df, sequence_analysis_df, key_cols['sequence_analysis'], "análisis de secuencias", how='left')

        # Guardar dataset no Moodle
        output_path = "data/interim/full_short_dataset_no_moodle.csv"
        self._save_dataset(non_moodle_df, output_path)
        return non_moodle_df

    def process_full_dataset(self):
        """
        Combina todos los datasets usando enrollments.csv como base
        Procesamiento diferenciado:
        - Asignaturas Moodle (1,2,3,4): INNER JOINS para dataset completo sin nulos
        - Asignaturas no Moodle: LEFT JOINS para mantener todos los registros

        1. enrollments.csv (base)
        2. estudiantes_clean.csv (por documento_identificación + sede)
        3. calificaciones (por documento_identificación + sede + year + id_grado, expandirá por períodos)
        4. Procesamiento diferenciado por tipo de asignatura
        """
        self.logger.info("Iniciando procesamiento del dataset completo con estrategia diferenciada...")

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
        self.logger.info("=== PROCESAMIENTO DIFERENCIADO POR TIPO DE ASIGNATURA ===")
        
        # Procesar asignaturas Moodle con INNER JOIN
        df_moodle = self._process_moodle_subjects(
            combined_df, student_logins_df, courses_base_df, courses_df, 
            student_interactions_df, teachers_df, sequence_analysis_df
        )
        
        # Procesar asignaturas no Moodle con LEFT JOIN
        df_no_moodle = self._process_non_moodle_subjects(
            combined_df, student_logins_df, courses_base_df, courses_df, 
            student_interactions_df, teachers_df, sequence_analysis_df
        )

        # Analizar valores nulos en ambos datasets
        self.logger.info("=== ANÁLISIS FINAL DE VALORES NULOS ===")
        self.logger.info("Dataset Moodle:")
        self._analyze_null_values(df_moodle)
        self.logger.info("Dataset no Moodle:")
        self._analyze_null_values(df_no_moodle)
        
        # Retornar dataset combinado para compatibilidad
        return pd.concat([df_moodle, df_no_moodle], ignore_index=True)


if __name__ == "__main__":
    processor = FullShortDatasetProcessor()
    processor.process_full_dataset()
    processor.logger.info("Full dataset processed successfully.")
    processor.close()