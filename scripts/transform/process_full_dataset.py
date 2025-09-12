import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.dataset_processor_base import DatasetProcessorBase


class FullDatasetProcessor(DatasetProcessorBase):

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


    def _load_and_prepare_grades(self):
        """Carga y prepara el dataset de calificaciones"""
        grades_df = pd.read_csv("data/interim/calificaciones/calificaciones_2024_2025_long.csv")
        grades_df = grades_df[self.COLUMNS_GRADES]
        self.logger.info(f"Dataset de calificaciones cargado: {grades_df.shape}")
        return grades_df


    def _verify_duplicates(self, enrollments_df, students_df, grades_df, student_logins_df, 
                          courses_base_df, courses_df, student_interactions_df, teachers_df, sequence_analysis_df):
        """Verifica la presencia de duplicados en todos los datasets"""
        self.logger.info("=== VERIFICACIÓN DE DUPLICADOS ===")

        # Definir las llaves de unión
        key_columns_enrollments = ["documento_identificación", "year", "id_grado", "sede"]
        key_columns_student = ["documento_identificación"]
        key_columns_grades = ["documento_identificación", "id_asignatura", "id_grado", "period", "year", "sede", "dimensión"]
        key_columns_student_period = ["documento_identificación", "year", "period"]
        key_columns_course = ["sede", "year", "id_grado", "id_asignatura", "period"]
        key_columns_full_renamed = ["period", "year", "sede", "id_asignatura", "id_grado", "documento_identificación"]
        key_columns_teachers = ["id_docente", "year"]
        key_columns_sequence_analysis = ["documento_identificación", "id_asignatura", "id_grado", "sede", "year", "period"]
        # Para teachers verificar duplicados usando todas las columnas
        key_columns_teachers_all = teachers_df.columns.tolist()

        # Lista de datasets y sus llaves correspondientes
        datasets_to_check = [
            (enrollments_df, key_columns_enrollments, "enrollments"),
            (students_df, key_columns_student, "estudiantes"),
            (student_logins_df, key_columns_student_period, "logins"),
            (courses_base_df, key_columns_course, "cursos base"),
            (courses_df, key_columns_course, "cursos"),
            (student_interactions_df, key_columns_full_renamed, "interacciones"),
            (grades_df, key_columns_grades, "calificaciones"),
            (teachers_df, key_columns_teachers_all, "teachers"),
            (sequence_analysis_df, key_columns_sequence_analysis, "análisis de secuencias")
        ]

        # Verificar duplicados en cada dataset
        for df, key_cols, name in datasets_to_check:
            dups = df.duplicated(subset=key_cols).sum()
            if dups > 0:
                self.logger.error(f"Duplicados en {name} por {key_cols}: {dups}")
                raise ValueError(f"Se encontraron {dups} duplicados en el dataset de {name} usando las llaves: {key_cols}")

        self.logger.info("✓ No se encontraron duplicados en ningún dataset")


    def _get_datasets_analysis_specific(self):
        """Retorna el análisis específico para el dataset largo con calificaciones"""
        # Combinar el análisis base con el específico de calificaciones
        base_analysis = self._get_datasets_analysis_base()
        grades_analysis = [(self.COLUMNS_GRADES, ['documento_identificación', 'id_asignatura', 'id_grado', 'period', 'year', 'sede'], "Calificaciones")]
        return grades_analysis + base_analysis

    def process_full_dataset(self):
        """
        Combina todos los datasets usando enrollments.csv como base con INNER JOINS
        1. enrollments.csv (base)
        2. estudiantes_clean.csv (por documento_identificación + sede)
        3. calificaciones (por documento_identificación + sede + year + id_grado, expandirá por períodos)
        4. moodle datasets (por llaves correspondientes)
        5. teachers_merged_final.csv (por id_docente desde courses_base)

        NOTA: Se usan INNER JOINS para evitar valores nulos. Solo se incluyen registros 
        que tengan correspondencia en todos los datasets.
        """
        self.logger.info("Iniciando procesamiento del dataset completo usando INNER JOINS (sin valores nulos)...")

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

        # Iniciar con el dataset base
        self.logger.info("Iniciando combinación de datasets...")
        combined_df = enrollments_df.copy()

        # Obtener definiciones de llaves comunes
        key_cols = self._get_key_columns_definitions()
        key_columns_grades_join = ["documento_identificación", "sede", "year", "id_grado"]

        # Realizar uniones secuenciales
        combined_df = self._merge_with_analysis(combined_df, students_df, key_cols['student'], "estudiantes", how='inner')
        combined_df = self._merge_with_analysis(combined_df, grades_df, key_columns_grades_join, "calificaciones", how='inner')

        # Limpieza y cálculos adicionales después del merge con calificaciones
        combined_df = self._remove_duplicate_columns(combined_df)
        combined_df = self._calculate_student_age_in_months(combined_df)

        self.logger.info("=== UNIONES RESTANTES (INNER JOINS - SOLO REGISTROS CON CORRESPONDENCIA) ===")
        combined_df = self._merge_with_analysis(combined_df, student_logins_df, key_cols['student_period'], "logins de estudiantes", how='inner')
        combined_df = self._merge_with_analysis(combined_df, courses_base_df, key_cols['course'], "cursos base", how='inner')
        combined_df = self._merge_with_analysis(combined_df, teachers_df, key_cols['teachers'], "teachers featured", how='inner')
        combined_df = self._merge_courses_with_analysis(combined_df, courses_df, key_cols['course'], how='inner')
        combined_df = self._merge_with_analysis(combined_df, student_interactions_df, key_cols['student_course_interaction'], "interacciones de estudiantes", how='inner')
        combined_df = self._merge_with_analysis(combined_df, sequence_analysis_df, key_cols['sequence_analysis'], "análisis de secuencias", how='inner')

        # Analizar valores nulos
        self._analyze_null_values(combined_df)
        #Dividir el dataset en 2 datasets:
        # 1. Asignaturas con moodle
        # 2. Asignaturas sin moodle
        # 1. Filtra por asignaturas: 1, 2, 3, 4
        df_moodle = combined_df[combined_df['id_asignatura'].isin([1, 2, 3, 4])]
        output_path = "data/interim/full_dataset_moodle.csv"
        # Guardar dataset
        self._save_dataset(df_moodle, output_path)
        
        # 2. Las demás asignaturas
        df_no_moodle = combined_df[~combined_df['id_asignatura'].isin([1, 2, 3, 4])]
        output_path = "data/interim/full_dataset_no_moodle.csv"
        # Guardar dataset
        self._save_dataset(df_no_moodle, output_path)
        return combined_df


if __name__ == "__main__":
    processor = FullDatasetProcessor()
    processor.process_full_dataset()
    processor.logger.info("Full dataset processed successfully.")
    processor.close()