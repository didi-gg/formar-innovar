import pandas as pd
import os
import sys
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.base_script import BaseScript


class CoursesProcessor(BaseScript):
    @staticmethod
    def _create_module_sequence(df):
        """
        Crea una columna 'sequence' con los module_unique_id separados por comas
        y ordenados por la columna 'order' para cada agrupamiento de curso.

        Para el año 2025, solo incluye módulos de la plataforma Edukrea.

        Args:
            df: DataFrame con las columnas module_unique_id, order, year, platform y las columnas de agrupamiento

        Returns:
            DataFrame con la columna sequence agregada al agrupamiento
        """
        # Crear una copia para evitar modificar el original
        df_copy = df.copy()

        # Filtrar solo módulos de Edukrea para el año 2025
        df_filtered = df_copy[
            (df_copy["year"] != 2025) | 
            ((df_copy["year"] == 2025) & (df_copy["platform"] == "edukrea"))
        ]

        # Ordenar por orden dentro de cada grupo
        df_filtered = df_filtered.sort_values(["id_asignatura", "id_grado", "year", "period", "sede", "order"])

        # Agrupar y crear la secuencia
        sequence_df = (
            df_filtered.groupby(["id_asignatura", "id_grado", "year", "period", "sede"])
            .agg(sequence=("module_unique_id", lambda x: ",".join(x.astype(str))))
            .reset_index()
        )
        return sequence_df

    @staticmethod
    def _categorize_moodle_modules(df, module_col="module_type"):
        categories = {
            "is_evaluation": ["assign", "quiz", "workshop", "lesson", "choice", "feedback", "lti"],
            "is_collaboration": ["forum", "chat", "glossary", "workshop"],
            "is_content": ["resource", "page", "book", "folder", "url", "label", "bootstrapelements", "hvp", "lesson", "lti"],
        }
        for cat_name, module_list in categories.items():
            df[cat_name] = df[module_col].isin(module_list).astype(int)

        return df

    @staticmethod
    def _calculate_module_metrics(df):
        # Create a copy to avoid the SettingWithCopyWarning
        df_copy = df.copy()
        df_copy["was_updated"] = (df_copy["teacher_updated_before_start"] | df_copy["teacher_updated_during_week_planned"]).astype(int)

        agg = (
            df_copy.groupby(["id_asignatura", "id_grado", "year", "period", "sede"])
            .agg(
                num_modules=("course_module_id", "count"),
                num_modules_updated=("was_updated", "sum"),
                num_teacher_views_before_planned_start_date=("teacher_accessed_before_start", "sum"),
                teacher_total_updates=("teacher_total_updates", "sum"),
                teacher_total_views=("teacher_total_views", "sum"),
                student_total_views=("student_total_views", "sum"),
                student_total_interactions=("student_total_interactions", "sum"),
                min_days_since_creation=("days_since_creation", "min"),
                max_days_since_creation=("days_since_creation", "max"),
                avg_days_since_creation=("days_since_creation", "mean"),
                median_days_since_creation=("days_since_creation", "median"),
                avg_days_since_last_update=("days_since_last_update", "mean"),
                median_days_since_last_update=("days_since_last_update", "median"),
            )
            .reset_index()
        )

        # Redondear promedios a 2 decimales
        cols_to_round = [
            "avg_days_since_creation",
            "avg_days_since_last_update",
        ]
        agg[cols_to_round] = agg[cols_to_round].round(2)

        return agg

    @staticmethod
    def _adding_percentages(moodle_summary_df):
        moodle_summary_df["percent_evaluation"] = (moodle_summary_df["count_evaluation"] / moodle_summary_df["num_modules"]).fillna(0).round(2)
        moodle_summary_df["percent_collaboration"] = (moodle_summary_df["count_collaboration"] / moodle_summary_df["num_modules"]).fillna(0).round(2)
        moodle_summary_df["percent_content"] = (moodle_summary_df["count_content"] / moodle_summary_df["num_modules"]).fillna(0).round(2)
        moodle_summary_df["percent_in_english"] = (moodle_summary_df["count_in_english"] / moodle_summary_df["num_modules"]).fillna(0).round(2)
        moodle_summary_df["percent_interactive"] = (moodle_summary_df["count_interactive"] / moodle_summary_df["num_modules"]).fillna(0).round(2)
        moodle_summary_df["percent_updated"] = (moodle_summary_df["num_modules_updated"] / moodle_summary_df["num_modules"]).fillna(0).round(2)

        return moodle_summary_df

    @staticmethod
    def _group_moodle_modules(df):
        # Group by the required columns and aggregate counts
        moodle_summary_df = (
            df.groupby(["id_asignatura", "id_grado", "year", "period", "sede"])
            .agg(
                {
                    "is_evaluation": "sum",
                    "is_collaboration": "sum",
                    "is_content": "sum",
                    "is_in_english": "sum",
                    "is_interactive": "sum",
                }
            )
            .reset_index()
        )

        # Rename columns for better readability
        moodle_summary_df = moodle_summary_df.rename(
            columns={
                "is_evaluation": "count_evaluation",
                "is_collaboration": "count_collaboration",
                "is_content": "count_content",
                "is_in_english": "count_in_english",
                "is_interactive": "count_interactive",
            }
        )

        # Display the resulting DataFrame
        return moodle_summary_df

    @staticmethod
    def _process_student_modules(students_modules_df):
        # 1. Flags por estudiante y curso
        student_summary = (
            students_modules_df.groupby(["documento_identificación", "id_asignatura", "id_grado", "year", "period", "sede"])
            .agg(any_viewed=("has_viewed", "max"), any_participated=("has_participated", "max"))
            .reset_index()
        )

        # 2. Resumen por curso
        course_summary = (
            student_summary.groupby(["id_asignatura", "id_grado", "year", "period", "sede"])
            .agg(
                num_students=("documento_identificación", "count"),
                num_students_viewed=("any_viewed", "sum"),
                num_students_interacted=("any_participated", "sum"),
            )
            .reset_index()
        )

        # 3. Módulos únicos y vistos
        module_summary = (
            students_modules_df.groupby(["id_asignatura", "id_grado", "year", "period", "sede"])
            .agg(
                num_unique_modules=("course_module_id", "nunique"),
                num_modules_viewed=("has_viewed", lambda x: x.sum()),
            )
            .reset_index()
        )

        # 4. Métricas promedio de vistas e interacciones
        metric_cols = [
            "avg_views_per_student",
            "median_views_per_student",
            "avg_interactions_per_student",
            "median_interactions_per_student",
        ]

        avg_metrics = (
            students_modules_df.groupby(
                ["id_asignatura", "id_grado", "year", "period", "sede"]
            )
            .agg(
                avg_views_per_student=("num_views", "mean"),
                median_views_per_student=("num_views", "median"),
                avg_interactions_per_student=("num_interactions", "mean"),
                median_interactions_per_student=("num_interactions", "median"),
            )
            .reset_index()
        )

        # Solo sobre las métricas:
        avg_metrics[metric_cols] = (
            avg_metrics[metric_cols]
            .fillna(0)
            .astype(float)
            .round(2)
        )

        # 5. Módulo menos visto (renombrando para evitar conflictos)
        module_views = (
            students_modules_df[students_modules_df["has_viewed"] == 1]
            .groupby(["id_asignatura", "id_grado", "year", "period", "sede", "course_module_id"])["documento_identificación"]
            .nunique()
            .reset_index(name="num_students_viewed")
        )

        least_viewed_module = (
            module_views.sort_values("num_students_viewed")
            .groupby(["id_asignatura", "id_grado", "year", "period", "sede"])
            .first()
            .reset_index()
            .rename(columns={"course_module_id": "id_least_viewed_module", "num_students_viewed": "students_viewed_least_module"})
        )

        # 6. Módulo más tarde abierto
        opened_late = (
            students_modules_df.groupby(["id_asignatura", "id_grado", "year", "period", "sede", "course_module_id"])["days_from_planned_start"].mean().reset_index()
        )

        most_late_opened = (
            opened_late.sort_values("days_from_planned_start", ascending=False)
            .groupby(["id_asignatura", "id_grado", "year", "period", "sede"])
            .first()
            .reset_index()
            .rename(columns={"course_module_id": "id_most_late_opened_module"})
            .drop(columns=["days_from_planned_start"])
        )

        # 7. Porcentaje de accesos fuera de fecha
        students_modules_df = students_modules_df.copy()
        keys = ["id_asignatura", "id_grado", "year", "period", "sede"]

        # Marcar si el acceso está fuera de fecha
        students_modules_df["out_of_date"] = (
            (students_modules_df["days_from_planned_start"] < 0) |
            (students_modules_df["days_after_end"] > 0)
        ).astype(int)

        # Agrupar por módulo y curso: un módulo está fuera de fecha si al menos un estudiante lo abrió fuera de fecha
        module_out_of_date = (
            students_modules_df.groupby(keys + ["course_module_id"])["out_of_date"]
            .max()  # 1 si algún estudiante lo abrió fuera de fecha
            .reset_index()
        )

        # Calcular el porcentaje sobre el total de módulos del curso
        percent_modules_out_of_date = (
            module_out_of_date.groupby(keys)["out_of_date"]
            .mean()  # proporción de módulos fuera de fecha
            .round(2)
            .reset_index(name="percent_modules_out_of_date")
        )

        # 8. Merge de todas las métricas
        summary_df = (
            course_summary.merge(module_summary, on=["id_asignatura", "id_grado", "year", "period", "sede"], how="left")
            .merge(avg_metrics, on=["id_asignatura", "id_grado", "year", "period", "sede"], how="left")
            .merge(least_viewed_module, on=["id_asignatura", "id_grado", "year", "period", "sede"], how="left")
            .merge(most_late_opened, on=["id_asignatura", "id_grado", "year", "period", "sede"], how="left")
            .merge(percent_modules_out_of_date, on=["id_asignatura", "id_grado", "year", "period", "sede"], how="left")
        )

        # 9. Porcentajes
        summary_df["percent_students_viewed"] = (summary_df["num_students_viewed"] / summary_df["num_students"]).round(2)
        summary_df["percent_students_interacted"] = (summary_df["num_students_interacted"] / summary_df["num_students"]).round(2)
        summary_df["percent_modules_viewed"] = (summary_df["num_modules_viewed"] / summary_df["num_unique_modules"]).round(2)

        return summary_df

    @staticmethod
    def process_df(df, students_modules_moodle):
        modules_df = CoursesProcessor._categorize_moodle_modules(df)
        moodle_summary_df = CoursesProcessor._group_moodle_modules(modules_df)
        temporal_metrics_df = CoursesProcessor._calculate_module_metrics(modules_df)

        # Crear la secuencia de módulos ordenados
        sequence_df = CoursesProcessor._create_module_sequence(df)

        # Merge on course_id consistently
        moodle_summary_df = moodle_summary_df.merge(temporal_metrics_df, on=["id_asignatura", "id_grado", "year", "period", "sede"], how="left")
        moodle_summary_df = CoursesProcessor._adding_percentages(moodle_summary_df)

        # Agregar la secuencia de módulos
        moodle_summary_df = moodle_summary_df.merge(sequence_df, on=["id_asignatura", "id_grado", "year", "period", "sede"], how="left")

        students_course_summary = CoursesProcessor._process_student_modules(students_modules_moodle)
        moodle_summary_df = moodle_summary_df.merge(students_course_summary, on=["id_asignatura", "id_grado", "year", "period", "sede"], how="left")

        # El DataFrame final ahora incluye una columna 'sequence' que contiene
        # los module_unique_id separados por comas y ordenados por la columna 'order'
        return moodle_summary_df

    def process_course_data(self):
        students_modules = pd.read_csv("data/interim/moodle/student_modules.csv")
        modules_df = pd.read_csv("data/interim/moodle/modules_featured.csv")

        moodle_courses = CoursesProcessor.process_df(modules_df, students_modules)

        self.save_to_csv(moodle_courses, "data/interim/moodle/courses.csv")

        self.logger.info("Courses data processed and saved successfully.")


if __name__ == "__main__":
    processor = CoursesProcessor()
    processor.process_course_data()
    processor.logger.info("Courses processed successfully.")
    processor.close()
