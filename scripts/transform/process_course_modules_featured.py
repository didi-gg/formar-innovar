import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.moodle_path_resolver import MoodlePathResolver
from utils.base_script import BaseScript


class ModuleFeaturesProcessor(BaseScript):
    def _get_logs_updated(self, logs_file, year):
        sql_logs = f"""
            SELECT 
                '{year}' AS year,
                contextinstanceid AS course_module_id,
                MAX(to_timestamp(timecreated)) AS fecha_ultima_actualizacion,
                COUNT(*) AS total_actualizaciones_docente
            FROM '{logs_file}'
            WHERE eventname LIKE '%updated%'
            GROUP BY contextinstanceid
        """
        return self.con.execute(sql_logs).df()

    def merge_modules_logs_update(self, modules_df, logs_df):
        # 2. Unir módulos con fecha de última actualización
        modules_df["year"] = modules_df["year"].astype(int)
        logs_df["year"] = logs_df["year"].astype(int)
        modules_df = modules_df.merge(logs_df, on=["course_module_id", "year"], how="left")

        # 3. Calcular días desde creación y última actualización
        modules_df["module_creation_date"] = pd.to_datetime(modules_df["module_creation_date"]).dt.tz_localize(None)
        modules_df["fecha_ultima_actualizacion"] = pd.to_datetime(modules_df["fecha_ultima_actualizacion"]).dt.tz_localize(None)
        modules_df["planned_start_date"] = pd.to_datetime(modules_df["planned_start_date"]).dt.tz_localize(None)

        # Resta de fechas segura
        modules_df["dias_desde_creacion"] = (modules_df["planned_start_date"] - modules_df["module_creation_date"]).dt.days
        modules_df["dias_desde_ultima_actualizacion"] = (modules_df["planned_start_date"] - modules_df["fecha_ultima_actualizacion"]).dt.days
        return modules_df

    def _get_vistas_docente(self, logs_file):
        # 1. Cargar logs docentes
        logs_docentes = pd.read_csv(logs_file)

        # 2. Convertir columnas de tiempo
        logs_docentes["timecreated"] = pd.to_datetime(logs_docentes["timecreated"], unit="s", errors="coerce")

        # 3. Filtrar eventos de vista docente
        vistas_docentes = logs_docentes[logs_docentes["eventname"].str.contains("viewed", case=False, na=False)]

        # 4. Agrupar por módulo y año (usando columna 'year' existente)
        vistas_agg = (
            vistas_docentes.groupby(["contextinstanceid", "year"])
            .agg(
                total_vistas_docente=("contextinstanceid", "count"),
                fecha_primera_vista=("timecreated", "min"),
                fecha_ultima_vista=("timecreated", "max"),
            )
            .reset_index()
            .rename(columns={"contextinstanceid": "course_module_id"})
        )

        return vistas_agg

    def _get_students_count(self, students_df):
        students_df["year"] = students_df["year"].astype(int)
        students_df["course_id"] = students_df["course_id"].astype(int)
        student_counts = students_df.groupby(["year", "course_id"]).size().reset_index(name="total_estudiantes")
        return student_counts

    def _get_student_interactions(self, student_logs_file):
        """
        Carga los logs de estudiantes y calcula métricas de interacción por módulo y año.
        """
        logs = pd.read_csv(student_logs_file)
        logs["timecreated"] = pd.to_datetime(logs["timecreated"], unit="s", errors="coerce")

        # Validación: asegurarse de que 'year' ya existe
        if "year" not in logs.columns:
            raise ValueError("La columna 'year' no está presente en los logs de estudiantes.")

        # Asegura que `contextinstanceid` sea entero para merge posterior
        logs["contextinstanceid"] = logs["contextinstanceid"].astype("Int64")

        # Separar eventos viewed vs interactivos
        logs["is_viewed"] = logs["eventname"].str.contains("viewed", case=False, na=False)

        # Total vistas por módulo y año
        total_views = (
            logs[logs["is_viewed"]]
            .groupby(["contextinstanceid", "year"])
            .agg(
                total_vistas_estudiantes=("id", "count"),
                estudiantes_que_vieron=("userid", pd.Series.nunique),
            )
        )

        # Total interacciones por módulo y año (no viewed)
        interacciones = (
            logs[~logs["is_viewed"]]
            .groupby(["contextinstanceid", "year"])
            .agg(
                total_interacciones_estudiantes=("id", "count"),
                estudiantes_que_interactuaron=("userid", pd.Series.nunique),
            )
        )

        # Vistas por estudiante por módulo y año
        vistas_por_estudiante = (
            logs[logs["is_viewed"]]
            .groupby(["contextinstanceid", "year", "userid"])
            .size()
            .groupby(["contextinstanceid", "year"])
            .agg(
                min_vistas_estudiante="min",
                max_vistas_estudiante="max",
                mediana_vistas_estudiante="median",
            )
        )

        # Combinar todas las métricas
        resumen = (
            total_views.join(interacciones, how="outer")
            .join(vistas_por_estudiante, how="outer")
            .reset_index()
            .rename(columns={"contextinstanceid": "course_module_id"})
        )

        # Rellenar NaNs
        resumen.fillna(0, inplace=True)

        # Convertir a enteros donde aplique
        int_cols = [
            "total_vistas_estudiantes",
            "estudiantes_que_vieron",
            "total_interacciones_estudiantes",
            "estudiantes_que_interactuaron",
            "min_vistas_estudiante",
            "max_vistas_estudiante",
        ]
        for col in int_cols:
            resumen[col] = resumen[col].astype(int)

        return resumen

    def _add_metrics(self, df):
        df["porcentaje_estudiantes_que_interactuaron"] = (df["estudiantes_que_interactuaron"] / df["total_estudiantes"]).fillna(0) * 100
        df["porcentaje_estudiantes_que_vieron"] = (df["estudiantes_que_vieron"] / df["total_estudiantes"]).fillna(0) * 100
        df["ratio_interaccion_vs_vista"] = (df["total_interacciones_estudiantes"] / df["total_vistas_estudiantes"]).fillna(0)
        df["docente_activo"] = ((df["total_vistas_docente"] > 0) & (df["total_actualizaciones_docente"] > 0)).astype(int)

    def process_df(self, df, logs_df, students_courses_df, teacher_logs_file, students_log_file):
        df = self.merge_modules_logs_update(df, logs_df)

        # Agregar vistas docentes edukrea
        vistas_agg = self._get_vistas_docente(teacher_logs_file)
        df = df.merge(vistas_agg, on=["course_module_id", "year"], how="left")

        # Calcular si accedió antes
        df["accedio_antes"] = df["fecha_primera_vista"] < df["planned_start_date"]

        df["total_actualizaciones_docente"] = df["total_actualizaciones_docente"].fillna(0).astype(int)

        df["total_vistas_docente"] = df["total_vistas_docente"].fillna(0).astype(int)

        # Agregar el conteo de estudiantes por curso
        students_count = self._get_students_count(students_courses_df)
        df = df.merge(students_count, on=["year", "course_id"], how="left")

        df["total_estudiantes"] = df["total_estudiantes"].fillna(0).astype(int)

        # Agregar interacciones de estudiantes
        interacciones_estudiantes_edukrea = self._get_student_interactions(students_log_file)
        df = df.merge(interacciones_estudiantes_edukrea, on=["course_module_id", "year"], how="left")

        # Rellenar NaNs y asegurar tipos
        metrics_fill_zero = [
            "total_vistas_estudiantes",
            "estudiantes_que_vieron",
            "total_interacciones_estudiantes",
            "estudiantes_que_interactuaron",
            "min_vistas_estudiante",
            "max_vistas_estudiante",
        ]
        for col in metrics_fill_zero:
            df[col] = df[col].fillna(0).astype(int)
        df["mediana_vistas_estudiante"] = df["mediana_vistas_estudiante"].fillna(0).astype(int)

        return df

    def process_course_data(self):
        moodle_df = pd.read_csv("data/interim/moodle/modules_active_moodle.csv")
        edukrea_df = pd.read_csv("data/interim/moodle/modules_active_edukrea.csv")

        students_courses_moodle = pd.read_csv("data/interim/moodle/student_moodle_courses.csv")
        students_courses_edukrea = pd.read_csv("data/interim/moodle/student_edukrea_courses.csv")

        logs_table = "logstore_standard_log"

        # Get logs for 2024
        logs_parquet_2024 = MoodlePathResolver.get_paths(2024, logs_table)[0]
        logs_parquet_2025 = MoodlePathResolver.get_paths(2025, logs_table)[0]
        logs_edukrea = MoodlePathResolver.get_paths("Edukrea", logs_table)[0]

        logs_2024 = self._get_logs_updated(logs_parquet_2024, 2024)
        logs_2025 = self._get_logs_updated(logs_parquet_2025, 2025)
        logs_edukrea = self._get_logs_updated(logs_edukrea, 2025)
        logs_2024_2025 = pd.concat([logs_2024, logs_2025], ignore_index=True)

        edukrea_df = self.process_df(
            edukrea_df,
            logs_edukrea,
            students_courses_edukrea,
            "data/interim/moodle/teacher_logs_edukrea.csv",
            "data/interim/moodle/student_logs_edukrea.csv",
        )

        moodle_df = self.process_df(
            moodle_df,
            logs_2024_2025,
            students_courses_moodle,
            "data/interim/moodle/teacher_logs_moodle.csv",
            "data/interim/moodle/student_logs_moodle.csv",
        )

        # Finally, save the Edukrea data to CSV
        output_file = "data/interim/moodle/modules_moodle_featured.csv"
        self.save_to_csv(moodle_df, output_file)

        edukrea_output_file = "data/interim/moodle/modules_edukrea_featured.csv"
        self.save_to_csv(edukrea_df, edukrea_output_file)


if __name__ == "__main__":
    processor = ModuleFeaturesProcessor()
    processor.process_course_data()
    processor.close()
    processor.logger.info("Course modules processed successfully.")
