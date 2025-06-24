import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.moodle_path_resolver import MoodlePathResolver
from utils.base_script import BaseScript


class ModuleFeaturesProcessor(BaseScript):
    # ---------- LOGS DOCENTE ----------
    def _get_logs_updated(self, logs_file, year):
        query = f"""
            SELECT 
                '{year}' AS year,
                contextinstanceid AS course_module_id,
                MAX(to_timestamp(timecreated)) AS last_update_date,
                COUNT(*) AS teacher_total_updates
            FROM '{logs_file}'
            WHERE eventname LIKE '%updated%'
            GROUP BY contextinstanceid
        """
        return self.con.execute(query).df()

    def _get_vistas_docente(self, logs_file):
        logs = pd.read_csv(logs_file)
        logs["timecreated"] = pd.to_datetime(logs["timecreated"], unit="s", errors="coerce")

        vistas = logs[logs["eventname"].str.contains("view", case=False, na=False)]
        vistas_agg = (
            vistas.groupby(["contextinstanceid", "year"])
            .agg(
                teacher_total_views=("contextinstanceid", "count"),
                teacher_first_view_date=("timecreated", "min"),
                teacher_last_view_date=("timecreated", "max"),
            )
            .reset_index()
            .rename(columns={"contextinstanceid": "course_module_id"})
        )
        return vistas_agg

    # ---------- LOGS ESTUDIANTES ----------
    def _get_students_count(self, students_df):
        students_df["year"] = students_df["year"].astype(int)
        students_df["course_id"] = students_df["course_id"].astype(int)
        return students_df.groupby(["year", "course_id"]).size().reset_index(name="total_students")

    def _get_student_interactions(self, student_logs_file):
        logs = pd.read_csv(student_logs_file)
        logs["timecreated"] = pd.to_datetime(logs["timecreated"], unit="s", errors="coerce")
        logs["contextinstanceid"] = logs["contextinstanceid"].astype("Int64")
        logs["is_viewed"] = logs["eventname"].str.contains("view", case=False, na=False)

        # --- MÉTRICAS POR VISTAS ---
        total_views = (
            logs[logs["is_viewed"]]
            .groupby(["contextinstanceid", "year"])
            .agg(
                student_total_views=("id", "count"),
                students_who_viewed=("userid", pd.Series.nunique),
            )
        )

        # --- MÉTRICAS POR INTERACCIONES (NO VISTAS) ---
        interacciones = (
            logs[~logs["is_viewed"]]
            .groupby(["contextinstanceid", "year"])
            .agg(
                student_total_interactions=("id", "count"),
                students_who_interacted=("userid", pd.Series.nunique),
            )
        )

        # --- MÉTRICAS POR ESTUDIANTE ---
        vistas_por_estudiante = (
            logs[logs["is_viewed"]]
            .groupby(["contextinstanceid", "year", "userid"])
            .size()
            .groupby(["contextinstanceid", "year"])
            .agg(
                min_views_per_student="min",
                max_views_per_student="max",
                median_views_per_student="median",
            )
        )

        # --- UNIÓN POR INDEX: contextinstanceid + year ---
        resumen = (
            total_views.join(interacciones, how="outer")
            .join(vistas_por_estudiante, how="outer")
            .fillna(0)
            .reset_index()
            .rename(columns={"contextinstanceid": "course_module_id"})
        )

        # --- AJUSTE DE TIPOS ---
        int_cols = [
            "student_total_views",
            "students_who_viewed",
            "student_total_interactions",
            "students_who_interacted",
            "min_views_per_student",
            "max_views_per_student",
        ]
        for col in int_cols:
            resumen[col] = resumen[col].astype(int)

        resumen["median_views_per_student"] = resumen["median_views_per_student"].astype(int)

        return resumen

    # ---------- MÉTRICAS Y UNIONES ----------
    def merge_modules_logs_update(self, modules_df, logs_df):
        modules_df["year"] = modules_df["year"].astype(int)
        logs_df["year"] = logs_df["year"].astype(int)

        df = modules_df.merge(logs_df, on=["course_module_id", "year"], how="left")

        # Convertir fechas y calcular diferencias
        for col in ["module_creation_date", "last_update_date", "planned_start_date"]:
            df[col] = pd.to_datetime(df[col]).dt.tz_localize(None)

        df["days_since_creation"] = (df["planned_start_date"] - df["module_creation_date"]).dt.days
        df["days_since_last_update"] = (df["planned_start_date"] - df["last_update_date"]).dt.days

        return df

    def _add_metrics(self, df):
        df["percent_students_interacted"] = (df["students_who_interacted"] / df["total_students"]).fillna(0) * 100
        df["percent_students_viewed"] = (df["students_who_viewed"] / df["total_students"]).fillna(0) * 100
        df["interaction_to_view_ratio"] = (df["student_total_interactions"] / df["student_total_views"]).fillna(0)
        df["teacher_accessed_before_start"] = (df["teacher_first_view_date"] < df["planned_start_date"]).astype(int)
        df["teacher_updated_before_start"] = (df["last_update_date"] < df["planned_start_date"]).astype(int)
        df["teacher_updated_during_week_planned"] = (
            (df["last_update_date"] >= df["planned_start_date"]) & (df["last_update_date"] < df["planned_end_date"])
        ).astype(int)
        df["teacher_active"] = ((df["teacher_updated_before_start"] > 0) | (df["teacher_updated_during_week_planned"] > 0)).astype(int)

    def _set_interactive_flag(self, df, hvp_path):
        if not os.path.exists(hvp_path):
            self.logger.warning(f"HVP file not found: {hvp_path}")
            df["is_interactive"] = 0
            return df

        hvp_df = pd.read_csv(hvp_path, dtype={"course_module_id": int, "course_id": int, "year": int, "is_interactive": int})

        hvp_lookup = {(row["year"], row["course_id"], row["course_module_id"]): row["is_interactive"] for _, row in hvp_df.iterrows()}

        df["is_interactive"] = df.apply(lambda row: hvp_lookup.get((row["year"], row["course_id"], row["course_module_id"]), 0), axis=1).astype(int)

        return df

    def process_df(self, df, logs_df, students_df, teacher_logs_file, student_logs_file, hvp_path):
        df = self.merge_modules_logs_update(df, logs_df)

        # Vistas docentes
        vistas_docente = self._get_vistas_docente(teacher_logs_file)
        df = df.merge(vistas_docente, on=["course_module_id", "year"], how="left")

        # Logs docentes
        df["teacher_total_updates"] = df["teacher_total_updates"].fillna(0).astype(int)
        df["teacher_total_views"] = df["teacher_total_views"].fillna(0).astype(int)

        # Estudiantes por curso
        student_counts = self._get_students_count(students_df)
        df = df.merge(student_counts, on=["year", "course_id"], how="left")
        df["total_students"] = df["total_students"].fillna(0).astype(int)

        # Interacciones estudiantes
        interacciones = self._get_student_interactions(student_logs_file)
        df = df.merge(interacciones, on=["course_module_id", "year"], how="left")

        # Completar valores faltantes
        metric_cols = [
            "student_total_views",
            "students_who_viewed",
            "student_total_interactions",
            "students_who_interacted",
            "min_views_per_student",
            "max_views_per_student",
            "median_views_per_student",
        ]
        for col in metric_cols:
            df[col] = df[col].fillna(0).astype(int)

        self._add_metrics(df)

        self._set_interactive_flag(df, hvp_path)

        return df

    # ---------- EJECUCIÓN PRINCIPAL ----------
    def process_course_data(self):
        moodle_df = pd.read_csv("data/interim/moodle/modules_active_moodle.csv")
        edukrea_df = pd.read_csv("data/interim/moodle/modules_active_edukrea.csv")

        students_moodle = pd.read_csv("data/interim/moodle/student_courses_moodle.csv")
        students_edukrea = pd.read_csv("data/interim/moodle/student_courses_edukrea.csv")

        logs_table = "logstore_standard_log"

        # Logs para los años
        logs_2024 = self._get_logs_updated(MoodlePathResolver.get_paths(2024, logs_table)[0], 2024)
        logs_2025 = self._get_logs_updated(MoodlePathResolver.get_paths(2025, logs_table)[0], 2025)
        logs_moodle = pd.concat([logs_2024, logs_2025], ignore_index=True)

        logs_edukrea = self._get_logs_updated(MoodlePathResolver.get_paths("Edukrea", logs_table)[0], 2025)

        # Procesar Moodle
        moodle_df = self.process_df(
            moodle_df,
            logs_moodle,
            students_moodle,
            "data/interim/moodle/teacher_logs_moodle.csv",
            "data/interim/moodle/student_logs_moodle.csv",
            "data/interim/moodle/hvp_moodle.csv",
        )
        self.save_to_csv(moodle_df, "data/interim/moodle/modules_featured_moodle.csv")

        # Procesar Edukrea
        edukrea_df = self.process_df(
            edukrea_df,
            logs_edukrea,
            students_edukrea,
            "data/interim/moodle/teacher_logs_edukrea.csv",
            "data/interim/moodle/student_logs_edukrea.csv",
            "data/interim/moodle/hvp_edukrea.csv",
        )
        self.save_to_csv(edukrea_df, "data/interim/moodle/modules_featured_edukrea.csv")


if __name__ == "__main__":
    processor = ModuleFeaturesProcessor()
    processor.process_course_data()
    processor.close()
    processor.logger.info("Course modules processed successfully.")
