import pandas as pd
import os
import sys
import unicodedata


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.moodle_path_resolver import MoodlePathResolver
from utils.base_script import BaseScript


class TeacherMoodleUserProcessor(BaseScript):
    @staticmethod
    def normalize_name(name):
        if pd.isnull(name):
            return ""
        name = unicodedata.normalize("NFKD", name)
        name = "".join([c for c in name if not unicodedata.combining(c)])
        return name.upper().strip()

    def _get_teachers(self, year, course_file, unique_courses_file, context_file, role_assignments_file, role_file, user_file):
        sql_docentes = f"""
            SELECT DISTINCT u.id AS userid, 
                u.firstname, u.lastname
            FROM '{course_file}' c
            JOIN '{unique_courses_file}' uc ON c.id = uc.course_id
            JOIN '{context_file}' ctx ON ctx.instanceid = c.id AND ctx.contextlevel = 50
            JOIN '{role_assignments_file}' ra ON ra.contextid = ctx.id
            JOIN '{role_file}' r ON r.id = ra.roleid
            JOIN '{user_file}' u ON u.id = ra.userid
            WHERE r.shortname = 'editingteacher'
            AND uc.year = {year}
            AND NOT (u.firstname = 'Provisional' AND u.lastname = 'Girardot')
            """
        try:
            docentes_df = self.con.execute(sql_docentes).df()
            docentes_df["nombre"] = docentes_df["firstname"] + " " + docentes_df["lastname"]
            docentes_df["nombre_normalized"] = docentes_df["nombre"].apply(self.normalize_name)
            return docentes_df
        except Exception as e:
            self.logger.error(f"Error cargando datos de actividad para el año {year}: {str(e)}")
            raise

    def _check_missing_teachers(self, docentes_df, moodle_teachers_df):
        # Asegúrate de que la columna 'nombre_normalized' exista y esté bien procesada
        nombres_docentes = set(docentes_df["nombre_normalized"])
        nombres_moodle = set(moodle_teachers_df["nombre_normalized"])

        # Sólo los que están en docentes_df pero NO en moodle_teachers_df
        nombres_faltantes = nombres_docentes - nombres_moodle

        # Devuelve los nombres reales de los faltantes (no los "normalized")
        faltantes = docentes_df[docentes_df["nombre"].isin(nombres_faltantes)]["nombre"].unique().tolist()

        return faltantes

    def _get_teachers_by_year(self, carga_df, docentes_df, year):
        carga_year = carga_df[carga_df["year"] == year].copy()
        id_docente_unique = carga_year["id_docente"].unique()
        return docentes_df[docentes_df["id_docente"].isin(id_docente_unique)]

    def process_teacher_logs(self):
        unique_courses_file = "data/interim/moodle/unique_courses_moodle.csv"

        carga = "data/raw/tablas_maestras/carga_horaria.csv"
        carga_df = pd.read_csv(carga, dtype={"id_docente": "Int64", "id_curso": "Int64"})

        docentes_file = "data/raw/tablas_maestras/docentes.csv"
        docentes_df = pd.read_csv(docentes_file, dtype={"id_docente": "Int64"})
        docentes_df["nombre_normalized"] = docentes_df["nombre"].apply(self.normalize_name)

        # Get logs for 2024
        year = 2024
        course_file, context_file, role_assignments_file, role_file, user_file, logs_parquet = MoodlePathResolver.get_paths(
            year, "course", "context", "role_assignments", "role", "user", "logstore_standard_log"
        )
        teacher_moodle_2024 = self._get_teachers(year, course_file, unique_courses_file, context_file, role_assignments_file, role_file, user_file)
        teachers_2024 = self._get_teachers_by_year(carga_df, docentes_df, year)
        missing_teachers_2024 = self._check_missing_teachers(teacher_moodle_2024, teachers_2024)
        if missing_teachers_2024:
            raise ValueError(f"Teachers missing in 2024 logs: {', '.join(missing_teachers_2024)}")

        # Get logs for 2025
        year = 2025
        course_file, context_file, role_assignments_file, role_file, user_file, logs_parquet = MoodlePathResolver.get_paths(
            year, "course", "context", "role_assignments", "role", "user", "logstore_standard_log"
        )
        teacher_moodle_2025 = self._get_teachers(year, course_file, unique_courses_file, context_file, role_assignments_file, role_file, user_file)
        teachers_2025 = self._get_teachers_by_year(carga_df, docentes_df, year)
        missing_teachers_2025 = self._check_missing_teachers(teacher_moodle_2025, teachers_2025)
        if missing_teachers_2025:
            raise ValueError(f"Teachers missing in 2025 logs: {', '.join(missing_teachers_2025)}")

        # Get logs Edukrea
        year = 2025
        course_file, context_file, role_assignments_file, role_file, user_file, logs_parquet = MoodlePathResolver.get_paths(
            "Edukrea", "course", "context", "role_assignments", "role", "user", "logstore_standard_log"
        )
        unique_courses_file = "data/interim/moodle/unique_courses_edukrea.csv"
        teacher_moodle_edukrea = self._get_teachers(year, course_file, unique_courses_file, context_file, role_assignments_file, role_file, user_file)
        missing_teachers_edukrea = self._check_missing_teachers(teacher_moodle_edukrea, teachers_2025)
        if missing_teachers_edukrea:
            raise ValueError(f"Teachers missing in Edukrea logs: {', '.join(missing_teachers_edukrea)}")

        # Merge docentes_df con Moodle 2024 y 2025 para obtener moodle_user_id
        teacher_moodle_2024["moodle_user_id"] = teacher_moodle_2024["userid"]
        teacher_moodle_2025["moodle_user_id"] = teacher_moodle_2025["userid"]
        teacher_moodle_edukrea["edukrea_user_id"] = teacher_moodle_edukrea["userid"]

        # Unir por nombre_normalized
        docentes_con_moodle_ids = docentes_df.merge(
            pd.concat([teacher_moodle_2024, teacher_moodle_2025])[["nombre_normalized", "moodle_user_id"]], on="nombre_normalized", how="left"
        )

        # Merge con Edukrea (renombramos userid a edukrea_user_id)
        docentes_con_todos_ids = docentes_con_moodle_ids.merge(
            teacher_moodle_edukrea[["nombre_normalized", "edukrea_user_id"]], on="nombre_normalized", how="left"
        )

        final_df = docentes_con_todos_ids[["id_docente", "nombre", "moodle_user_id", "edukrea_user_id", "sede"]].drop_duplicates(
            subset=["nombre", "moodle_user_id", "edukrea_user_id"]
        )

        # Convertir IDs a tipo entero que permite nulos
        final_df["moodle_user_id"] = final_df["moodle_user_id"].astype("Int64")
        final_df["edukrea_user_id"] = final_df["edukrea_user_id"].astype("Int64")

        output_ids = "data/interim/moodle/teachers_users.csv"
        self.save_to_csv(final_df, output_ids)


if __name__ == "__main__":
    processor = TeacherMoodleUserProcessor()
    processor.process_teacher_logs()
    processor.logger.info("Teacher logs processed successfully.")
    processor.close()
