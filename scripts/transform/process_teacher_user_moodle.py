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

    def _get_teachers(self, year, context_file, role_assignments_file, role_file, user_file, platform="moodle"):
        sql_docentes = f"""
            SELECT DISTINCT u.id AS userid, 
                u.firstname, u.lastname
            FROM '{context_file}' ctx 
            JOIN '{role_assignments_file}' ra ON ra.contextid = ctx.id
            JOIN '{role_file}' r ON r.id = ra.roleid
            JOIN '{user_file}' u ON u.id = ra.userid
            WHERE r.shortname = 'editingteacher'
            AND NOT (u.firstname = 'Provisional' AND u.lastname = 'Girardot')
            AND ctx.contextlevel = 50
            """
        try:
            docentes_df = self.con.execute(sql_docentes).df()
            docentes_df["nombre"] = docentes_df["firstname"] + " " + docentes_df["lastname"]
            docentes_df["nombre_normalized"] = docentes_df["nombre"].apply(self.normalize_name)
            return docentes_df
        except Exception as e:
            self.logger.error(f"Error cargando datos de actividad para el año {year}: {str(e)}")
            raise

    def process_teacher_logs(self):
        carga = "data/raw/tablas_maestras/carga_horaria.csv"
        carga_df = pd.read_csv(carga, dtype={"id_docente": "Int64", "id_curso": "Int64"})
        docentes_con_carga_2024_2025 = carga_df[carga_df["year"].isin([2024, 2025])]["id_docente"].unique()

        docentes_file = "data/raw/tablas_maestras/docentes.csv"
        docentes_df = pd.read_csv(docentes_file, dtype={"id_docente": "Int64"})
        docentes_df["nombre_normalized"] = docentes_df["nombre"].apply(self.normalize_name)

        # Filtrar docentes que tienen registros en carga_df para 2024 y 2025
        docentes_con_carga_2024_2025 = carga_df[carga_df["year"].isin([2024, 2025])]["id_docente"].unique()
        docentes_df = docentes_df[docentes_df["id_docente"].isin(docentes_con_carga_2024_2025)]

        # Get logs for 2024
        year = 2024
        context_file, role_assignments_file, role_file, user_file = MoodlePathResolver.get_paths(
            year, "context", "role_assignments", "role", "user"
        )
        teacher_moodle_2024 = self._get_teachers(year, context_file, role_assignments_file, role_file, user_file)

        # Get logs for 2025
        year = 2025
        context_file, role_assignments_file, role_file, user_file = MoodlePathResolver.get_paths(
            year, "context", "role_assignments", "role", "user"
        )
        teacher_moodle_2025 = self._get_teachers(year, context_file, role_assignments_file, role_file, user_file)

        # Get logs Edukrea
        year = 2025
        context_file, role_assignments_file, role_file, user_file = MoodlePathResolver.get_paths(
            "Edukrea", "context", "role_assignments", "role", "user"
        )
        teacher_moodle_edukrea = self._get_teachers(year, context_file, role_assignments_file, role_file, user_file)

        teachers_unique = pd.concat([teacher_moodle_2024, teacher_moodle_2025])
        teachers_unique = teachers_unique.drop_duplicates(subset=["userid", "nombre_normalized"])

        # Unir por nombre_normalized
        teachers_unique = teachers_unique.rename(columns={"userid": "moodle_user_id"})
        docentes_con_moodle_ids = docentes_df.merge(
            teachers_unique[["nombre_normalized", "moodle_user_id"]], on="nombre_normalized", how="left"
        )

        teacher_moodle_edukrea = teacher_moodle_edukrea.rename(columns={"userid": "edukrea_user_id"})
        docentes_con_todos_ids = docentes_con_moodle_ids.merge(
            teacher_moodle_edukrea[["nombre_normalized", "edukrea_user_id"]], on="nombre_normalized", how="left"
        )

        final_df = docentes_con_todos_ids[["id_docente", "nombre", "moodle_user_id", "edukrea_user_id", "sede"]].drop_duplicates(
            subset=["nombre", "moodle_user_id", "edukrea_user_id"]
        )

        # Convertir IDs a tipo entero que permite nulos
        final_df["moodle_user_id"] = final_df["moodle_user_id"].astype("Int64")
        final_df["edukrea_user_id"] = final_df["edukrea_user_id"].astype("Int64")

        final_df = final_df[final_df["id_docente"].isin(docentes_con_carga_2024_2025)]

        output_ids = "data/interim/moodle/teachers_users.csv"
        self.save_to_csv(final_df, output_ids)


if __name__ == "__main__":
    processor = TeacherMoodleUserProcessor()
    processor.process_teacher_logs()
    processor.logger.info("Teacher logs processed successfully.")
    processor.close()
