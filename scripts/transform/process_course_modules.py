import pandas as pd
import os
import sys
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.moodle_path_resolver import MoodlePathResolver
from utils.base_script import BaseScript
from utils.academic_period_utils import AcademicPeriodUtils


class MoodleModulesProcessor(BaseScript):
    """
    Procesa los módulos de cursos de Moodle y Edukrea combinando información de distintas fuentes.
    Extrae, transforma y guarda los datos unificados de módulos activos por curso.
    """

    INTERACTIVE_MODULES = {"assign", "quiz", "forum", "hvp", "choice", "feedback", "chat", "workshop", "lti"}
    ENGLISH_REGEX = r"(?:reto(?:\s+\w+){0,2}\s+ingl[eé]s|english)"

    @staticmethod
    def is_edukrea_url_module(row):
        if pd.isna(row.module_name):
            return 0
        return 1 if "edukrea" in str(row.module_name).lower() and row.module_type == "url" else 0

    @staticmethod
    def is_inasistencia_assignment(row):
        module_name = str(row.module_name).lower() if pd.notna(row.module_name) else ""
        keywords = {"inasistencia", "insistencia", "inasistecia"}
        return int(any(word in module_name for word in keywords))

    def _load_modules(
        self,
        courses_file,
        asignaturas_file,
        modules_file,
        module_names_file,
        sections_file,
        forum_file,
        quiz_file,
        assign_file,
        page_file,
        resource_file,
        label_file,
        choice_file,
        bootstrapelements_file,
        folder_file,
        hvp_file,
        lti_file,
        url_file,
        lesson_file,
        workshop_file,
        book_file,
        chat_file,
        feedback_file,
        glossary_file,
        year=2025,
        platform="moodle",
    ) -> pd.DataFrame:
        """
        Carga los módulos de cursos activos y enriquecidos con su nombre según el tipo.

        Returns:
            pd.DataFrame: Información unificada de módulos activos.
        """
        try:
            sql = f"""
            WITH base AS (
                SELECT 
                    c.year,
                    cm.course AS course_id,
                    cm.id AS course_module_id,
                    c.sede,
                    c.id_grado,
                    c.id_asignatura,
                    a.nombre AS asignatura_name,
                    c.course_name,
                    cm.section AS section_id,
                    s.name AS section_name,
                    cm.module AS module_type_id,
                    cm.instance,
                    cm.added AS module_creation_date,
                    m.name AS module_type
                FROM '{modules_file}' AS cm
                INNER JOIN '{module_names_file}' AS m ON cm.module = m.id
                INNER JOIN '{sections_file}' AS s ON cm.section = s.id
                INNER JOIN '{courses_file}' AS c ON cm.course = c.course_id
                INNER JOIN '{asignaturas_file}' AS a ON c.id_asignatura = a.id_asignatura
                WHERE c.year = {year}
                  AND cm.visible = 1
                  AND c.platform = '{platform}'
            )
            SELECT 
                b.*,
                COALESCE(
                    f.name, q.name, a.name, p.name, r.name, l.name, c.name,
                    be.name, fo.name, h.name, lt.name, u.name, le.name, w.name,
                    bk.name, ch.name, fb.name, g.name
                ) AS module_name
            FROM base AS b
            LEFT JOIN '{forum_file}' AS f ON b.module_type = 'forum' AND b.instance = f.id
            LEFT JOIN '{quiz_file}' AS q ON b.module_type = 'quiz' AND b.instance = q.id
            LEFT JOIN '{assign_file}' AS a ON b.module_type = 'assign' AND b.instance = a.id
            LEFT JOIN '{page_file}' AS p ON b.module_type = 'page' AND b.instance = p.id
            LEFT JOIN '{resource_file}' AS r ON b.module_type = 'resource' AND b.instance = r.id
            LEFT JOIN '{label_file}' AS l ON b.module_type = 'label' AND b.instance = l.id
            LEFT JOIN '{choice_file}' AS c ON b.module_type = 'choice' AND b.instance = c.id
            LEFT JOIN '{bootstrapelements_file}' AS be ON b.module_type = 'bootstrapelements' AND b.instance = be.id
            LEFT JOIN '{folder_file}' AS fo ON b.module_type = 'folder' AND b.instance = fo.id
            LEFT JOIN '{hvp_file}' AS h ON b.module_type = 'hvp' AND b.instance = h.id
            LEFT JOIN '{lti_file}' AS lt ON b.module_type = 'lti' AND b.instance = lt.id
            LEFT JOIN '{url_file}' AS u ON b.module_type = 'url' AND b.instance = u.id
            LEFT JOIN '{lesson_file}' AS le ON b.module_type = 'lesson' AND b.instance = le.id
            LEFT JOIN '{workshop_file}' AS w ON b.module_type = 'workshop' AND b.instance = w.id
            LEFT JOIN '{book_file}' AS bk ON b.module_type = 'book' AND b.instance = bk.id
            LEFT JOIN '{chat_file}' AS ch ON b.module_type = 'chat' AND b.instance = ch.id
            LEFT JOIN '{feedback_file}' AS fb ON b.module_type = 'feedback' AND b.instance = fb.id
            LEFT JOIN '{glossary_file}' AS g ON b.module_type = 'glossary' AND b.instance = g.id
            WHERE module_name IS NOT NULL AND module_name != ''
            """
            return self.con.execute(sql).df()

        except Exception as e:
            self.logger.error(f"Error cargando datos para el año {year}: {str(e)}")
            raise

    def _load_modules_edukrea(
        self,
        courses_file,
        asignaturas_file,
        platform="edukrea",
    ) -> pd.DataFrame:
        try:
            tables = ["course_modules", "modules", "course_sections", "label", "hvp", "forum", "page", "assign", "resource", "url"]
            modules_file, module_names_file, sections_file, label_file, hvp_file, forum_file, page_file, assign_file, resource_file, url_file = (
                MoodlePathResolver.get_paths("Edukrea", *tables)
            )

            sql = f"""
            WITH base AS (
                SELECT 
                    c.year,
                    cm.course AS course_id,
                    cm.id AS course_module_id,
                    c.sede AS sede,
                    c.id_grado AS id_grado,
                    c.id_asignatura AS id_asignatura,
                    a.nombre AS asignatura_name,
                    c.course_name AS course_name,
                    cm.section AS section_id,
                    s.name AS section_name,
                    cm.module AS module_type_id,
                    cm.instance,
                    cm.added AS module_creation_date,
                    m.name AS module_type
                FROM '{modules_file}' AS cm
                INNER JOIN '{module_names_file}' AS m ON cm.module = m.id
                INNER JOIN '{sections_file}' AS s ON cm.section = s.id
                INNER JOIN '{courses_file}' AS c ON cm.course = c.course_id
                INNER JOIN '{asignaturas_file}' AS a ON c.id_asignatura = a.id_asignatura
                WHERE c.year = 2025
                AND cm.visible = 1
                AND c.platform = '{platform}'
            )
            SELECT 
                b.*,
                COALESCE(
                    l.name, h.name, f.name, p.name, a.name, r.name, u.name
                ) AS module_name
            FROM base AS b
            LEFT JOIN '{label_file}' AS l ON b.module_type = 'label' AND b.instance = l.id
            LEFT JOIN '{hvp_file}' AS h ON b.module_type = 'hvp' AND b.instance = h.id
            LEFT JOIN '{forum_file}' AS f ON b.module_type = 'forum' AND b.instance = f.id
            LEFT JOIN '{page_file}' AS p ON b.module_type = 'page' AND b.instance = p.id
            LEFT JOIN '{assign_file}' AS a ON b.module_type = 'assign' AND b.instance = a.id
            LEFT JOIN '{resource_file}' AS r ON b.module_type = 'resource' AND b.instance = r.id
            LEFT JOIN '{url_file}' AS u ON b.module_type = 'url' AND b.instance = u.id
            WHERE module_name IS NOT NULL AND module_name != ''
            """
            return self.con.execute(sql).df()

        except Exception as e:
            self.logger.error(f"Error cargando datos para edukrea: {str(e)}")
            raise

    def process_moodle_modules(self, df, edukrea=False):
        if not edukrea:
            # Add 'is_edukrea_access' column to indicate if the module has Edukrea access
            df["is_edukrea_access"] = df.apply(self.is_edukrea_url_module, axis=1)

        # Add 'is_absence_assignment' column to indicate if the module is an assignment for inasistencia
        df["is_absence_assignment"] = df.apply(self.is_inasistencia_assignment, axis=1)

        # Add the week and period columns based on section names
        if edukrea:
            df["week"] = df["section_name"].apply(AcademicPeriodUtils.extract_week_number_from_title)
        else:
            df["week"] = df["section_name"].apply(AcademicPeriodUtils.extract_week_number_string)
        df["period"] = df["week"].apply(AcademicPeriodUtils.get_period_from_week)

        df = df[df["is_absence_assignment"] == 0]
        if not edukrea:
            df = df[df["is_edukrea_access"] == 0]

        # Agregar columna is_interactive con valor 1 si el module_type esta en el array de tipos interactivos
        df["is_interactive"] = df["module_type"].apply(lambda x: 1 if x in self.INTERACTIVE_MODULES else 0)

        df["is_in_english"] = (
            df["module_name"].notna() & df["module_name"].str.contains(self.ENGLISH_REGEX, flags=re.IGNORECASE, regex=True)
        ).astype(int)

        df_calendar = AcademicPeriodUtils._load_calendar()

        df["year"] = df["year"].astype(int)
        df["period"] = df["period"].astype(int)
        df["week"] = df["week"].astype(int)
        df = df.merge(df_calendar[["year", "period", "week", "inicio"]], on=["year", "period", "week"], how="left").rename(
            columns={"inicio": "planned_start_date"}
        )

        df["planned_start_date"] = pd.to_datetime(df["planned_start_date"]).dt.normalize()
        df["planned_end_date"] = df["planned_start_date"] + pd.Timedelta(days=6)
        df["planned_end_date"] = df["planned_end_date"].dt.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)

        df = df.drop(columns=["is_absence_assignment", "is_edukrea_access"], errors="ignore")

        # Excluir el module_type 23 # bootstrapelements
        df["module_type_id"] = df["module_type_id"].astype(int)
        df = df[df["module_type_id"] != 23]
        return df

    def cast_column_types(self, df):
        df["year"] = df["year"].astype(int)
        df["course_id"] = df["course_id"].astype(int)
        df["course_module_id"] = df["course_module_id"].astype(int)
        df["id_grado"] = df["id_grado"].astype(int)
        df["id_asignatura"] = df["id_asignatura"].astype(int)
        df["section_id"] = df["section_id"].astype(int)
        df["module_type_id"] = df["module_type_id"].astype(int)
        df["instance"] = df["instance"].astype(int)
        df["module_creation_date"] = pd.to_datetime(df["module_creation_date"], unit="s", errors="coerce").dt.tz_localize("America/Bogota")
        df["week"] = df["week"].astype(int)
        df["period"] = df["period"].astype(int)
        df["sede"] = df["sede"].astype(str)
        df["asignatura_name"] = df["asignatura_name"].astype(str)
        df["course_name"] = df["course_name"].astype(str)
        df["section_name"] = df["section_name"].astype(str)
        df["module_type"] = df["module_type"].astype(str)
        df["module_name"] = df["module_name"].astype(str)
        return df

    def process_course_data(self):
        """ """
        courses_file = "data/interim/moodle/unique_courses.csv"
        asignaturas_file = "data/raw/tablas_maestras/asignaturas.csv"

        tables = [
            "course_modules",
            "modules",
            "course_sections",
            "forum",
            "quiz",
            "assign",
            "page",
            "resource",
            "label",
            "choice",
            "bootstrapelements",
            "folder",
            "hvp",
            "lti",
            "url",
            "lesson",
            "workshop",
            "book",
            "chat",
            "feedback",
            "glossary",
        ]
        file_paths = MoodlePathResolver.get_paths(2024, *tables)
        modules_2024 = self._load_modules(courses_file, asignaturas_file, *file_paths, year=2024, platform="moodle")

        file_paths = MoodlePathResolver.get_paths(2025, *tables)
        modules_2025 = self._load_modules(courses_file, asignaturas_file, *file_paths, year=2025, platform="moodle")

        modules_2024 = self.process_moodle_modules(modules_2024)
        modules_2025 = self.process_moodle_modules(modules_2025)

        # No incluir los módulos con secciones por fuera de los periodos académicos
        modules_2024 = modules_2024[modules_2024["period"].isin([1, 2, 3, 4])]

        # Filtrar solo por el primer bimestre o 0, lo demás esta en construcción
        modules_2025 = modules_2025[modules_2025["period"] == 1]

        # Unir los DataFrames de 2024 y 2025
        modules_combined = pd.concat([modules_2024, modules_2025], ignore_index=True)

        # Edukrea data processing
        edukrea_df = self._load_modules_edukrea(courses_file, asignaturas_file, platform="edukrea")
        edukrea_df = self.process_moodle_modules(edukrea_df, edukrea=True)

        # Filtrar solo por el primer bimestre 1 lo demás esta en construcción
        edukrea_df = edukrea_df[edukrea_df["period"] == 1]

        # Finally, save the Edukrea data to CSV
        output_file = "data/interim/moodle/modules_active_moodle.csv"
        self.save_to_csv(self.cast_column_types(modules_combined), output_file)
        edukrea_output_file = "data/interim/moodle/modules_active_edukrea.csv"
        self.save_to_csv(self.cast_column_types(edukrea_df), edukrea_output_file)


if __name__ == "__main__":
    processor = MoodleModulesProcessor()
    processor.process_course_data()
    processor.logger.info("Modules processed successfully.")
    processor.close()
