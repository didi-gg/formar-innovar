import duckdb
import pandas as pd
import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.academic_period_utils import AcademicPeriodUtils


class MoodleModuleProcessor:
    """
    Procesa los módulos de cursos de Moodle y Edukrea combinando información de distintas fuentes.
    Extrae, transforma y guarda los datos unificados de módulos activos por curso.
    """

    def __init__(self):
        self.connection = duckdb.connect()
        self.logger = logging.getLogger(__name__)

    def __del__(self):
        self.close_connection()

    def close_connection(self):
        """
        Cierra la conexión activa a DuckDB.
        """
        if hasattr(self, "connection") and self.connection:
            self.connection.close()
            self.connection = None

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
                    cm.added AS module_added,
                    m.name AS module_type
                FROM '{modules_file}' AS cm
                INNER JOIN '{module_names_file}' AS m ON cm.module = m.id
                INNER JOIN '{sections_file}' AS s ON cm.section = s.id
                INNER JOIN '{courses_file}' AS c ON cm.course = c.course_id
                INNER JOIN '{asignaturas_file}' AS a ON c.id_asignatura = a.id_asignatura
                WHERE c.year = {year}
                  AND cm.visible = 1
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
            """
            return self.connection.execute(sql).df()

        except Exception as e:
            self.logger.error(f"Error cargando datos para el año {year}: {str(e)}")
            raise

    def _load_modules_edukrea(
        self,
        courses_file,
        asignaturas_file,
    ) -> pd.DataFrame:
        """
        Carga los módulos de cursos activos y enriquecidos con su nombre según el tipo.

        Returns:
            pd.DataFrame: Información unificada de módulos activos.
        """

        try:
            modules_file = "data/raw/moodle/Edukrea/Courses/mdl_course_modules.parquet"
            module_names_file = "data/raw/moodle/Edukrea/Other/mdl_modules.parquet"
            sections_file = "data/raw/moodle/Edukrea/Courses/mdl_course_sections.parquet"

            label_file = "data/raw/moodle/Edukrea/Basic Resources/mdl_label.parquet"
            hvp_file = "data/raw/moodle/Edukrea/Interactive Content/mdl_hvp.parquet"
            forum_file = "data/raw/moodle/Edukrea/Forums/mdl_forum.parquet"
            page_file = "data/raw/moodle/Edukrea/Basic Resources/mdl_page.parquet"
            assign_file = "data/raw/moodle/Edukrea/Assignments and Grades/mdl_assign.parquet"
            resource_file = "data/raw/moodle/Edukrea/Files and Resources/mdl_resource.parquet"
            url_file = "data/raw/moodle/Edukrea/Basic Resources/mdl_url.parquet"

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
                    cm.added AS module_added,
                    m.name AS module_type
                FROM '{modules_file}' AS cm
                INNER JOIN '{module_names_file}' AS m ON cm.module = m.id
                INNER JOIN '{sections_file}' AS s ON cm.section = s.id
                INNER JOIN '{courses_file}' AS c ON cm.course = c.course_id
                INNER JOIN '{asignaturas_file}' AS a ON c.id_asignatura = a.id_asignatura
                WHERE c.year = 2025
                AND cm.visible = 1
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
            """
            return self.connection.execute(sql).df()

        except Exception as e:
            self.logger.error(f"Error cargando datos para edukrea: {str(e)}")
            raise

    def save_to_csv(self, df: pd.DataFrame, file_path: str):
        """
        Guarda un DataFrame como archivo CSV.

        Args:
            df (pd.DataFrame): Datos a guardar.
            file_path (str): Ruta del archivo de salida.
        """
        df.to_csv(file_path, index=False, encoding="utf-8-sig", quoting=1)
        self.logger.info(f"Archivo guardado exitosamente en: {file_path}")

    def _build_module_file_paths(self, year):
        """
        Genera las rutas de los archivos parquet por año.

        Returns:
            list[str]: Lista ordenada de archivos a usar en `_load_modules`.
        """
        base_path = f"data/raw/moodle/{year}"
        return [
            f"{base_path}/Course/mdlvf_course_modules.parquet",
            f"{base_path}/Course Formats/mdlvf_modules.parquet",
            f"{base_path}/Course/mdlvf_course_sections.parquet",
            f"{base_path}/Forum/mdlvf_forum.parquet",
            f"{base_path}/Quiz/mdlvf_quiz.parquet",
            f"{base_path}/Assignments/mdlvf_assign.parquet",
            f"{base_path}/Page/mdlvf_page.parquet",
            f"{base_path}/Course/mdlvf_resource.parquet",
            f"{base_path}/Content/mdlvf_label.parquet",
            f"{base_path}/Choice/mdlvf_choice.parquet",
            f"{base_path}/Content/mdlvf_bootstrapelements.parquet",
            f"{base_path}/Content/mdlvf_folder.parquet",
            f"{base_path}/h5/mdlvf_hvp.parquet",
            f"{base_path}/LTI/mdlvf_lti.parquet",
            f"{base_path}/Content/mdlvf_url.parquet",
            f"{base_path}/Lesson/mdlvf_lesson.parquet",
            f"{base_path}/Workshop/mdlvf_workshop.parquet",
            f"{base_path}/Book/mdlvf_book.parquet",
            f"{base_path}/Chat/mdlvf_chat.parquet",
            f"{base_path}/Feedback/mdlvf_feedback.parquet",
            f"{base_path}/Content/mdlvf_glossary.parquet",
        ]

    def process_moodle_modules(self, df):
        # Add 'edukrea_access' column to indicate if the module has Edukrea access
        df["edukrea_access"] = df.apply(self.is_edukrea_url_module, axis=1)

        # Add 'inasistencia_assign' column to indicate if the module is an assignment for inasistencia
        df["inasistencia_assign"] = df.apply(self.is_inasistencia_assignment, axis=1)

        # Add the weeok and period columns based on section names
        df["week"] = df["section_name"].apply(AcademicPeriodUtils.extract_week_number_string)

        df["period"] = df["week"].apply(AcademicPeriodUtils.get_period_from_week)

        df = df[df["inasistencia_assign"] == 0]
        return df

    def process_moodle_modules_edukrea(self, df):
        # Add 'inasistencia_assign' column to indicate if the module is an assignment for inasistencia
        df["inasistencia_assign"] = df.apply(self.is_inasistencia_assignment, axis=1)

        # Add the weeok and period columns based on section names
        df["week"] = df["section_name"].apply(AcademicPeriodUtils.extract_week_number_from_title)
        df["period"] = df["week"].apply(AcademicPeriodUtils.get_period_from_week)

        df = df[df["inasistencia_assign"] == 0]
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
        if "edukrea_access" in df.columns:
            df["edukrea_access"] = df["edukrea_access"].astype(int)
        df["inasistencia_assign"] = df["inasistencia_assign"].astype(int)
        df["module_added"] = pd.to_datetime(df["module_added"], unit="s", errors="coerce").dt.tz_localize("America/Bogota")
        df["week"] = df["week"].astype(int)
        df["period"] = df["period"].astype(int)
        df["sede"] = df["sede"].astype(str)
        df["asignatura_name"] = df["asignatura_name"].astype(str)
        df["course_name"] = df["course_name"].astype(str)
        df["section_name"] = df["section_name"].astype(str)
        df["module_type"] = df["module_type"].astype(str)
        df["module_name"] = df["module_name"].astype(str)
        return df

    def _get_logs_updated(self, logs_file):
        sql_logs = """
            SELECT 
                contextinstanceid AS course_module_id,
                MAX(to_timestamp(timecreated)) AS fecha_ultima_actualizacion,
                COUNT(*) AS total_actualizaciones_docente
            FROM '{}'
            WHERE eventname LIKE '%updated%'
            GROUP BY contextinstanceid
        """.format(logs_file)
        return self.connection.execute(sql_logs).df()

    def _get_calendar_df(self):
        calendario_file = "data/raw/tablas_maestras/calendario_escolar.csv"
        calendario_df = pd.read_csv(calendario_file)
        calendario_df["inicio"] = pd.to_datetime(calendario_df["inicio"], dayfirst=True)
        calendario_df.rename(columns={"año": "year", "bimestre": "period", "semana_general": "week"}, inplace=True)
        return calendario_df

    def merge_modules_logs_update(self, modules_df, logs_df, calendario_df):
        # 1. Unir módulos con calendario
        modules_df["year"] = modules_df["year"].astype(int)
        modules_df["period"] = modules_df["period"].astype(int)
        modules_df["week"] = modules_df["week"].astype(int)

        calendario_df["year"] = calendario_df["year"].astype(int)
        calendario_df["period"] = calendario_df["period"].astype(int)
        calendario_df["week"] = calendario_df["week"].astype(int)

        modules_df = modules_df.merge(calendario_df[["year", "period", "week", "inicio"]], on=["year", "period", "week"], how="left").rename(
            columns={"inicio": "fecha_inicio_semana"}
        )

        # 2. Unir módulos con fecha de última actualización
        modules_df = modules_df.merge(logs_df, on="course_module_id", how="left")

        # 3. Calcular días desde creación y última actualización
        modules_df["module_added"] = pd.to_datetime(modules_df["module_added"]).dt.tz_localize(None)
        modules_df["fecha_ultima_actualizacion"] = pd.to_datetime(modules_df["fecha_ultima_actualizacion"]).dt.tz_localize(None)
        modules_df["fecha_inicio_semana"] = pd.to_datetime(modules_df["fecha_inicio_semana"]).dt.tz_localize(None)

        # Resta de fechas segura
        modules_df["dias_desde_creacion"] = (modules_df["fecha_inicio_semana"] - modules_df["module_added"]).dt.days
        modules_df["dias_desde_ultima_actualizacion"] = (modules_df["fecha_inicio_semana"] - modules_df["fecha_ultima_actualizacion"]).dt.days
        return modules_df

    def _get_vistas_docente(self, logs_file):
        # 1. Cargar logs docentes
        logs_docentes = pd.read_csv(logs_file)

        # 2. Convertir columnas de tiempo
        logs_docentes["timecreated"] = pd.to_datetime(logs_docentes["timecreated"], unit="s", errors="coerce")

        # 3. Filtrar eventos de vista docente
        vistas_docentes = logs_docentes[logs_docentes["eventname"].str.contains("viewed", case=False, na=False)]

        # 4. Agrupar por módulo
        vistas_agg = (
            vistas_docentes.groupby("contextinstanceid")
            .agg(
                total_vistas_docente=("contextinstanceid", "count"),
                fecha_primera_vista=("timecreated", "min"),
                fecha_ultima_vista=("timecreated", "max"),
            )
            .rename_axis("course_module_id")
            .reset_index()
        )
        return vistas_agg

    def process_course_data(self):
        """ """
        courses_file = "data/interim/moodle/courses_unique_moodle.csv"
        asignaturas_file = "data/raw/tablas_maestras/asignaturas.csv"

        calendario_df = self._get_calendar_df()

        file_paths = self._build_module_file_paths(2024)
        modules_2024 = self._load_modules(courses_file, asignaturas_file, *file_paths, year=2024)
        logs_2024 = self._get_logs_updated("data/raw/moodle/2024/Log/mdlvf_logstore_standard_log.parquet")

        file_paths = self._build_module_file_paths(2025)
        modules_2025 = self._load_modules(courses_file, asignaturas_file, *file_paths, year=2025)
        logs_2025 = self._get_logs_updated("data/raw/moodle/2025/Log/mdlvf_logstore_standard_log.parquet")

        modules_2024 = self.process_moodle_modules(modules_2024)
        modules_2025 = self.process_moodle_modules(modules_2025)

        # Filtrar solo por el primer bimestre o 0, lo demás esta en construcción
        modules_2025 = modules_2025[modules_2025["period"].isin([1, 0])]
        # Excluir el module_type 23 # bootstrapelements
        modules_2025 = modules_2025[modules_2025["module_type_id"] != 23]

        # Unir los DataFrames de 2024 y 2025
        modules_combined = pd.concat([modules_2024, modules_2025], ignore_index=True)

        # Edukrea data processing
        edukrea_df = self._load_modules_edukrea(courses_file, asignaturas_file)
        edukrea_df = self.process_moodle_modules_edukrea(edukrea_df)

        # Filtrar solo por el primer bimestre 1 lo demás esta en construcción
        edukrea_df = edukrea_df[edukrea_df["period"] == 1]

        logs_edukrea = self._get_logs_updated("data/raw/moodle/Edukrea/Logs and Events/mdl_logstore_standard_log.parquet")
        edukrea_df = self.merge_modules_logs_update(edukrea_df, logs_edukrea, calendario_df)

        # Agregar columnas para los días desde la creación y última actualización
        logs_2024_2025 = pd.concat([logs_2024, logs_2025], ignore_index=True)
        modules_combined = self.merge_modules_logs_update(modules_combined, logs_2024_2025, calendario_df)

        # Agregar vistas docentes
        vistas_agg = self._get_vistas_docente("data/interim/moodle/teachers_logs_moodle.csv")
        modules_combined = modules_combined.merge(vistas_agg, on="course_module_id", how="left")

        # Calcular si accedió antes
        modules_combined["accedio_antes"] = modules_combined["fecha_primera_vista"] < modules_combined["fecha_inicio_semana"]

        # Agregar vistas docentes edukrea
        vistas_agg = self._get_vistas_docente("data/interim/moodle/teachers_logs_edukrea.csv")
        edukrea_df = edukrea_df.merge(vistas_agg, on="course_module_id", how="left")

        # Calcular si accedió antes
        edukrea_df["accedio_antes"] = edukrea_df["fecha_primera_vista"] < edukrea_df["fecha_inicio_semana"]

        modules_combined["total_actualizaciones_docente"] = modules_combined["total_actualizaciones_docente"].fillna(0).astype(int)
        edukrea_df["total_actualizaciones_docente"] = edukrea_df["total_actualizaciones_docente"].fillna(0).astype(int)

        modules_combined["total_vistas_docente"] = modules_combined["total_vistas_docente"].fillna(0).astype(int)
        edukrea_df["total_vistas_docente"] = edukrea_df["total_vistas_docente"].fillna(0).astype(int)

        # Agregar columna interactivo con valor 1 si el module_type esta en el array de tipos interactivos
        interactive_types = ["assign", "quiz", "forum", "hvp", "choice", "feedback", "chat", "workshop", "lti"]
        modules_combined["interactivo"] = modules_combined["module_type"].apply(lambda x: 1 if x in interactive_types else 0)
        edukrea_df["interactivo"] = edukrea_df["module_type"].apply(lambda x: 1 if x in interactive_types else 0)

        # Finally, save the Edukrea data to CSV
        output_file = "data/interim/moodle/modules_active_moodle.csv"
        self.save_to_csv(self.cast_column_types(modules_combined), output_file)
        edukrea_output_file = "data/interim/moodle/modules_active_edukrea.csv"
        self.save_to_csv(self.cast_column_types(edukrea_df), edukrea_output_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    processor = MoodleModuleProcessor()
    processor.process_course_data()
