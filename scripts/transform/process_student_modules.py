import pandas as pd
import os
import sys
import re
import duckdb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.base_script import BaseScript
from utils.moodle_path_resolver import MoodlePathResolver


class StudentModulesProcessor(BaseScript):
    @staticmethod
    def _classify_event(eventname):
        eventname = str(eventname).lower()
        if "view" in eventname:
            return "view"
        else:
            return "interaction"

    @staticmethod
    def _merge_modules_students(students_df, modules_df):
        students_sel = students_df[["documento_identificación","year","id_grado","sede"]]
        return students_sel.merge(modules_df, on=["id_grado", "year", "sede"], how="inner")

    @staticmethod
    def _get_student_logs_summary(logs_df):
        """
        Genera un resumen de la actividad de los estudiantes por módulo.

        Métricas base calculadas:
        - num_views: Número total de visualizaciones del módulo por estudiante
        - num_interactions: Número total de interacciones del módulo por estudiante
        - first_view: Fecha y hora del primer acceso al módulo
        - last_view: Fecha y hora del último acceso al módulo

        Args:
            logs_df (pd.DataFrame): DataFrame con logs de actividad de estudiantes

        Returns:
            pd.DataFrame: Resumen de actividad por estudiante y módulo
        """
        student_logs_summary = (
            logs_df.groupby(["documento_identificación", "contextinstanceid", "year", "platform"])
            .agg(
                num_views=("event_type", lambda x: (x == "view").sum()),
                num_interactions=("event_type", lambda x: (x == "interaction").sum()),
                first_view=("timecreated", "min"),
                last_view=("timecreated", "max"),
            )
            .reset_index()
            .rename(columns={"contextinstanceid": "course_module_id"})
        )
        return student_logs_summary

    @staticmethod
    def _get_student_grades(enrollments_file, year, platform):
        """
        Obtiene las calificaciones finales de los estudiantes para los módulos usando SQL.
        
        Args:
            enrollments_file (str): Ruta al archivo de enrollments
            year (int): Año para el cual obtener las calificaciones
            platform (str): Plataforma ('moodle' o 'edukrea')
            
        Returns:
            pd.DataFrame: DataFrame con calificaciones finales por estudiante y módulo
        """
        try:
            # Determinar el folder y la columna de join según la plataforma
            if platform.lower() == 'moodle':
                folder = year
                user_id_column = 'moodle_user_id'
            elif platform.lower() == 'edukrea':
                folder = "Edukrea"
                user_id_column = 'edukrea_user_id'
            else:
                raise ValueError(f"Platform '{platform}' no es válida. Usa 'moodle' o 'edukrea'.")
            
            # Obtener rutas de las tablas
            tables = ["grade_grades", "grade_items"]
            grades_file, grades_items_file = MoodlePathResolver.get_paths(folder, *tables)
            
            # Query SQL para obtener las calificaciones
            con = duckdb.connect()
            sql = f"""
            SELECT 
                e.documento_identificación,
                gi.iteminstance AS course_module_id,
                g.finalgrade
            FROM '{grades_file}' g
            JOIN '{grades_items_file}' gi ON g.itemid = gi.id
            JOIN '{enrollments_file}' e ON g.userid = e.{user_id_column}
            WHERE e.year = {year}
            """
            
            result = con.execute(sql).df()
            con.close()
            
            return result
            
        except Exception as e:
            print(f"Error obteniendo calificaciones para {platform} {year}: {e}")
            return pd.DataFrame(columns=['documento_identificación', 'course_module_id', 'finalgrade'])

    @staticmethod
    def _calculate_metrics(df):
        """
        Calcula métricas de engagement y temporalidad para los módulos de estudiantes.
        
        Métricas calculadas:
        - has_viewed: Indica si el estudiante visualizó el módulo al menos una vez (1=sí, 0=no)
        - has_participated: Indica si el estudiante interactuó con el módulo al menos una vez (1=sí, 0=no)
        - days_from_planned_start: Días antes de la fecha de inicio planificada en que el estudiante 
          accedió por primera vez al módulo (valores negativos indican acceso temprano)
        - days_after_end: Días después de la fecha de fin planificada en que el estudiante 
          accedió por última vez al módulo (valores negativos indican acceso dentro del período)
        - was_on_time: Indica si el primer acceso del estudiante fue dentro del período 
          planificado del módulo (1=sí, 0=no)

        Args:
            df (pd.DataFrame): DataFrame con datos de módulos y logs de estudiantes

        Returns:
            pd.DataFrame: DataFrame con las métricas calculadas agregadas
        """
        df["first_view"] = pd.to_datetime(df["first_view"], errors="coerce")
        df["last_view"] = pd.to_datetime(df["last_view"], errors="coerce")
        df["planned_start_date"] = pd.to_datetime(df["planned_start_date"], errors="coerce")
        df["planned_end_date"] = pd.to_datetime(df["planned_end_date"], errors="coerce")

        # Métricas de engagement
        df["has_viewed"] = (df["num_views"].fillna(0) > 0).astype(int)
        df["has_participated"] = (df["num_interactions"].fillna(0) > 0).astype(int)
        
        # Métricas temporales
        df["days_from_planned_start"] = (df["first_view"] - df["planned_start_date"]).dt.days
        df["days_after_end"] = (df["last_view"] - df["planned_end_date"]).dt.days
        df["was_on_time"] = ((df["first_view"] >= df["planned_start_date"]) & (df["first_view"] <= df["planned_end_date"])).astype(int)

        return df

    @staticmethod
    def _process_df(modules_df, students_df, logs_df):
        """
        Procesa y combina los datos de módulos, estudiantes y logs para generar 
        un dataset con métricas de engagement y temporalidad.
        
        El dataset final incluye:
        - Información básica: year, course_id, platform, course_module_id, sede, id_grado
        - Datos del estudiante: documento_identificación
        - Métricas de actividad: num_views, num_interactions, first_view, last_view
        - Métricas de engagement: has_viewed, has_participated
        - Métricas temporales: days_from_planned_start, days_after_end, was_on_time
        - Calificaciones: finalgrade
        
        Args:
            modules_df (pd.DataFrame): DataFrame con información de módulos
            students_df (pd.DataFrame): DataFrame con información de estudiantes
            logs_df (pd.DataFrame): DataFrame con logs de actividad
            
        Returns:
            pd.DataFrame: Dataset procesado con todas las métricas calculadas
        """
        modules_df = modules_df[
            [
                "year",
                "sede",
                "id_grado",
                "id_asignatura",
                "period",
                "course_id",
                "platform",
                "course_module_id",
                "planned_start_date",
                "planned_end_date",
            ]
        ].copy()

        df_base = StudentModulesProcessor._merge_modules_students(students_df, modules_df)

        logs_df["event_type"] = logs_df["eventname"].apply(StudentModulesProcessor._classify_event)

        student_logs_summary = StudentModulesProcessor._get_student_logs_summary(logs_df)

        df_full = df_base.merge(student_logs_summary, on=["documento_identificación", "course_module_id", "year", "platform"], how="left")

        df_full = StudentModulesProcessor._calculate_metrics(df_full)

        # Obtener calificaciones para cada combinación de año y plataforma
        enrollments_file = "data/interim/estudiantes/enrollments.csv"
        year_platform_combinations = df_full[['year', 'platform']].drop_duplicates()
        
        all_grades = []
        for _, row in year_platform_combinations.iterrows():
            year = row['year']
            platform = row['platform']
            grades = StudentModulesProcessor._get_student_grades(enrollments_file, year, platform)
            if not grades.empty:
                grades['year'] = year
                grades['platform'] = platform
                all_grades.append(grades)
        
        # Combinar calificaciones si hay datos
        if all_grades:
            grades_df = pd.concat(all_grades, ignore_index=True)
            df_full = df_full.merge(
                grades_df[['documento_identificación', 'course_module_id', 'finalgrade', 'year', 'platform']], 
                on=['documento_identificación', 'course_module_id', 'year', 'platform'], 
                how='left'
            )
        else:
            df_full['finalgrade'] = None

        # Remove the date columns used only for calculations
        df_full = df_full.drop(columns=["planned_start_date", "planned_end_date"])

        return df_full

    def process_course_data(self):
        modules_df = pd.read_csv("data/interim/moodle/modules_featured.csv")
        students_df = pd.read_csv("data/interim/estudiantes/enrollments.csv")
        logs_df = pd.read_csv("data/interim/moodle/student_logs.csv")

        student_modules_moodle = StudentModulesProcessor._process_df(modules_df, students_df, logs_df)

        self.save_to_csv(student_modules_moodle, "data/interim/moodle/student_modules.csv")
        self.logger.info("Courses data processed and saved successfully.")

if __name__ == "__main__":
    processor = StudentModulesProcessor()
    processor.process_course_data()
    processor.logger.info("Student Modules processed successfully.")
    processor.close()
