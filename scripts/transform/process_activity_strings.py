import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.moodle_path_resolver import MoodlePathResolver
from utils.base_script import BaseScript
from utils.academic_period_utils import AcademicPeriodUtils


class StudentCourseActivityStringsProcessor(BaseScript):
    def _get_module_access_logs(self, year, logs_parquet, student_courses_file, platform='moodle'):
        """
        Obtiene los logs de acceso a módulos específicos filtrados por estudiantes-cursos.
        Se enfocan en eventos de acceso/visualización de módulos.
        """
        sql_logs_estudiantes = f"""
            SELECT 
                {year} AS year,
                logs.userid,
                student_courses.documento_identificación,
                logs.courseid,
                logs.contextinstanceid AS course_module_id,
                logs.timecreated,
                logs.eventname,
                '{platform}' AS platform,
                student_courses.id_grado,
                student_courses.id_asignatura,
                student_courses.sede
            FROM '{logs_parquet}' AS logs
            INNER JOIN '{student_courses_file}' AS student_courses 
                ON student_courses.course_id = logs.courseid 
                AND student_courses.moodle_user_id = logs.userid 
                AND student_courses.platform = '{platform}'
            WHERE
                EXTRACT(YEAR FROM to_timestamp(logs.timecreated)) = {year}
                AND student_courses.year = {year}
                AND logs.contextinstanceid IS NOT NULL
            ORDER BY logs.userid, logs.courseid, logs.timecreated
        """
        try:
            return self.con.execute(sql_logs_estudiantes).df()
        except Exception as e:
            self.logger.error(f"Error al cargar los logs de módulos: {str(e)}")
            raise

    def _load_module_mapping(self):
        """
        Carga el mapeo de course_module_id a module_unique_id desde modules_active.csv
        """
        try:
            modules_df = pd.read_csv("data/interim/moodle/modules_active.csv")
            # Crear un diccionario de mapeo: course_module_id -> module_unique_id
            module_mapping = modules_df.set_index('course_module_id')['module_unique_id'].to_dict()
            return module_mapping, modules_df[['course_module_id', 'module_unique_id', 'year', 'course_id', 'period', 'week']]
        except Exception as e:
            self.logger.error(f"Error al cargar el mapeo de módulos: {str(e)}")
            raise

    def _generate_activity_sequences(self, logs_df, modules_df, student_courses_df):
        """
        Genera secuencias de actividades cronológicas por estudiante-curso-período-año
        Incluye todos los estudiantes-cursos, incluso aquellos sin actividad registrada
        """
        if not logs_df.empty:
            # Convertir timestamp a datetime
            logs_df['access_datetime'] = pd.to_datetime(logs_df['timecreated'], unit='s')

            # Merge con información de módulos para obtener período y week
            logs_with_modules = logs_df.merge(
                modules_df[['course_module_id', 'module_unique_id', 'period', 'week']], 
                on='course_module_id', 
                how='left'
            )

            # Filtrar solo logs que tienen module_unique_id (módulos válidos)
            logs_with_modules = logs_with_modules.dropna(subset=['module_unique_id'])

            if not logs_with_modules.empty:
                # Ordenar por estudiante, curso, período y tiempo de acceso
                logs_with_modules = logs_with_modules.sort_values([
                    'documento_identificación', 'courseid', 'year', 'period', 'access_datetime'
                ])

                # Remover accesos duplicados del mismo estudiante al mismo módulo el mismo día
                logs_with_modules['access_date'] = logs_with_modules['access_datetime'].dt.date
                logs_unique = logs_with_modules.drop_duplicates(
                    subset=['documento_identificación', 'courseid', 'year', 'period', 'course_module_id', 'access_date'],
                    keep='first'
                )

                # Agrupar por estudiante-asignatura-período-año (COMBINANDO Moodle y Edukrea)
                sequences_with_activity = (
                    logs_unique.groupby([
                        'documento_identificación', 'id_asignatura', 'id_grado', 'sede', 'year', 'period'
                    ])
                    .agg({
                        'module_unique_id': lambda x: ','.join(x.astype(str)),
                        'access_datetime': ['min', 'max', 'count'],
                        'platform': lambda x: ','.join(sorted(set(str(p) for p in x if pd.notna(p))))
                    })
                    .reset_index()
                )

                # Aplanar nombres de columnas
                sequences_with_activity.columns = [
                    'documento_identificación', 'id_asignatura', 'id_grado', 'sede', 'year', 'period',
                    'activity_sequence', 'first_access', 'last_access', 'total_accesses', 'platforms_used'
                ]
            else:
                sequences_with_activity = pd.DataFrame()
        else:
            sequences_with_activity = pd.DataFrame()

        # Crear base de estudiantes-asignaturas-períodos (combinando ambas plataformas)
        # Agrupar student_courses por asignatura (no por curso individual)
        base_asignaturas = student_courses_df.groupby([
            'documento_identificación', 'id_asignatura', 'id_grado', 'sede', 'year'
        ]).agg({
            'platform': lambda x: ','.join(sorted(set(x))),  # Plataformas disponibles
            'course_id': lambda x: ','.join(x.astype(str)),  # IDs de cursos
            'course_name': 'first'  # Tomar el primer nombre de curso
        }).reset_index()

        # Crear registros por período (1-4) para cada asignatura
        base_periods = []
        for period in [1, 2, 3, 4]:
            period_base = base_asignaturas.copy()
            period_base['period'] = period
            base_periods.append(period_base)

        complete_base = pd.concat(base_periods, ignore_index=True)

        # Asegurar tipos de datos consistentes para el merge
        complete_base['year'] = complete_base['year'].astype(int)
        complete_base['id_grado'] = complete_base['id_grado'].astype(int)
        complete_base['id_asignatura'] = complete_base['id_asignatura'].astype(int)
        complete_base['period'] = complete_base['period'].astype(float)

        # Hacer LEFT JOIN para preservar todos los estudiantes-asignaturas-períodos
        if not sequences_with_activity.empty:
            # Asegurar tipos consistentes en sequences_with_activity también
            sequences_with_activity['year'] = sequences_with_activity['year'].astype(int)
            sequences_with_activity['id_grado'] = sequences_with_activity['id_grado'].astype(int)
            sequences_with_activity['id_asignatura'] = sequences_with_activity['id_asignatura'].astype(int)
            sequences_with_activity['period'] = sequences_with_activity['period'].astype(float)

            final_sequences = complete_base.merge(
                sequences_with_activity,
                on=['documento_identificación', 'id_asignatura', 'id_grado', 'sede', 'year', 'period'],
                how='left'
            )
        else:
            final_sequences = complete_base.copy()
            final_sequences['activity_sequence'] = ''
            final_sequences['first_access'] = None
            final_sequences['last_access'] = None
            final_sequences['total_accesses'] = 0
            final_sequences['platforms_used'] = ''

        # Rellenar valores nulos para estudiantes sin actividad
        final_sequences['activity_sequence'] = final_sequences['activity_sequence'].fillna('')
        final_sequences['total_accesses'] = final_sequences['total_accesses'].fillna(0)
        final_sequences['platforms_used'] = final_sequences['platforms_used'].fillna('')

        # Filtrar solo registros que tienen actividad para evitar explosión de datos
        final_sequences = final_sequences[final_sequences['activity_sequence'] != '']
        return final_sequences

    def process_student_activity_sequences(self):
        student_courses_file = "data/interim/moodle/student_courses.csv"
        logs_table = "logstore_standard_log"

        # Cargar mapeo de módulos
        self.logger.info("Cargando mapeo de módulos...")
        module_mapping, modules_df = self._load_module_mapping()

        # Cargar base de estudiantes-cursos
        self.logger.info("Cargando base de estudiantes-cursos...")
        student_courses_df = pd.read_csv(student_courses_file)

        # Obtener logs de acceso a módulos para 2024
        self.logger.info("Procesando logs de 2024...")
        year = 2024
        logs_parquet = MoodlePathResolver.get_paths(year, logs_table)[0]
        logs_2024 = self._get_module_access_logs(year, logs_parquet, student_courses_file, platform='moodle')

        # Para 2025, solo obtener logs de Edukrea (no de Moodle)
        self.logger.info("Procesando logs de Edukrea para 2025...")
        logs_parquet = MoodlePathResolver.get_paths("Edukrea", logs_table)[0]
        logs_edukrea_2025 = self._get_module_access_logs(2025, logs_parquet, student_courses_file, platform='edukrea')

        # Concatenar todos los logs (2024 Moodle + 2025 Edukrea únicamente)
        self.logger.info("Combinando logs de todas las plataformas...")
        all_logs = pd.concat([logs_2024, logs_edukrea_2025], ignore_index=True)

        if all_logs.empty:
            self.logger.warning("No se encontraron logs de acceso a módulos")
            return

        # Generar secuencias de actividades
        self.logger.info("Generando secuencias de actividades...")
        activity_sequences = self._generate_activity_sequences(all_logs, modules_df, student_courses_df)

        # Guardar resultado
        output_file = "data/interim/moodle/student_activity_sequences.csv"
        self.save_to_csv(activity_sequences, output_file)

        self.logger.info(f"Secuencias de actividades generadas: {len(activity_sequences)} registros")
        self.logger.info(f"Estudiantes únicos: {activity_sequences['documento_identificación'].nunique()}")
        self.logger.info(f"Asignaturas únicas: {activity_sequences['id_asignatura'].nunique()}")
        self.logger.info(f"Combinaciones estudiante-asignatura-período con actividad: {len(activity_sequences)}")

        # Mostrar uso de plataformas
        moodle_only = (activity_sequences['platforms_used'] == 'moodle').sum()
        edukrea_only = (activity_sequences['platforms_used'] == 'edukrea').sum()
        both_platforms = (activity_sequences['platforms_used'].str.contains(',', na=False)).sum()

        self.logger.info(f"Solo Moodle: {moodle_only}")
        self.logger.info(f"Solo Edukrea: {edukrea_only}")
        self.logger.info(f"Ambas plataformas: {both_platforms}")

if __name__ == "__main__":
    processor = StudentCourseActivityStringsProcessor()
    processor.process_student_activity_sequences()
    processor.logger.info("Student activity sequences processed successfully.")
    processor.close()
