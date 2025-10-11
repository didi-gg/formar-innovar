import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.moodle_path_resolver import MoodlePathResolver
from utils.academic_period_utils import AcademicPeriodUtils
from utils.base_script import BaseScript


class StudentLoginsProcessor(BaseScript):
    """
    Procesa datos de login de Moodle para generar estadísticas de acceso.

    Esta clase analiza los registros de login de usuarios en Moodle y genera
    estadísticas sobre patrones de acceso, incluyendo frecuencia por día,
    jornada, y periodo académico.
    """

    def __init__(self):
        self.period_utils = AcademicPeriodUtils()
        super().__init__()

    def _load_moodle_data(self, logs_parquet, students_enrollment, year=2024):
        try:
            sql = f"""
                SELECT 
                    logs.userid AS moodle_user_id,
                    logs.timecreated,
                    estudiantes.documento_identificación,
                    estudiantes.id_grado,
                    estudiantes.year   
                FROM '{logs_parquet}' AS logs
                INNER JOIN '{students_enrollment}' AS estudiantes
                    ON logs.userid = estudiantes.moodle_user_id
                WHERE logs.eventname = '\\core\\event\\user_loggedin'
                AND EXTRACT(YEAR FROM to_timestamp(logs.timecreated)) = {year}
                AND estudiantes.year = {year}
                ORDER BY logs.timecreated
            """
            result = self.con.execute(sql).df()
            return result
        except Exception as e:
            self.logger.error(f"Error cargando datos para el año {year}: {str(e)}")
            raise

    def process_moodle_logins(self):
        # --- Paso 1: Cargar datos de Moodle 2024 y 2025
        logs_path_2024 = MoodlePathResolver.get_paths(2024, "logstore_standard_log")[0]
        logs_path_2025 = MoodlePathResolver.get_paths(2025, "logstore_standard_log")[0]
        logs_path_edukrea = MoodlePathResolver.get_paths("Edukrea", "logstore_standard_log")[0]

        data_2024 = self._load_moodle_data(
            logs_parquet=logs_path_2024,
            students_enrollment="data/interim/estudiantes/enrollments.csv",
            year=2024,
        )
        data_2025_raw = self._load_moodle_data(
            logs_parquet=logs_path_2025,
            students_enrollment="data/interim/estudiantes/enrollments.csv",
            year=2025,
        )
        edukrea_data = self._load_moodle_data(
            logs_parquet=logs_path_edukrea,
            students_enrollment="data/interim/estudiantes/enrollments.csv",
            year=2025,
        )

        combined_data = pd.concat([data_2024, data_2025_raw, edukrea_data], ignore_index=True)

        # --- Paso 2: Convertir timecreated a formato de fecha
        combined_data["timecreated"] = pd.to_datetime(combined_data["timecreated"], unit="s").dt.tz_localize("UTC")
        combined_data["fecha_local"] = combined_data["timecreated"].dt.tz_convert("America/Bogota")
        combined_data["documento_identificación"] = combined_data["documento_identificación"].astype(str)

        # --- Paso 3: Asignar periodo
        combined_data["periodo"] = combined_data["fecha_local"].apply(self.period_utils.determine_period_from_date)

        # --- Paso 4: Marcar logins en vacaciones ---
        combined_data["en_vacaciones"] = combined_data["fecha_local"].apply(self.period_utils.is_vacation)

        # --- Paso 5: Día de la semana y jornada ---
        combined_data["hora"] = combined_data["fecha_local"].dt.hour
        combined_data["dia"] = combined_data["fecha_local"].dt.day_name().str.lower()
        combined_data["jornada"] = combined_data["hora"].apply(self.period_utils.classify_daytime)

        # --- Paso 6: Calcular inactividad ---
        # Usar solo logins fuera de vacaciones para calcular `delta`
        resultado_no_vacaciones = combined_data[~combined_data["en_vacaciones"]].copy()

        # Ordenar y calcular diferencias por usuario + periodo + año
        resultado_no_vacaciones = resultado_no_vacaciones.sort_values(["documento_identificación", "year", "periodo", "fecha_local"])
        resultado_no_vacaciones["delta"] = resultado_no_vacaciones.groupby(["documento_identificación", "year", "periodo"])["fecha_local"].diff()

        # --- Paso 7: Resumen general por usuario + periodo + año ---
        # Total logins (sin filtrar vacaciones)
        summary = combined_data.groupby(["documento_identificación", "year", "periodo"]).agg(count_login=("fecha_local", "count")).reset_index()

        # Agregar `max_inactividad` desde el nuevo cálculo (excluyendo vacaciones)
        resultado_no_vacaciones["delta_horas"] = resultado_no_vacaciones["delta"].dt.total_seconds() / 3600
        max_inactividad = (
            resultado_no_vacaciones.groupby(["documento_identificación", "year", "periodo"])["delta_horas"].max().reset_index(name="max_inactividad")
        )
        summary = summary.merge(max_inactividad, on=["documento_identificación", "year", "periodo"], how="left")

        # Rellenar valores faltantes de max_inactividad con la duración del periodo en horas
        for idx, row in summary.iterrows():
            if pd.isna(row["max_inactividad"]):
                # Si no hay registros de inactividad, usar la duración total del periodo convertido a horas
                duracion_periodo = self.period_utils.calculate_period_duration(int(row["year"]), row["periodo"])
                summary.at[idx, "max_inactividad"] = duracion_periodo.total_seconds() / 3600

        # --- Paso 7.5: Calcular features adicionales de comportamiento ---

        # 1. gaps_between_sessions_avg - Promedio de horas entre sesiones
        avg_gaps = (
            resultado_no_vacaciones.groupby(["documento_identificación", "year", "periodo"])["delta_horas"]
            .mean().reset_index(name="gaps_between_sessions_avg")
        )
        summary = summary.merge(avg_gaps, on=["documento_identificación", "year", "periodo"], how="left")

        # 2. login_regularity_score - Score de regularidad basado en varianza de intervalos
        std_gaps = (
            resultado_no_vacaciones.groupby(["documento_identificación", "year", "periodo"])["delta_horas"]
            .std().reset_index(name="std_gaps")
        )
        summary = summary.merge(std_gaps, on=["documento_identificación", "year", "periodo"], how="left")

        # Calcular regularity score (1 - coef_variacion normalizado)
        # Manejar casos donde gaps_between_sessions_avg es 0 o NaN
        summary["coef_variacion"] = summary["std_gaps"] / (summary["gaps_between_sessions_avg"] + 1e-6)  # Evitar división por 0
        summary["login_regularity_score"] = (1 - summary["coef_variacion"]).clip(0, 1)
        summary["login_regularity_score"] = summary["login_regularity_score"].fillna(0)

        # Limpiar columna auxiliar
        summary = summary.drop(columns=["coef_variacion"])

        # 3. longest_inactivity_streak - Mayor periodo de inactividad en días
        summary["longest_inactivity_streak"] = summary["max_inactividad"] / 24  # Convertir horas a días

        # 4. consecutive_days_max y engagement_decay - Requieren análisis por días únicos
        behavioral_features = []

        for _, group_info in summary.iterrows():
            doc_id = group_info["documento_identificación"]
            year = group_info["year"]
            periodo = group_info["periodo"]

            # Filtrar datos del usuario específico
            user_data = combined_data[
                (combined_data["documento_identificación"] == doc_id) & 
                (combined_data["year"] == year) & 
                (combined_data["periodo"] == periodo)
            ].copy()

            if len(user_data) == 0:
                behavioral_features.append({
                    "documento_identificación": doc_id,
                    "year": year,
                    "periodo": periodo,
                    "consecutive_days_max": 0,
                    "engagement_decay": 0
                })
                continue

            # Obtener días únicos con actividad
            user_data["fecha_solo"] = user_data["fecha_local"].dt.date
            dias_unicos = sorted(user_data["fecha_solo"].unique())

            # 4a. consecutive_days_max - Días consecutivos máximos
            max_consecutive = 0
            current_consecutive = 1

            if len(dias_unicos) > 1:
                for i in range(1, len(dias_unicos)):
                    if (dias_unicos[i] - dias_unicos[i-1]).days == 1:
                        current_consecutive += 1
                    else:
                        max_consecutive = max(max_consecutive, current_consecutive)
                        current_consecutive = 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                max_consecutive = 1 if len(dias_unicos) == 1 else 0

            # 4b. engagement_decay - Tendencia de actividad en el tiempo
            # Correlación entre número de orden del día y cantidad de logins
            if len(dias_unicos) > 2:
                daily_counts = user_data.groupby("fecha_solo").size().reset_index(name="login_count")
                daily_counts["day_order"] = range(len(daily_counts))

                # Validar que hay variabilidad en los datos antes de calcular correlación
                if (daily_counts["login_count"].std() > 0 and 
                    daily_counts["day_order"].std() > 0 and 
                    len(daily_counts) > 2):

                    # Suprimir warnings de numpy para esta operación específica
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        correlation = daily_counts["day_order"].corr(daily_counts["login_count"])

                    engagement_decay = -correlation if not pd.isna(correlation) else 0
                else:
                    engagement_decay = 0
            else:
                engagement_decay = 0

            behavioral_features.append({
                "documento_identificación": doc_id,
                "year": year,
                "periodo": periodo,
                "consecutive_days_max": max_consecutive,
                "engagement_decay": engagement_decay
            })

        behavioral_df = pd.DataFrame(behavioral_features)
        summary = summary.merge(behavioral_df, on=["documento_identificación", "year", "periodo"], how="left")

        # 5. activity_percentile - Percentil de actividad vs todos los estudiantes
        summary["activity_percentile"] = summary["count_login"].rank(pct=True) * 100

        # Limpiar columnas auxiliares
        summary = summary.drop(columns=["std_gaps"])

        # --- Paso 8: Conteo por día de la semana ---
        conteo_dias = combined_data.pivot_table(
            index=["documento_identificación", "year", "periodo"],
            columns="dia",
            values="fecha_local",
            aggfunc="count",
            fill_value=0,
        ).reset_index()

        # Renombrar días
        dias = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for d in dias:
            if d in conteo_dias.columns:
                conteo_dias = conteo_dias.rename(columns={d: f"count_login_{d[:3]}"})
            else:
                conteo_dias[f"count_login_{d[:3]}"] = 0

        # --- Paso 7: Conteo por jornada ---
        conteo_jornada = combined_data.pivot_table(
            index=["documento_identificación", "year", "periodo"], columns="jornada", values="fecha_local", aggfunc="count", fill_value=0
        ).reset_index()

        # Renombrar jornadas
        jornadas = ["madrugada", "mañana", "tarde", "noche"]
        for j in jornadas:
            if j in conteo_jornada.columns:
                conteo_jornada = conteo_jornada.rename(columns={j: f"count_jornada_{j}"})
            else:
                conteo_jornada[f"count_jornada_{j}"] = 0

        # --- Paso 9: Combinar todo ---
        df_final = summary.merge(conteo_dias, on=["documento_identificación", "year", "periodo"], how="left")
        df_final = df_final.merge(conteo_jornada, on=["documento_identificación", "year", "periodo"], how="left")

        # --- Paso 10: Calcular métricas adicionales ---
        # Calcular login_consistency (desviación estándar de logins por día)
        columnas_dias = [f"count_login_{d[:3]}" for d in dias]
        df_final["login_consistency"] = df_final[columnas_dias].std(axis=1)

        # Calcular dia_preferido (día con mayor número de logins)
        columnas_dias_completas = ["count_login_mon","count_login_tue","count_login_wed","count_login_thu","count_login_fri","count_login_sat","count_login_sun"]
        nombres_dias = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]

        # Crear DataFrame temporal para encontrar el día con mayor valor
        df_temp_dias = df_final[columnas_dias_completas].copy()
        df_temp_dias.columns = nombres_dias
        df_final["dia_preferido"] = df_temp_dias.idxmax(axis=1)

        # Si todos los valores son 0, asignar "sin_actividad"
        mask_sin_actividad_dias = df_final[columnas_dias_completas].sum(axis=1) == 0
        df_final.loc[mask_sin_actividad_dias, "dia_preferido"] = "sin_actividad"

        # Calcular jornada_preferida (jornada con mayor número de logins)
        columnas_jornadas = [f"count_jornada_{j}" for j in jornadas]
        # Crear un DataFrame temporal para encontrar la jornada con mayor valor
        df_temp = df_final[columnas_jornadas].copy()
        df_temp.columns = jornadas  # Cambiar nombres temporalmente para facilitar el cálculo
        df_final["jornada_preferida"] = df_temp.idxmax(axis=1)

        # En caso de empate, tomar la primera jornada alfabéticamente
        # Si todos los valores son 0, asignar "sin_actividad"
        mask_sin_actividad = df_final[columnas_jornadas].sum(axis=1) == 0
        df_final.loc[mask_sin_actividad, "jornada_preferida"] = "sin_actividad"

        # Asegurar orden de columnas (opcional)
        columnas_finales = (
            ["documento_identificación", "year", "periodo", "count_login", "max_inactividad"]
            + [f"count_login_{d[:3]}" for d in dias]
            + [f"count_jornada_{j}" for j in jornadas]
            + ["login_consistency", "dia_preferido", "jornada_preferida"]
            + ["login_regularity_score", "consecutive_days_max", "gaps_between_sessions_avg", 
               "engagement_decay", "activity_percentile", "longest_inactivity_streak"]
        )

        df_final = df_final[columnas_finales]

        # --- Paso 11: Completar logins faltantes para estudiantes con actividad ---
        self.logger.info("=== DETECTANDO ESTUDIANTES CON ACTIVIDAD PERO SIN LOGINS ===")

        # Cargar enrollments para obtener todos los estudiantes matriculados
        enrollments_df = pd.read_csv("data/interim/estudiantes/enrollments.csv")
        enrollments_df['documento_identificación'] = enrollments_df['documento_identificación'].astype(str).str.strip()

        # Cargar actividad de estudiantes desde múltiples fuentes
        try:
            # Fuente 1: course_activity_summary.csv
            activity_df = pd.read_csv("data/interim/moodle/course_activity_summary.csv")
            self.logger.info(f"Archivo de actividad cargado: {activity_df.shape}")

            # Filtrar solo asignaturas Moodle (1, 2, 3, 4)
            activity_df = activity_df[activity_df['id_asignatura'].isin([1, 2, 3, 4])]

            # Obtener combinaciones únicas de estudiantes con actividad por (documento, year, period)
            students_with_activity = activity_df.groupby(
                ['documento_identificación', 'year', 'period']
            ).agg({
                'courseid': 'count',  # Número de cursos con actividad
                'course_viewed': 'sum',  # Total de vistas a cursos
                'hvp_module_viewed': 'sum',  # Vistas a módulos H5P
                'assign_module_viewed': 'sum',  # Vistas a tareas
                'forum_module_viewed': 'sum',  # Vistas a foros
                'resource_module_viewed': 'sum',  # Vistas a recursos
            }).reset_index()

            # Calcular actividad total
            students_with_activity['total_activity'] = (
                students_with_activity['course_viewed'] +
                students_with_activity['hvp_module_viewed'] +
                students_with_activity['assign_module_viewed'] +
                students_with_activity['forum_module_viewed'] +
                students_with_activity['resource_module_viewed']
            )

            # Filtrar solo los que tienen actividad real
            students_with_activity = students_with_activity[students_with_activity['total_activity'] > 0]
            self.logger.info(f"Estudiantes con actividad (course_activity_summary): {students_with_activity.shape[0]}")

            # Fuente 2: sequence_analysis_features.csv (para capturar actividad que no está en course_activity)
            try:
                sequence_df = pd.read_csv("data/interim/moodle/sequence_analysis_features.csv")

                # Filtrar solo con actividad real (total_accesses > 0)
                sequence_activity = sequence_df[sequence_df['total_accesses'] > 0].copy()
                sequence_activity = sequence_activity[sequence_activity['id_asignatura'].isin([1, 2, 3, 4])]

                # Agregar por (documento, year, period)
                sequence_activity_grouped = sequence_activity.groupby(
                    ['documento_identificación', 'year', 'period']
                ).agg({
                    'total_accesses': 'sum'
                }).reset_index()
                sequence_activity_grouped.rename(columns={'total_accesses': 'total_activity'}, inplace=True)
                sequence_activity_grouped['courseid'] = 1  # Al menos 1 curso

                self.logger.info(f"Estudiantes con actividad (sequences): {sequence_activity_grouped.shape[0]}")

                # Combinar ambas fuentes, manteniendo el máximo de actividad
                combined_activity = pd.concat([students_with_activity, sequence_activity_grouped], ignore_index=True)
                students_with_activity = combined_activity.groupby(
                    ['documento_identificación', 'year', 'period']
                ).agg({
                    'total_activity': 'sum',
                    'courseid': 'max'
                }).reset_index()

                self.logger.info(f"Total estudiantes únicos con actividad (combinado): {students_with_activity.shape[0]}")

            except FileNotFoundError:
                raise FileNotFoundError("No se encontró archivo de secuencias. Usando solo course_activity_summary.")
            except Exception as e:
                raise Exception(f"Error al cargar secuencias: {str(e)}. Continuando solo con course_activity_summary.")

            # Identificar estudiantes con actividad pero sin login
            # Normalizar el dataframe de logins existentes para comparación
            existing_logins_df = df_final[['documento_identificación', 'year', 'periodo']].copy()
            existing_logins_df['documento_identificación'] = existing_logins_df['documento_identificación'].astype(str).str.strip()
            existing_logins_df['year'] = existing_logins_df['year'].astype(int)
            existing_logins_df['periodo'] = existing_logins_df['periodo'].astype(int)

            existing_logins = set(
                existing_logins_df.itertuples(index=False, name=None)
            )

            # Normalizar el dataframe de actividad para comparación
            students_with_activity['documento_identificación'] = students_with_activity['documento_identificación'].astype(str).str.strip()
            students_with_activity['year'] = students_with_activity['year'].astype(int)
            students_with_activity['period'] = students_with_activity['period'].astype(int)

            missing_logins = []
            for _, row in students_with_activity.iterrows():
                key = (row['documento_identificación'], row['year'], row['period'])
                if key not in existing_logins:
                    missing_logins.append({
                        'documento_identificación': row['documento_identificación'],
                        'year': row['year'],
                        'period': row['period'],
                        'total_activity': int(row['total_activity']),
                        'num_courses': int(row['courseid'])
                    })

            self.logger.info(f"Estudiantes con actividad pero sin logins: {len(missing_logins)}")

            if len(missing_logins) > 0:
                # Crear registros sintéticos basados en actividad
                synthetic_logins = []

                for student in missing_logins:
                    # Estimar logins basados en actividad
                    # Regla heurística: ~1 login por cada 10-15 actividades
                    estimated_logins = max(1, int(student['total_activity'] / 12))

                    # Distribuir logins en días de semana (más actividad en días laborales)
                    # Distribución típica: 20% lun, 20% mar, 20% mié, 20% jue, 18% vie, 1% sáb, 1% dom
                    mon = int(estimated_logins * 0.20)
                    tue = int(estimated_logins * 0.20)
                    wed = int(estimated_logins * 0.20)
                    thu = int(estimated_logins * 0.20)
                    fri = int(estimated_logins * 0.18)
                    sat = int(estimated_logins * 0.01)
                    sun = max(0, estimated_logins - (mon + tue + wed + thu + fri + sat))

                    # Distribuir en jornadas (más en tarde y noche para estudiantes)
                    jornada_madrugada = int(estimated_logins * 0.02)
                    jornada_manana = int(estimated_logins * 0.46)
                    jornada_tarde = int(estimated_logins * 0.46)
                    jornada_noche = max(0, estimated_logins - (jornada_madrugada + jornada_manana + jornada_tarde))

                    # Calcular día y jornada preferida
                    dias_counts = [mon, tue, wed, thu, fri, sat, sun]
                    dias_nombres = ['lunes', 'martes', 'miercoles', 'jueves', 'viernes', 'sabado', 'domingo']
                    dia_preferido = dias_nombres[dias_counts.index(max(dias_counts))] if max(dias_counts) > 0 else 'sin_actividad'

                    jornadas_counts = [jornada_madrugada, jornada_manana, jornada_tarde, jornada_noche]
                    jornadas_nombres = ['madrugada', 'mañana', 'tarde', 'noche']
                    jornada_preferida = jornadas_nombres[jornadas_counts.index(max(jornadas_counts))] if max(jornadas_counts) > 0 else 'sin_actividad'

                    # Calcular métricas estimadas
                    # login_consistency: varianza de logins por día
                    login_consistency = pd.Series(dias_counts).std()

                    # Estimar inactividad basada en duración del periodo
                    period_duration_hours = self.period_utils.calculate_period_duration(
                        student['year'], str(int(student['period']))  # Convertir a string "1", "2", "3", "4"
                    ).total_seconds() / 3600

                    # Si tiene actividad regular, asumir menor inactividad
                    # Inactividad máxima ~ duración del periodo / (logins estimados + 1)
                    max_inactividad = period_duration_hours / (estimated_logins + 1)

                    synthetic_record = {
                        'documento_identificación': student['documento_identificación'],
                        'year': student['year'],
                        'periodo': int(student['period']),  # Guardar como int para consistencia
                        'count_login': estimated_logins,
                        'max_inactividad': max_inactividad,
                        'count_login_mon': mon,
                        'count_login_tue': tue,
                        'count_login_wed': wed,
                        'count_login_thu': thu,
                        'count_login_fri': fri,
                        'count_login_sat': sat,
                        'count_login_sun': sun,
                        'count_jornada_madrugada': jornada_madrugada,
                        'count_jornada_mañana': jornada_manana,
                        'count_jornada_tarde': jornada_tarde,
                        'count_jornada_noche': jornada_noche,
                        'login_consistency': login_consistency,
                        'dia_preferido': dia_preferido,
                        'jornada_preferida': jornada_preferida,
                        'login_regularity_score': 0.5,  # Valor medio por falta de datos exactos
                        'consecutive_days_max': min(estimated_logins, 5),  # Conservador
                        'gaps_between_sessions_avg': max_inactividad / 2,  # Promedio estimado
                        'engagement_decay': 0.0,  # Neutral por falta de datos temporales
                        'activity_percentile': 25,  # Percentil bajo por datos sintéticos
                        'longest_inactivity_streak': max_inactividad / 24  # Convertir a días
                    }
                    synthetic_logins.append(synthetic_record)

                synthetic_df = pd.DataFrame(synthetic_logins)
                self.logger.info(f"Registros sintéticos creados: {synthetic_df.shape[0]}")

                # Combinar con logins reales
                df_final = pd.concat([df_final, synthetic_df], ignore_index=True)

                # Verificar duplicados después de agregar sintéticos
                duplicates_check = df_final.duplicated(subset=['documento_identificación', 'year', 'periodo']).sum()
                if duplicates_check > 0:
                    self.logger.warning(f"Duplicados eliminados. Total de registros finales: {len(df_final)}")
                    df_final = df_final.drop_duplicates(subset=['documento_identificación', 'year', 'periodo'], keep='first')
                    self.logger.warning(f"¡ADVERTENCIA! Se detectaron {duplicates_check} duplicados después de agregar registros sintéticos")                    

                # Recalcular percentiles con todos los registros
                df_final['activity_percentile'] = df_final['count_login'].rank(pct=True) * 100

                self.logger.info(f"\nTotal de registros finales (reales + sintéticos): {df_final.shape[0]}")
                self.logger.info(f"Registros reales: {len(df_final) - len(synthetic_logins)}")
                self.logger.info(f"Registros sintéticos: {len(synthetic_logins)}")
                self.logger.info(f"Porcentaje sintético: {100*len(synthetic_logins)/len(df_final):.2f}%")

                # Guardar registros sintéticos para auditoría
                audit_path = "data/interim/moodle/synthetic_logins_audit.csv"
                synthetic_df.to_csv(audit_path, index=False)
                self.logger.info(f"Auditoría de registros sintéticos guardada en: {audit_path}")
            else:
                self.logger.info("No se encontraron estudiantes con actividad sin logins")

        except FileNotFoundError:
            self.logger.warning("No se encontró archivo de actividad. Saltando completación de logins.")
        except Exception as e:
            self.logger.error(f"Error al completar logins faltantes: {str(e)}")
            self.logger.warning("Continuando con los logins existentes...")

        # --- Paso 11.5: Rellenar valores null en métricas ---
        self.logger.info("=== RELLENANDO VALORES NULL EN MÉTRICAS ===")

        # Para registros con count_login > 1 pero gaps_between_sessions_avg null, usar max_inactividad/2
        mask_missing_gaps = (df_final['gaps_between_sessions_avg'].isna()) & (df_final['count_login'] > 1)
        if mask_missing_gaps.sum() > 0:
            self.logger.info(f"Rellenando {mask_missing_gaps.sum()} registros con gaps_between_sessions_avg null")
            df_final.loc[mask_missing_gaps, 'gaps_between_sessions_avg'] = (
                df_final.loc[mask_missing_gaps, 'max_inactividad'] / 2
            )

        # Para registros con count_login = 1 y gaps null, dejar como 0 (no hay gaps con 1 solo login)
        mask_single_login = (df_final['gaps_between_sessions_avg'].isna()) & (df_final['count_login'] == 1)
        if mask_single_login.sum() > 0:
            self.logger.info(f"Estableciendo gaps_between_sessions_avg = 0 para {mask_single_login.sum()} registros con 1 solo login")
            df_final.loc[mask_single_login, 'gaps_between_sessions_avg'] = 0

        # Verificar otros nulls problemáticos
        null_counts = df_final[['max_inactividad', 'gaps_between_sessions_avg', 'longest_inactivity_streak']].isnull().sum()
        if null_counts.sum() > 0:
            self.logger.warning(f"Valores null restantes en métricas:\n{null_counts[null_counts > 0]}")

        # --- Paso 12: Guardar resultados ---
        output_path = "data/interim/moodle/student_login_moodle.csv"
        self.save_to_csv(df_final, output_path)
        self.logger.info(f"Resultados guardados en: {output_path}")
        self.logger.info("Proceso de logins de Moodle completado.")


if __name__ == "__main__":
    processor = StudentLoginsProcessor()
    processor.process_moodle_logins()
    processor.close()
    processor.logger.info("Student logins processed successfully.")
