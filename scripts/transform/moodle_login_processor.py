import duckdb
import pandas as pd
import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.period_utils import PeriodUtils


class MoodleLoginProcessor:
    """
    Procesa datos de login de Moodle para generar estadísticas de acceso.

    Esta clase analiza los registros de login de usuarios en Moodle y genera
    estadísticas sobre patrones de acceso, incluyendo frecuencia por día,
    jornada, y periodo académico.
    """

    def __init__(self):
        self.con = duckdb.connect()
        self.logger = logging.getLogger(__name__)
        self.period_utils = PeriodUtils()

    def __del__(self):
        if hasattr(self, "con") and self.con:
            self.con.close()

    def close(self):
        if hasattr(self, "con") and self.con:
            self.con.close()
            self.con = None

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
        data_2024 = self._load_moodle_data(
            logs_parquet="data/raw/moodle/2024/Log/mdlvf_logstore_standard_log.parquet",
            students_enrollment="data/interim/estudiantes/enrollments.csv",
            year=2024,
        )
        data_2025 = self._load_moodle_data(
            logs_parquet="data/raw/moodle/2025/Log/mdlvf_logstore_standard_log.parquet",
            students_enrollment="data/interim/estudiantes/enrollments.csv",
            year=2025,
        )

        combined_data = pd.concat([data_2024, data_2025])

        # --- Paso 2: Convertir timecreated a formato de fecha
        combined_data["timecreated"] = pd.to_datetime(combined_data["timecreated"], unit="s").dt.tz_localize("UTC")
        combined_data["fecha_local"] = combined_data["timecreated"].dt.tz_convert("America/Bogota")
        combined_data["moodle_user_id"] = combined_data["moodle_user_id"].astype(str)

        # --- Paso 3: Asignar periodo
        combined_data["periodo"] = combined_data["fecha_local"].apply(self.period_utils.assign_period)

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
        resultado_no_vacaciones = resultado_no_vacaciones.sort_values(["moodle_user_id", "year", "periodo", "fecha_local"])
        resultado_no_vacaciones["delta"] = resultado_no_vacaciones.groupby(["moodle_user_id", "year", "periodo"])["fecha_local"].diff()

        # --- Paso 7: Resumen general por usuario + periodo + año ---
        # Total logins (sin filtrar vacaciones)
        summary = combined_data.groupby(["moodle_user_id", "year", "periodo"]).agg(count_login=("fecha_local", "count")).reset_index()

        # Agregar `max_inactividad` desde el nuevo cálculo (excluyendo vacaciones)
        resultado_no_vacaciones["delta_horas"] = resultado_no_vacaciones["delta"].dt.total_seconds() / 3600
        max_inactividad = (
            resultado_no_vacaciones.groupby(["moodle_user_id", "year", "periodo"])["delta_horas"].max().reset_index(name="max_inactividad")
        )
        summary = summary.merge(max_inactividad, on=["moodle_user_id", "year", "periodo"], how="left")

        # Rellenar valores faltantes de max_inactividad con la duración del periodo en horas
        for idx, row in summary.iterrows():
            if pd.isna(row["max_inactividad"]):
                # Si no hay registros de inactividad, usar la duración total del periodo convertido a horas
                duracion_periodo = self.period_utils.calculate_period_duration(int(row["year"]), row["periodo"])
                summary.at[idx, "max_inactividad"] = duracion_periodo.total_seconds() / 3600

        # --- Paso 8: Conteo por día de la semana ---
        conteo_dias = combined_data.pivot_table(
            index=["moodle_user_id", "year", "periodo", "documento_identificación"],
            columns="dia",
            values="fecha_local",
            aggfunc="count",
            fill_value=0,
        ).reset_index()

        # Renombrar días
        dias = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for d in dias:
            if d in conteo_dias.columns:
                conteo_dias.rename(columns={d: f"count_login_{d[:3]}"}, inplace=True)
            else:
                conteo_dias[f"count_login_{d[:3]}"] = 0

        # --- Paso 7: Conteo por jornada ---
        conteo_jornada = combined_data.pivot_table(
            index=["moodle_user_id", "year", "periodo"], columns="jornada", values="fecha_local", aggfunc="count", fill_value=0
        ).reset_index()

        # Renombrar jornadas
        jornadas = ["madrugada", "mañana", "tarde", "noche"]
        for j in jornadas:
            if j in conteo_jornada.columns:
                conteo_jornada.rename(columns={j: f"count_jornada_{j}"}, inplace=True)
            else:
                conteo_jornada[f"count_jornada_{j}"] = 0

        # --- Paso 9: Combinar todo ---
        df_final = summary.merge(conteo_dias, on=["moodle_user_id", "year", "periodo"], how="left")
        df_final = df_final.merge(conteo_jornada, on=["moodle_user_id", "year", "periodo"], how="left")

        # Asegurar orden de columnas (opcional)
        columnas_finales = (
            ["moodle_user_id", "year", "periodo", "count_login", "max_inactividad", "documento_identificación"]
            + [f"count_login_{d[:3]}" for d in dias]
            + [f"count_jornada_{j}" for j in jornadas]
        )

        df_final = df_final[columnas_finales]

        # --- Paso 10: Guardar resultados ---
        output_path = "data/interim/moodle/moodle_logins.csv"
        df_final.to_csv(output_path, index=False)
        self.logger.info(f"Resultados guardados en: {output_path}")
        self.logger.info("Proceso de logins de Moodle completado.")


if __name__ == "__main__":
    processor = MoodleLoginProcessor()
    processor.process_moodle_logins()
