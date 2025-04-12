import duckdb
import pandas as pd
import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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

        self.p1_start = pd.Timestamp("2024-01-01", tz="America/Bogota")
        self.p2_start = pd.Timestamp("2024-04-01", tz="America/Bogota")
        self.p3_start = pd.Timestamp("2024-07-08", tz="America/Bogota")
        self.p4_start = pd.Timestamp("2024-10-15", tz="America/Bogota")

        self.vacaciones = [
            ("2024-03-25", "2024-03-29"),  # Semana Santa
            ("2024-06-17", "2024-07-05"),  # Mitad de año
            ("2024-10-07", "2024-10-11"),  # Octubre
        ]

        self.vacaciones = [(pd.Timestamp(ini, tz="America/Bogota"), pd.Timestamp(fin, tz="America/Bogota")) for ini, fin in self.vacaciones]

    def __del__(self):
        if hasattr(self, "con") and self.con:
            self.con.close()

    def close(self):
        if hasattr(self, "con") and self.con:
            self.con.close()
            self.con = None

    def _asignar_periodo(self, date_log):
        if date_log < self.p2_start and date_log >= self.p1_start:
            return "Periodo 1"
        elif date_log < self.p3_start:
            return "Periodo 2"
        elif date_log < self.p4_start:
            return "Periodo 3"
        else:
            return "Periodo 4"

    def _esta_en_vacaciones(self, fecha):
        return any(inicio <= fecha <= fin for inicio, fin in self.vacaciones)

    def _clasificar_jornada(self, hora):
        if 0 <= hora < 6:
            return "madrugada"
        elif 6 <= hora < 12:
            return "mañana"
        elif 12 <= hora < 18:
            return "tarde"
        else:
            return "noche"

    def _load_moodle_data(self, logs_parquet, students_enrollment, year=2024):
        try:
            sql = f"""
                SELECT 
                    logs.userid,
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
        combined_data["userid"] = combined_data["userid"].astype(str)

        # --- Paso 3: Asignar periodo
        combined_data["periodo"] = combined_data["fecha_local"].apply(self._asignar_periodo)

        # --- Paso 4: Marcar logins en vacaciones ---
        combined_data["en_vacaciones"] = combined_data["fecha_local"].apply(self._esta_en_vacaciones)

        # --- Paso 5: Día de la semana y jornada ---
        combined_data["hora"] = combined_data["fecha_local"].dt.hour
        combined_data["dia"] = combined_data["fecha_local"].dt.day_name().str.lower()
        combined_data["jornada"] = combined_data["hora"].apply(self._clasificar_jornada)

        # --- Paso 6: Calcular inactividad ---
        # Usar solo logins fuera de vacaciones para calcular `delta`
        resultado_no_vacaciones = combined_data[~combined_data["en_vacaciones"]].copy()

        # Ordenar y calcular diferencias por usuario + periodo + año
        resultado_no_vacaciones = resultado_no_vacaciones.sort_values(["userid", "year", "periodo", "fecha_local"])
        resultado_no_vacaciones["delta"] = resultado_no_vacaciones.groupby(["userid", "year", "periodo"])["fecha_local"].diff()

        # --- Paso 7: Resumen general por usuario + periodo + año ---
        # Total logins (sin filtrar vacaciones)
        summary = combined_data.groupby(["userid", "year", "periodo"]).agg(count_login=("fecha_local", "count")).reset_index()

        # Agregar `max_inactividad` desde el nuevo cálculo (excluyendo vacaciones)
        max_inactividad = resultado_no_vacaciones.groupby(["userid", "year", "periodo"])["delta"].max().reset_index(name="max_inactividad")
        summary = summary.merge(max_inactividad, on=["userid", "year", "periodo"], how="left")

        # --- Paso 8: Conteo por día de la semana ---
        conteo_dias = combined_data.pivot_table(
            index=["userid", "year", "periodo", "documento_identificación"], columns="dia", values="fecha_local", aggfunc="count", fill_value=0
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
            index=["userid", "year", "periodo"], columns="jornada", values="fecha_local", aggfunc="count", fill_value=0
        ).reset_index()

        # Renombrar jornadas
        jornadas = ["madrugada", "mañana", "tarde", "noche"]
        for j in jornadas:
            if j in conteo_jornada.columns:
                conteo_jornada.rename(columns={j: f"count_jornada_{j}"}, inplace=True)
            else:
                conteo_jornada[f"count_jornada_{j}"] = 0

        # --- Paso 9: Combinar todo ---
        df_final = summary.merge(conteo_dias, on=["userid", "year", "periodo"], how="left")
        df_final = df_final.merge(conteo_jornada, on=["userid", "year", "periodo"], how="left")

        # Asegurar orden de columnas (opcional)
        columnas_finales = (
            ["userid", "year", "periodo", "count_login", "max_inactividad", "documento_identificación"]
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
