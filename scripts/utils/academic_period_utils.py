import pandas as pd
import re
import logging


class AcademicPeriodUtils:
    """
    Utilidades para manejo de periodos académicos y clasificación de fechas en Moodle.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.periods = {
            2024: {
                "p1_start": pd.Timestamp("2024-02-01", tz="America/Bogota"),
                "p2_start": pd.Timestamp("2024-04-01", tz="America/Bogota"),
                "p3_start": pd.Timestamp("2024-07-08", tz="America/Bogota"),
                "p4_start": pd.Timestamp("2024-10-15", tz="America/Bogota"),
                "vacations": [
                    ("2024-03-25", "2024-03-29"),
                    ("2024-06-17", "2024-07-05"),
                    ("2024-10-07", "2024-10-11"),
                ],
            },
            2025: {
                "p1_start": pd.Timestamp("2025-01-27", tz="America/Bogota"),
                "p2_start": pd.Timestamp("2025-04-15", tz="America/Bogota"),
                "p3_start": pd.Timestamp("2025-07-14", tz="America/Bogota"),
                "p4_start": pd.Timestamp("2025-10-14", tz="America/Bogota"),
                "vacations": [
                    ("2025-03-17", "2025-03-21"),
                    ("2025-06-16", "2025-07-11"),
                    ("2025-10-06", "2025-10-10"),
                ],
            },
        }

        # Convertir fechas de vacaciones a timestamps
        for year in self.periods:
            self.periods[year]["vacations"] = [
                (pd.Timestamp(start, tz="America/Bogota"), pd.Timestamp(end, tz="America/Bogota")) for start, end in self.periods[year]["vacations"]
            ]

    def get_period_start_date(self, row):
        """
        Retorna la fecha de inicio del periodo para una fila con campos 'year' y 'period'.
        """
        try:
            return self.periods[row["year"]][f"p{int(row['period'])}_start"]
        except Exception:
            return pd.NaT

    def determine_period_from_date(self, date_log):
        """
        Determina el número de periodo (1-4) según una fecha específica.
        """
        year = date_log.year
        if year not in self.periods:
            self.logger.warning(f"Año {year} no configurado, usando fechas de 2024 por defecto")
            year = 2024

        if date_log < self.periods[year]["p2_start"] and date_log >= self.periods[year]["p1_start"]:
            return "1"
        elif date_log < self.periods[year]["p3_start"]:
            return "2"
        elif date_log < self.periods[year]["p4_start"]:
            return "3"
        else:
            return "4"

    def is_vacation(self, date):
        """
        Verifica si una fecha dada cae dentro de un periodo de vacaciones.
        """
        year = date.year
        if year not in self.periods:
            self.logger.warning(f"Año {year} no configurado, usando fechas de 2024 por defecto")
            year = 2024

        return any(start <= date <= end for start, end in self.periods[year]["vacations"])

    def classify_daytime(self, hour):
        """
        Clasifica una hora del día como madrugada, mañana, tarde o noche.
        """
        if 0 <= hour < 6:
            return "madrugada"
        elif 6 <= hour < 12:
            return "mañana"
        elif 12 <= hour < 18:
            return "tarde"
        else:
            return "noche"

    def calculate_period_duration(self, year, period):
        """
        Calcula la duración de un periodo (1–4) excluyendo días de vacaciones.
        """
        if year not in self.periods:
            self.logger.warning(f"Año {year} no configurado, usando fechas de 2024 por defecto")
            year = 2024

        if period == "1":
            start = self.periods[year]["p1_start"]
            end = self.periods[year]["p2_start"] - pd.Timedelta(days=1)
        elif period == "2":
            start = self.periods[year]["p2_start"]
            end = self.periods[year]["p3_start"] - pd.Timedelta(days=1)
        elif period == "3":
            start = self.periods[year]["p3_start"]
            end = self.periods[year]["p4_start"] - pd.Timedelta(days=1)
        elif period == "4":
            start = self.periods[year]["p4_start"]
            end = pd.Timestamp(f"{year}-12-31", tz="America/Bogota")
        else:
            raise ValueError(f"Periodo desconocido: {period}")

        date_range = pd.date_range(start=start, end=end, freq="D")
        dates_no_vacation = [date for date in date_range if not self.is_vacation(date)]

        return pd.Timedelta(days=len(dates_no_vacation))

    def extract_week_number_string(section):
        """
        Extrae el número de semana de un nombre de sección de Moodle.
        Retorna 'na' si no se encuentra un número válido o si la sección es irrelevante.
        """
        if not isinstance(section, str) or section.strip() in [
            "",
            "-",
            "PLANTILLA",
            "General",
            "Syllabus",
            "syllabus",
            "Lineamientos Generales",
            "Lineamientos generales",
            "Lineamiemtos Generales",
            "LINEAMIENTOS GENERALES",
            "Lineamitos Generales",
            "Lineamientos generales.",
            "Lineamientos Generales.",
            "General guidelines",
            "General Guidelines",
            "Here starts the aleatorio-estadistico thinking",
            "Recursos",
            "Diagnóstico",
            "Induction",
            "BIENVENIDA",
            "Bienvenida",
            "Evaluation week",
            "Evaluación Primer Bimestre",
            "Evaluación Semestral II",
            "Syllabus Innovación y emprendimiento",
            "Seccion 32",
            "Lineamientos generales. ",
        ]:
            return "na"
        match = re.search(r"(\d+)", section)
        if match:
            return match.group(1)
        return "na"

    def get_period_from_week(week):
        """
        Asocia un número de semana con un periodo académico.
        - Semanas 1–8   => Periodo 1
        - Semanas 9–16  => Periodo 2
        - Semanas 17–24 => Periodo 3
        - Semanas 25–32 => Periodo 4
        """
        try:
            w = int(week)
        except (ValueError, TypeError):
            return "na"

        if 1 <= w <= 8:
            return 1
        elif 9 <= w <= 16:
            return 2
        elif 17 <= w <= 24:
            return 3
        elif 25 <= w <= 32:
            return 4
        else:
            return "na"
