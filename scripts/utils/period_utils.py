import pandas as pd
import logging


class PeriodUtils:
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
                "p2_start": pd.Timestamp("2025-04-07", tz="America/Bogota"),
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

    def assign_period(self, date_log):
        year = date_log.year
        if year not in self.periods:
            self.logger.warning(f"Año {year} no configurado, usando fechas de 2024 por defecto")
            year = 2024

        if date_log < self.periods[year]["p2_start"] and date_log >= self.periods[year]["p1_start"]:
            return "Periodo 1"
        elif date_log < self.periods[year]["p3_start"]:
            return "Periodo 2"
        elif date_log < self.periods[year]["p4_start"]:
            return "Periodo 3"
        else:
            return "Periodo 4"

    def is_vacation(self, date):
        year = date.year
        if year not in self.periods:
            self.logger.warning(f"Año {year} no configurado, usando fechas de 2024 por defecto")
            year = 2024

        return any(start <= date <= end for start, end in self.periods[year]["vacations"])

    def classify_daytime(self, hour):
        if 0 <= hour < 6:
            return "madrugada"
        elif 6 <= hour < 12:
            return "mañana"
        elif 12 <= hour < 18:
            return "tarde"
        else:
            return "noche"

    def calculate_period_duration(self, year, period):
        if year not in self.periods:
            self.logger.warning(f"Año {year} no configurado, usando fechas de 2024 por defecto")
            year = 2024

        if period == "Periodo 1":
            start = self.periods[year]["p1_start"]
            end = self.periods[year]["p2_start"] - pd.Timedelta(days=1)
        elif period == "Periodo 2":
            start = self.periods[year]["p2_start"]
            end = self.periods[year]["p3_start"] - pd.Timedelta(days=1)
        elif period == "Periodo 3":
            start = self.periods[year]["p3_start"]
            end = self.periods[year]["p4_start"] - pd.Timedelta(days=1)
        elif period == "Periodo 4":
            start = self.periods[year]["p4_start"]
            end = pd.Timestamp(f"{year}-12-31", tz="America/Bogota")
        else:
            raise ValueError(f"Periodo desconocido: {period}")

        date_range = pd.date_range(start=start, end=end, freq="D")
        dates_no_vacation = [date for date in date_range if not self.is_vacation(date)]

        return pd.Timedelta(days=len(dates_no_vacation))
