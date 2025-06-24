import pandas as pd
import re
import logging


class AcademicPeriodUtils:
    """
    Utilidades para manejo de periodos académicos y clasificación de fechas en Moodle.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.build_periods()

    def build_periods(self):
        # Leer CSVs
        calendario_df = pd.read_csv("data/raw/tablas_maestras/calendario_escolar.csv", dayfirst=True)
        vacaciones_df = pd.read_csv("data/raw/tablas_maestras/vacaciones_festivos.csv", dayfirst=True)

        # Limpiar columnas
        calendario_df.columns = calendario_df.columns.str.strip().str.lower()
        vacaciones_df.columns = vacaciones_df.columns.str.strip().str.lower()

        # Convertir fechas a datetime
        calendario_df["inicio"] = pd.to_datetime(calendario_df["inicio"], dayfirst=True)
        vacaciones_df["inicio"] = pd.to_datetime(vacaciones_df["inicio"], dayfirst=True)
        vacaciones_df["fin"] = pd.to_datetime(vacaciones_df["fin"], dayfirst=True)

        # Construir periods
        periods = {}

        for year in calendario_df["año"].unique():
            year_data = calendario_df[(calendario_df["año"] == year) & (calendario_df["semana"] == 1)]
            year_periods = {}

            # Fechas de inicio de bimestres
            for _, row in year_data.iterrows():
                bimestre = int(row["bimestre"])
                start_date = pd.Timestamp(row["inicio"], tz="America/Bogota")
                year_periods[f"p{bimestre}_start"] = start_date

            # Vacaciones como Timestamps con hora incluida
            year_vacaciones = vacaciones_df[vacaciones_df["año"] == year]
            vacations_list = [
                (
                    pd.Timestamp(row["inicio"].strftime("%Y-%m-%dT00:00:00"), tz="America/Bogota"),
                    pd.Timestamp(row["fin"].strftime("%Y-%m-%dT23:59:00"), tz="America/Bogota"),
                )
                for _, row in year_vacaciones.iterrows()
            ]
            year_periods["vacations"] = vacations_list

            periods[year] = year_periods
        self.periods = periods

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

    @staticmethod
    def extract_week_number_from_title(title: str) -> str:
        """
        Extrae el número de semana desde el título de un recurso.

        Casos válidos:
        - '1. Algo'
        - '1.Algo'
        - '10. Título'
        - '10.Título'
        - '10.'

        Args:
            title (str): Título del recurso.

        Returns:
            str: Número de semana o 0 si no se puede extraer.
        """
        if not isinstance(title, str):
            return 0

        match = re.match(r"^\s*(\d{1,2})\.\s*", title.strip())
        if not match:
            return 0
        # If is nan
        if pd.isna(match.group(1)):
            return 0
        return match.group(1)

    @staticmethod
    def extract_week_number_string(section):
        """
        Extrae el número de semana de un nombre de sección de Moodle.
        Retorna 0 si no se encuentra un número válido o si la sección es irrelevante.
        """
        if not isinstance(section, str):
            return 0

        if section.strip() in [
            "",
            "-",
            "PLANTILLA",
            "General",
            "Syllabus",
            "syllabus",
            "Lineamientos  generales.",
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
            "Lineamientos generales. ",
            "Evaluación de conocimientos ",
        ]:
            return 0

        match = re.search(r"(\d+)", section)
        if not match:
            return 0
        return match.group(1)

    @staticmethod
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
            return 0

        if 1 <= w <= 8:
            return 1
        elif 9 <= w <= 16:
            return 2
        elif 17 <= w <= 24:
            return 3
        elif 25 <= w <= 32:
            return 4
        else:
            return 0

    @staticmethod
    def _load_calendar() -> pd.DataFrame:
        calendar_file = "data/raw/tablas_maestras/calendario_escolar.csv"
        df = pd.read_csv(calendar_file)
        df["inicio"] = pd.to_datetime(df["inicio"], dayfirst=True, errors="coerce")
        df.rename(columns={"año": "year", "bimestre": "period", "semana_general": "week"}, inplace=True)
        df["year"] = df["year"].astype(int)
        df["period"] = df["period"].astype(int)
        df["week"] = df["week"].astype(int)
        return df
