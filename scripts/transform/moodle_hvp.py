import duckdb
import pandas as pd
import os
import sys
import logging
import re
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class MoodleHVP:
    def __init__(self):
        self.con = duckdb.connect()
        self.logger = logging.getLogger(__name__)

    def __del__(self):
        if hasattr(self, "con") and self.con:
            self.con.close()

    def close(self):
        if hasattr(self, "con") and self.con:
            self.con.close()
            self.con = None

    def parse_json(self, row):
        try:
            return json.loads(row)
        except:
            return {}

    def _get_hvp(self, hvp, hvp_libraries, moodle_modules, year):
        sql_hvp = f"""
            SELECT
                year,
                course_id,
                course_module_id,
                asignatura_name,
                sede,
                week,
                period,
                fecha_inicio_semana,
                total_actualizaciones_docente,
                dias_desde_creacion,
                total_vistas_docente,
                fecha_primera_vista,
                fecha_ultima_vista,
                accedio_antes,
                interactivo,
                total_estudiantes,
                total_vistas_estudiantes,
                estudiantes_que_vieron,
                total_interacciones_estudiantes,
                estudiantes_que_interactuaron,
                min_vistas_estudiante,
                max_vistas_estudiante,
                mediana_vistas_estudiante,
                id_grado,
                hl.machine_name,
                h.timecreated,
                h.timemodified,
                h.completionpass,
                h.course as course_hvp,
                cm.instance,
                section_name,
                cm.section_id,
                h.id AS hvp_id,
                h.name AS hvp_name,
                h.json_content  AS hvp_type,
            FROM '{moodle_modules}' cm
            JOIN '{hvp}' AS h ON cm.instance = h.id AND cm.module_type_id = 24
            LEFT JOIN '{hvp_libraries}' hl ON hl.id = h.main_library_id
            WHERE cm.year = year
            """
        try:
            return self.con.execute(sql_hvp).df()
        except Exception as e:
            self.logger.error(f"Error al cargar los HVP: {str(e)}")
            raise

    def process_hvp(self, df):
        df["json_parsed"] = df["hvp_type"].apply(self.parse_json)
        # 1. Obtener llaves de primer nivel
        df["json_keys"] = df["json_parsed"].apply(lambda x: list(x.keys()))

        # 2. Extraer todos los "H5P.*" encontrados en cualquier nivel
        def extract_libraries(obj):
            libraries = set()
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "library" and isinstance(v, str) and v.startswith("H5P."):
                        libraries.add(v)
                    else:
                        libraries.update(extract_libraries(v))
            elif isinstance(obj, list):
                for item in obj:
                    libraries.update(extract_libraries(item))
            return list(libraries)

        df["h5p_libraries"] = df["json_parsed"].apply(extract_libraries)

        # Elimina json_parsed si no lo necesitas
        df.drop(columns=["json_parsed"], inplace=True)
        df["json_kb"] = df["hvp_type"].apply(lambda x: round(len(x.encode("utf-8")) / 1024, 2))

        # Contar cuántas librerías hay en cada fila
        df["libraries_count"] = df["h5p_libraries"].apply(lambda libs: len(libs) if isinstance(libs, list) else 0)

        # Verificar si "reto inglés" aparece en el nombre, sin importar mayúsculas
        df["english"] = df["hvp_name"].str.lower().str.contains(r"reto(?:\s+\w+){0,2}\s+ingl[eé]s", flags=re.IGNORECASE)

        # 1. Set de librerías interactivas
        interactive_libraries = {
            "H5P.Blanks 1.12",
            "H5P.Blanks 1.14",
            "H5P.DragQuestionDropzone 0.1",
            "H5P.DragText 1.8",
            "H5P.MarkTheWords 1.9",
            "H5P.MultiChoice 1.14",
            "H5P.MultiChoice 1.16",
            "H5P.QuestionSet 1.17",
            "H5P.SingleChoiceSet 1.11",
            "H5P.TrueFalse 1.6",
            "H5P.TrueFalse 1.8",
        }

        # 2. Machine types explícitamente no interactivos
        non_interactive_machines = ["H5P.Audio", "H5P.DocumentationTool"]
        cond_machine = df["machine_name"].isin(non_interactive_machines)

        # 3. Condición por librerías: NO contiene ninguna interactiva → True (es no interactivo)
        cond_no_interactive_libraries = df["h5p_libraries"].apply(
            lambda libs: not any(lib in interactive_libraries for lib in libs) if isinstance(libs, list) else True
        )

        # 4. Resultado: solo es interactivo si tiene alguna librería válida Y no es machine bloqueado
        df["interactive"] = ~(cond_no_interactive_libraries | cond_machine)

        return df

    def process_all_hvp(self):
        # Load HVP data

        year = 2024
        hvp = f"data/raw/moodle/{year}/h5/mdlvf_hvp.parquet"
        hvp_libraries = f"data/raw/moodle/{year}/h5/mdlvf_hvp_libraries.parquet"
        moodle_modules = "data/interim/moodle/modules_active_moodle.csv"
        hvp_data_2024 = self._get_hvp(hvp, hvp_libraries, moodle_modules, year)

        year = 2025
        hvp = f"data/raw/moodle/{year}/h5/mdlvf_hvp.parquet"
        hvp_libraries = f"data/raw/moodle/{year}/h5/mdlvf_hvp_libraries.parquet"
        moodle_modules = "data/interim/moodle/modules_active_moodle.csv"
        hvp_data_2025 = self._get_hvp(hvp, hvp_libraries, moodle_modules, year)

        year = 2025
        hvp = "data/raw/moodle/Edukrea/Interactive Content/mdl_hvp.parquet"
        hvp_libraries = "data/raw/moodle/Edukrea/Interactive Content/mdl_hvp_libraries.parquet"
        moodle_modules = "data/interim/moodle/modules_active_edukrea.csv"
        hvp_data_edukrea = self._get_hvp(hvp, hvp_libraries, moodle_modules, year)

        # Combine HVP data for both years
        hvp_data = pd.concat([hvp_data_2024, hvp_data_2025], ignore_index=True)
        hvp_data = self.process_hvp(hvp_data)

        hvp_data_edukrea = self.process_hvp(hvp_data_edukrea)

        # Save the processed HVP data
        hvp_data.to_csv("data/interim/moodle/hvp_data_moodle.csv", index=False, encoding="utf-8-sig", quoting=1)
        self.logger.info("HVP data processed and saved successfully.")

        hvp_data_edukrea.to_csv("data/interim/moodle/hvp_data_edukrea.csv", index=False, encoding="utf-8-sig", quoting=1)
        self.logger.info("HVP data processed and saved successfully.")


if __name__ == "__main__":
    moodle_hvp = MoodleHVP()
    moodle_hvp.process_all_hvp()
    moodle_hvp.close()
    logging.info("Moodle HVP processing completed.")
