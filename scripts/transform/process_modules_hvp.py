import pandas as pd
import os
import sys
import re
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.moodle_path_resolver import MoodlePathResolver
from utils.base_script import BaseScript
from utils.academic_period_utils import AcademicPeriodUtils

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class MoodleHVPProcessor(BaseScript):
    def parse_json(self, row):
        try:
            return json.loads(row)
        except:
            return {}

    def _get_hvp(self, hvp, hvp_libraries, moodle_modules, year, platform):
        sql_hvp = f"""
            SELECT
                year,
                course_id,
                course_module_id,
                asignatura_name,
                sede,
                week,
                period,
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
            WHERE cm.year = '{year}'
            AND cm.platform = '{platform}'
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

        df.drop(columns=["json_parsed"], inplace=True)
        df["json_kb"] = df["hvp_type"].apply(lambda x: round(len(x.encode("utf-8")) / 1024, 2))

        # Contar cuántas librerías hay en cada fila
        df["libraries_count"] = df["h5p_libraries"].apply(lambda libs: len(libs) if isinstance(libs, list) else 0)

        # Verificar si "reto inglés" aparece en el nombre, sin importar mayúsculas
        df["is_in_english"] = df["hvp_name"].str.lower().str.contains(r"reto(?:\s+\w+){0,2}\s+ingl[eé]s", flags=re.IGNORECASE)

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

        # 4. Resultado: solo es interactivo si tiene alguna librería válida
        df["is_interactive"] = ~(cond_no_interactive_libraries | cond_machine)

        return df

    def process_all_hvp(self):
        # Load HVP data
        tables = ["hvp", "hvp_libraries"]

        year = 2024
        modules = "data/interim/moodle/modules_active.csv"
        hvp, hvp_libraries = MoodlePathResolver.get_paths(year, *tables)
        hvp_data_2024 = self._get_hvp(hvp, hvp_libraries, modules, year, platform="moodle")
        hvp_data_2024["platform"] = "moodle"

        year = 2025
        hvp, hvp_libraries = MoodlePathResolver.get_paths(year, *tables)
        hvp_data_2025 = self._get_hvp(hvp, hvp_libraries, modules, year, platform="moodle")
        hvp_data_2025["platform"] = "moodle"

        year = 2025
        hvp, hvp_libraries = MoodlePathResolver.get_paths("Edukrea", *tables)
        hvp_data_edukrea = self._get_hvp(hvp, hvp_libraries, modules, year, platform="edukrea")
        hvp_data_edukrea["platform"] = "edukrea"

        # Combine HVP data for all platforms and years
        hvp_data_moodle = pd.concat([hvp_data_2024, hvp_data_2025], ignore_index=True)
        hvp_data_moodle = self.process_hvp(hvp_data_moodle)

        hvp_data_edukrea = self.process_hvp(hvp_data_edukrea)

        # Combine all data into a single DataFrame
        hvp_data_combined = pd.concat([hvp_data_moodle, hvp_data_edukrea], ignore_index=True)

        # Save the processed HVP data
        self.save_to_csv(hvp_data_combined, "data/interim/moodle/hvp.csv")
        self.logger.info("HVP data processed and saved successfully.")


if __name__ == "__main__":
    processor = MoodleHVPProcessor()
    processor.process_all_hvp()
    processor.close()
    processor.logger.info("Moodle HVP processing completed.")
