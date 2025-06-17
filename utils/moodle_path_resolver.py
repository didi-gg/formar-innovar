from pathlib import Path
from utils.moodle_tables_schema import MOODLE_TABLES_GROUPED, EDUKREA_TABLES_GROUPED


class MoodlePathResolver:
    @staticmethod
    def get_paths(folder, *tables):
        base_path = Path("data/raw/moodle")

        folder = str(folder)

        if folder in ("2024", "2025"):
            schema = MOODLE_TABLES_GROUPED
            prefix = "mdlvf"
        elif folder == "Edukrea":
            schema = EDUKREA_TABLES_GROUPED
            prefix = "mdl"
        else:
            raise ValueError(f"Folder '{folder}' no es válido. Usa '2024', '2025' o 'Edukrea'.")

        table_to_group = {table: group for group, tables_list in schema.items() for table in tables_list}

        output_paths = []

        for short_name in tables:
            full_table_name = f"{prefix}_{short_name}"

            if full_table_name not in table_to_group:
                raise ValueError(f"La tabla '{full_table_name}' no está en el esquema para '{folder}'.")

            group_folder = table_to_group[full_table_name]
            path = base_path / folder / group_folder / f"{full_table_name}.parquet"

            if not path.is_file():
                raise FileNotFoundError(f"Archivo no encontrado: {path}")

            output_paths.append(str(path))

        return tuple(output_paths)
