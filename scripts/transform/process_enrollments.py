import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.moodle_path_resolver import MoodlePathResolver
from utils.base_script import BaseScript
from utils.hash_utility import HashUtility


class EnrollmentProcessor(BaseScript):
    def _load_moodle_data(self, parquet_users_path, parquet_user_info_path):
        sql = f"""
        SELECT 
            u.id AS UserID,
            u.idnumber AS documento_identificación,
            CONCAT(u.firstname, ' ', u.lastname) AS "Nombre Completo",
            u.city AS Sede,
            to_timestamp(u.firstaccess) AS "Fecha Primer Acceso",
            to_timestamp(u.lastaccess) AS "Feha Último Acceso",
            to_timestamp(u.lastlogin) AS "Fecha Último Inicio de Sesión",
            to_timestamp(u.timecreated) AS "Fecha Creación"
        FROM 
            '{parquet_users_path}' u
        JOIN 
            '{parquet_user_info_path}' uid 
            ON u.id = uid.userid
        WHERE 
            uid.data = 'Estudiante'
            AND u.idnumber <> ''
            AND u.deleted = 0;
        """
        return self.con.execute(sql).df()

    def _load_edukrea_mapping(self, year, edukrea_users_path):
        edukrea_mapping = {}

        if year == 2025 and edukrea_users_path:  # Added check for path existence
            sql = f"""
            SELECT 
                u.id AS UserID,
                u.idnumber AS documento_identificación
            FROM
                '{edukrea_users_path}' u
            WHERE 
                u.idnumber <> ''
                AND u.deleted = 0;
            """
            edukrea_df = self.con.execute(sql).df()
            edukrea_df["documento_identificación"] = edukrea_df["documento_identificación"].apply(HashUtility.hash_stable)
            edukrea_mapping = dict(zip(edukrea_df["documento_identificación"], edukrea_df["UserID"]))

        return edukrea_mapping

    def create_enrollments_df(self, parquet_users_path, parquet_user_info_path, students_df, year, edukrea_users_path=None):
        result_df = self._load_moodle_data(parquet_users_path, parquet_user_info_path)

        # Cargar y aplicar hashing
        result_df["documento_identificación"] = result_df["documento_identificación"].astype(str).str.replace(r"\s+", "", regex=True)
        result_df["documento_identificación"] = result_df["documento_identificación"].apply(HashUtility.hash_stable)

        # Combinar con estudiantes - Asegúrate que 'sede' esté en students_df
        if "sede" not in students_df.columns:
            raise ValueError("La columna 'sede' no se encuentra en students_df")
        merged_df = pd.merge(
            result_df, students_df[["documento_identificación", "grado", "sede"]], on="documento_identificación", how="inner"
        )  # Select 'sede' here

        # Inicializar DataFrame de matrículas
        enrollments = pd.DataFrame(
            {"documento_identificación": merged_df["documento_identificación"], "moodle_user_id": merged_df["UserID"], "year": year}
        )

        # Agregar mapeo de edukrea si aplica
        edukrea_mapping = self._load_edukrea_mapping(year, edukrea_users_path)
        enrollments["edukrea_user_id"] = (
            enrollments["documento_identificación"].map(edukrea_mapping) if edukrea_mapping else None
        )  # Use None instead of ""

        # Agregar grado y sede, y hacer join con ID de grado
        # We already have 'grado' and 'sede' in merged_df from the first merge
        enrollments = pd.merge(enrollments, merged_df[["documento_identificación", "grado", "sede"]], on="documento_identificación", how="left")

        # Reordenar columnas si es necesario (opcional)
        cols_order = ["documento_identificación", "moodle_user_id", "edukrea_user_id", "year", "grado", "sede"]
        # Asegurarse que todas las columnas existan antes de reordenar
        cols_order = [col for col in cols_order if col in enrollments.columns]
        enrollments = enrollments[cols_order]

        return enrollments

    def process_all_years(self):
        # Procesar 2024
        students_2024 = pd.read_csv("data/interim/estudiantes/estudiantes_2024_hashed.csv")
        parquet_users_path, parquet_user_info_path = MoodlePathResolver.get_paths(2024, "user", "user_info_data")

        enrollments_2024 = self.create_enrollments_df(
            parquet_users_path=parquet_users_path,
            parquet_user_info_path=parquet_user_info_path,
            students_df=students_2024,
            year=2024,
            # edukrea_users_path=None, # No se pasa para 2024
        )

        # Procesar 2025
        students_2025 = pd.read_csv("data/interim/estudiantes/estudiantes_2025_hashed.csv")
        parquet_users_path, parquet_user_info_path = MoodlePathResolver.get_paths(2025, "user", "user_info_data")
        parquet_users_path_edukrea = MoodlePathResolver.get_paths("Edukrea", "user")[0]

        enrollments_2025 = self.create_enrollments_df(
            parquet_users_path=parquet_users_path,
            parquet_user_info_path=parquet_user_info_path,
            students_df=students_2025,
            year=2025,
            edukrea_users_path=parquet_users_path_edukrea,
        )

        # Combinar los dataframes
        all_enrollments = pd.concat([enrollments_2024, enrollments_2025], ignore_index=True)

        grados_to_remove = ["Prejardín", "Jardín", "Transición"]
        all_enrollments = all_enrollments[~all_enrollments["grado"].isin(grados_to_remove)]
        all_enrollments = all_enrollments.reset_index(drop=True)

        # Rename grado to id_grado
        all_enrollments.rename(columns={"grado": "id_grado"}, inplace=True)

        all_enrollments = all_enrollments[["documento_identificación", "moodle_user_id", "year", "edukrea_user_id", "id_grado", "sede"]]

        # Guardar el dataframe combinado
        output_path = "data/interim/estudiantes/enrollments.csv"
        self.save_to_csv(all_enrollments, output_path)


if __name__ == "__main__":
    processor = EnrollmentProcessor()
    final_enrollments = processor.process_all_years()
    processor.logger.info("Enrollment data processed successfully.")
    processor.close()
