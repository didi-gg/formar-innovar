import duckdb
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.hash_utility import HashUtility


class EnrollmentProcessor:
    def __init__(self, grados_df):
        self.grados_df = grados_df
        self.con = duckdb.connect()

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

        if year == 2025:
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

    def create_enrollments_df(self, parquet_users_path, parquet_user_info_path, students_df, year, edukrea_users_path):
        result_df = self._load_moodle_data(parquet_users_path, parquet_user_info_path)

        # Cargar y aplicar hashing
        result_df["documento_identificación"] = result_df["documento_identificación"].apply(HashUtility.hash_stable)

        # Combinar con estudiantes
        merged_df = pd.merge(result_df, students_df, on="documento_identificación", how="inner")

        # Inicializar DataFrame de matrículas
        enrollments = pd.DataFrame(
            {"documento_identificación": merged_df["documento_identificación"], "moodle_user_id": merged_df["UserID"], "year": year}
        )

        # Agregar mapeo de edukrea si aplica
        edukrea_mapping = self._load_edukrea_mapping(year, edukrea_users_path)
        enrollments["edukrea_user_id"] = enrollments["documento_identificación"].map(edukrea_mapping) if edukrea_mapping else ""

        # Agregar grado y hacer join con ID de grado
        enrollments = pd.merge(enrollments, merged_df[["documento_identificación", "grado"]], on="documento_identificación", how="left")
        enrollments = pd.merge(enrollments, self.grados_df[["grado", "ID"]], left_on="grado", right_on="grado", how="left")
        enrollments.drop(columns=["grado"], inplace=True)
        enrollments.rename(columns={"ID": "id_grado"}, inplace=True)

        return enrollments

    def process_all_years(self):
        # Procesar 2024
        students_2024 = pd.read_csv("data/processed/Estudiantes_2024_hashed.csv")
        enrollments_2024 = self.create_enrollments_df(
            parquet_users_path="data/processed/parquets/Users/mdlvf_user.parquet",
            parquet_user_info_path="data/processed/parquets/Users/mdlvf_user_info_data.parquet",
            students_df=students_2024,
            year=2024,
            edukrea_users_path=None,
        )

        # Procesar 2025
        students_2025 = pd.read_csv("data/processed/Estudiantes_imputed_encoded.csv")
        enrollments_2025 = self.create_enrollments_df(
            parquet_users_path="data/processed/parquets_2025/Users/mdlvf_user.parquet",
            parquet_user_info_path="data/processed/parquets_2025/Users/mdlvf_user_info_data.parquet",
            students_df=students_2025,
            year=2025,
            edukrea_users_path="data/processed/Edukrea/Users/mdl_user.parquet",
        )
        return enrollments_2024, enrollments_2025


if __name__ == "__main__":
    grados_df = pd.read_csv("data/processed/Grados.csv")
    processor = EnrollmentProcessor(grados_df)
    processor.process_all_years()
