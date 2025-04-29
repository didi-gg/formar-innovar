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

    def create_enrollments_df(self, parquet_users_path, parquet_user_info_path, students_df, year, edukrea_users_path=None):  # Added default None
        result_df = self._load_moodle_data(parquet_users_path, parquet_user_info_path)

        # Cargar y aplicar hashing
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
        enrollments = pd.merge(
            enrollments, merged_df[["documento_identificación", "grado", "sede"]], on="documento_identificación", how="left"
        )  # Add 'sede' here
        enrollments = pd.merge(enrollments, self.grados_df[["grado", "ID"]], left_on="grado", right_on="grado", how="left")
        enrollments.drop(columns=["grado"], inplace=True)
        enrollments.rename(columns={"ID": "id_grado"}, inplace=True)

        # Reordenar columnas si es necesario (opcional)
        cols_order = ["documento_identificación", "moodle_user_id", "edukrea_user_id", "year", "id_grado", "sede"]
        # Asegurarse que todas las columnas existan antes de reordenar
        cols_order = [col for col in cols_order if col in enrollments.columns]
        enrollments = enrollments[cols_order]

        return enrollments

    def process_all_years(self):
        # Procesar 2024
        students_2024 = pd.read_csv("data/interim/estudiantes/estudiantes_2024_hashed.csv")
        # Asegurarse que 'sede' existe en students_2024
        if "sede" not in students_2024.columns:
            print("Advertencia: La columna 'sede' no existe en 'estudiantes_2024_hashed.csv'. Se omitirá.")
            # Opcional: Añadir una columna 'sede' con valores por defecto si es necesario
            # students_2024['sede'] = 'Desconocida'

        enrollments_2024 = self.create_enrollments_df(
            parquet_users_path="data/raw/moodle/2024/Users/mdlvf_user.parquet",
            parquet_user_info_path="data/raw/moodle/2024/Users/mdlvf_user_info_data.parquet",
            students_df=students_2024,
            year=2024,
            # edukrea_users_path=None, # No se pasa para 2024
        )

        # Procesar 2025
        students_2025 = pd.read_csv("data/interim/estudiantes/estudiantes_imputed_encoded.csv")
        # Asegurarse que 'sede' existe en students_2025
        if "sede" not in students_2025.columns:
            raise ValueError("La columna 'sede' no existe en 'estudiantes_imputed_encoded.csv'.")

        enrollments_2025 = self.create_enrollments_df(
            parquet_users_path="data/raw/moodle/2025/Users/mdlvf_user.parquet",
            parquet_user_info_path="data/raw/moodle/2025/Users/mdlvf_user_info_data.parquet",
            students_df=students_2025,
            year=2025,
            edukrea_users_path="data/raw/moodle/Edukrea/Users/mdl_user.parquet",
        )

        # Combinar los dataframes
        all_enrollments = pd.concat([enrollments_2024, enrollments_2025], ignore_index=True)

        # Eliminar grados de preescolar
        all_enrollments = all_enrollments.merge(grados_df, left_on="id_grado", right_on="ID", how="left")
        all_enrollments = all_enrollments.drop(columns=["ID"])

        grados_to_remove = ["Prejardín", "Jardín", "Transición"]
        all_enrollments = all_enrollments[~all_enrollments["grado"].isin(grados_to_remove)]
        all_enrollments = all_enrollments.reset_index(drop=True)

        all_enrollments = all_enrollments[["documento_identificación", "moodle_user_id", "year", "edukrea_user_id", "id_grado", "sede"]]

        # Guardar el dataframe combinado
        output_path = "data/interim/estudiantes/enrollments.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        all_enrollments.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Archivo de matrículas combinado guardado en: {output_path}")

        return all_enrollments  # Devolver el dataframe combinado


if __name__ == "__main__":
    grados_df = pd.read_csv("data/raw/tablas_maestras/grados.csv")
    processor = EnrollmentProcessor(grados_df)
    final_enrollments = processor.process_all_years()
    print("Procesamiento de matrículas completado.")
