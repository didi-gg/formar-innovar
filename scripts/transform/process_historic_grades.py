"""
Script para procesar calificaciones históricas de los años 2021-2025.

Este script consolida las calificaciones de diferentes períodos y formatos:
- Años 2021-2022: Datos en formato Excel con estructura wide
- Años 2023-2025: Datos en formato CSV con estructura long

El proceso incluye:
1. Anonimización de datos sensibles
2. Transformación de formato wide a long (2021-2022)
3. Estandarización de columnas
4. Consolidación de datasets
5. Clasificación de niveles de aprendizaje
"""

import os
import sys
import pandas as pd
from typing import Dict, List, Optional

# Agregar el directorio raíz del proyecto al path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.base_script import BaseScript
from utils.hash_utility import HashUtility


class ProcessHistoricGrades(BaseScript):
    """Procesador de calificaciones históricas 2021-2025."""

    def __init__(self):
        super().__init__()
        self.data_dir = os.path.join(project_root, 'data')
        self.raw_dir = os.path.join(self.data_dir, 'raw')
        self.interim_dir = os.path.join(self.data_dir, 'interim')
        self.processed_dir = os.path.join(self.data_dir, 'processed')

        # Asegurar que los directorios existen
        os.makedirs(os.path.join(self.interim_dir, 'calificaciones'), exist_ok=True)

    def process_grades_2021_2022(self) -> pd.DataFrame:
        """
        Procesa calificaciones de los años 2021-2022 desde archivo Excel.

        Returns:
            DataFrame con calificaciones en formato short
        """
        self.logger.info("Procesando calificaciones 2021-2022...")

        # Cargar datos
        excel_path = os.path.join(self.raw_dir, 'calificaciones', 'excel', 'calificaciones_2021_2022.xlsx')
        df_asignaturas = pd.read_excel(excel_path, sheet_name="Asignaturas")
        df_bd = pd.read_excel(excel_path, sheet_name="BD")
        carga_horaria_df = pd.read_csv(os.path.join(self.raw_dir, 'tablas_maestras', 'carga_horaria.csv'))

        # Limpiar identificación
        df_bd["Identificación"] = df_bd["Identificación"].astype(str).str.strip()

        # Anonimización
        # Verificar si existe la variable de entorno, si no usar una clave por defecto
        if not os.getenv("SECRET_KEY_DOC_ID"):
            os.environ["SECRET_KEY_DOC_ID"] = "default_secret_key_for_anonymization_2024"
            self.logger.warning("Variable de entorno SECRET_KEY_DOC_ID no encontrada. Usando clave por defecto.")

        df_bd["Identificación"] = df_bd["Identificación"].apply(HashUtility.hash_stable)
        df_bd["Estudiante"] = df_bd["Estudiante"].apply(HashUtility.hash_stable)

        # Crear mapeo de asignaturas
        abv_to_id = dict(zip(df_asignaturas["Abv"], df_asignaturas["Id"]))

        # Verificar que el mapeo no esté vacío
        if not abv_to_id:
            raise ValueError("No se pudo crear el mapeo de asignaturas desde el archivo Excel.")

        # Transformar de wide a long
        competencias = ["Cog", "Proc", "Axi", "Act", "Com", "Prom"]
        datos_generales = ["Sede", "Año", "Periodo", "Grado", "Identificación", "Estudiante"]

        # Encontrar todas las abreviaturas usadas en las columnas
        asignaturas_abv = sorted(set(
            col.split('_')[-1] for col in df_bd.columns 
            if any(col.startswith(c + "_") for c in competencias)
        ))

        registros = []

        for _, fila in df_bd.iterrows():
            datos_base = {col: fila[col] for col in datos_generales}
            for abv in asignaturas_abv:
                asignatura_nombre = abv_to_id.get(abv, abv)

                # Verificar que la asignatura se mapeó correctamente
                if asignatura_nombre == abv and abv not in abv_to_id:
                    self.logger.error(f"❌ Asignatura '{abv}' no encontrada en el mapeo de asignaturas.")
                    raise ValueError(f"Asignatura '{abv}' no encontrada en el mapeo de asignaturas del Excel.")

                registro = {
                    **datos_base,
                    "Asignatura": asignatura_nombre
                }
                for comp in competencias:
                    col_name = f"{comp}_{abv}"
                    registro[comp] = fila.get(col_name)
                registros.append(registro)

        # Convertir a DataFrame
        df_final = pd.DataFrame(registros)

        # Renombrar columnas
        df_final.rename(columns={
            "Cog": "cognitivo",
            "Proc": "procedimental",
            "Axi": "axiológico",
            "Act": "actitudinal",
            "Com": "comunicativo",
            "Prom": "resultado",
            "Sede": "sede",
            "Año": "año",
            "Periodo": "periodo",
            "Grado": "grado",
            "Identificación": "identificación",
            "Estudiante": "estudiante",
            "Asignatura": "asignatura"
        }, inplace=True)

        # Agregar docente e intensidad horaria
        carga_horaria_df_renamed = carga_horaria_df.rename(columns={
            "year": "año",
            "id_grado": "grado",
            "asignatura": "nombre_asignatura",
            "id_asignatura": "asignatura",
            "intensidad": "intensidad_horaria"
        })

        df_final = df_final.merge(
            carga_horaria_df_renamed[["año", "sede", "grado", "asignatura", "docente", "intensidad_horaria"]],
            on=["año", "sede", "grado", "asignatura"],
            how="left"
        )

        # Remover filas con valores de evaluación vacíos
        columnas_a_verificar = ["cognitivo", "procedimental", "axiológico", "actitudinal", "comunicativo"]
        df_final = df_final[~df_final[columnas_a_verificar].isna().all(axis=1)]
        df_final = df_final.reset_index(drop=True)

        # Convertir a formato short (mantener solo las columnas necesarias)
        df_short = df_final[['sede', 'año', 'periodo', 'grado', 'identificación', 'estudiante', 
                            'asignatura', 'cognitivo', 'procedimental', 'axiológico', 'actitudinal', 
                            'comunicativo', 'resultado', 'intensidad_horaria']].copy()

        # Renombrar para el formato final esperado
        df_short.rename(columns={
            'cognitivo': 'cognitivo',
            'procedimental': 'procedimental',
            'axiológico': 'axiológico',
            'actitudinal': 'actitudinal',
            'comunicativo': 'comunicativo',
            'resultado': 'resultado',
            'sede': 'sede',
            'año': 'año',
            'periodo': 'periodo',
            'grado': 'grado',
            'identificación': 'identificación',
            'estudiante': 'estudiante',
            'asignatura': 'asignatura'
        }, inplace=True)

        self.logger.info(f"Calificaciones 2021-2022 procesadas: {len(df_short)} registros")
        return df_short

    def process_grades_2023_2025(self) -> pd.DataFrame:
        """
        Procesa calificaciones de los años 2023-2025 desde archivo CSV.

        Returns:
            DataFrame con calificaciones procesadas en formato short
        """
        self.logger.info("Procesando calificaciones 2023-2025...")

        # Cargar datos desde el archivo imputado
        csv_path = os.path.join(self.interim_dir, 'calificaciones', 'data_imputed_notes_2023_2025.csv')
        calificaciones = pd.read_csv(csv_path)

        # Filtrar solo años 2023-2025
        calificaciones = calificaciones[calificaciones['Año'].isin([2023, 2024, 2025])].copy()

        # Eliminar filas con valores nulos en columnas críticas
        columnas_criticas = ['Sede', 'Estudiante', 'Proc', 'Cog', 'Act', 'Axi', 'Resultado']
        filas_iniciales = len(calificaciones)

        # Identificar filas con valores nulos en columnas críticas
        mask_filas_vacias = calificaciones[columnas_criticas].isna().any(axis=1)
        filas_a_eliminar = calificaciones[mask_filas_vacias]

        if len(filas_a_eliminar) > 0:
            self.logger.warning(f"⚠️  Se encontraron {len(filas_a_eliminar)} filas con valores nulos en columnas críticas.")
            self.logger.warning(f"Columnas críticas: {columnas_criticas}")

            # Mostrar resumen de las filas a eliminar
            estudiantes_afectados = filas_a_eliminar['Identificación'].nunique()
            self.logger.warning(f"Estudiantes afectados: {estudiantes_afectados}")

            # Eliminar las filas con valores nulos en columnas críticas
            calificaciones = calificaciones[~mask_filas_vacias].copy()
            filas_despues = len(calificaciones)

            self.logger.info(f"✓ Se eliminaron {filas_iniciales - filas_despues} filas incompletas.")
            self.logger.info(f"Registros restantes: {filas_despues}")

        # Verificar y reportar filas con Sede nula (después de eliminar filas críticas)
        filas_sin_sede_inicial = calificaciones['Sede'].isna().sum()
        if filas_sin_sede_inicial > 0:
            self.logger.warning(f"⚠️  Se encontraron {filas_sin_sede_inicial} filas con Sede nula.")
            self.logger.info("Intentando rellenar la Sede basándose en otros registros del mismo estudiante...")

            # Crear un mapeo de Identificación -> Sede basado en registros no nulos
            estudiante_sede_map = calificaciones[calificaciones['Sede'].notna()].groupby('Identificación')['Sede'].first().to_dict()

            # Rellenar los valores nulos usando el mapeo
            mask_sede_nula = calificaciones['Sede'].isna()
            calificaciones.loc[mask_sede_nula, 'Sede'] = calificaciones.loc[mask_sede_nula, 'Identificación'].map(estudiante_sede_map)

            # Verificar cuántos se pudieron rellenar
            filas_sin_sede_despues = calificaciones['Sede'].isna().sum()
            filas_rellenadas = filas_sin_sede_inicial - filas_sin_sede_despues

            if filas_rellenadas > 0:
                self.logger.info(f"✓ Se rellenaron {filas_rellenadas} filas con la sede del mismo estudiante.")

            if filas_sin_sede_despues > 0:
                estudiantes_sin_sede = calificaciones[calificaciones['Sede'].isna()]['Identificación'].unique()
                self.logger.error(f"❌ Aún quedan {filas_sin_sede_despues} filas con Sede nula para {len(estudiantes_sin_sede)} estudiante(s).")
                self.logger.error(f"Estos estudiantes no tienen ningún registro con Sede: {list(estudiantes_sin_sede)}")

        # Estandarizar nombres de sede
        calificaciones.loc[:, 'Sede'] = calificaciones['Sede'].replace({
            'FUSAGASUGÁ': 'Fusagasugá',
            'GIRARDOT': 'Girardot'
        })

        # Mapeo de asignaturas usando el archivo asignaturas.csv por año
        asignaturas_bd = pd.read_csv(os.path.join(self.raw_dir, 'tablas_maestras', 'asignaturas.csv'))

        # Crear mapeo de asignaturas por año
        mapa_asignaturas = {}

        # Obtener años únicos en los datos
        años_unicos = calificaciones['Año'].unique()
        self.logger.info(f"Años encontrados en los datos: {sorted(años_unicos)}")

        for año in años_unicos:
            columna_año = f"nombre_{año}"
            if columna_año in asignaturas_bd.columns:
                self.logger.info(f"Usando columna '{columna_año}' para mapeo del año {año}")

                for _, fila in asignaturas_bd.iterrows():
                    nombre = fila[columna_año]
                    if pd.notna(nombre) and str(nombre).strip():
                        nombre_normalizado = str(nombre).strip().lower()
                        if nombre_normalizado not in mapa_asignaturas:
                            mapa_asignaturas[nombre_normalizado] = int(fila["id_asignatura"])
                            self.logger.debug(f"Mapeado: '{nombre_normalizado}' -> {fila['id_asignatura']}")
            else:
                self.logger.warning(f"No se encontró columna '{columna_año}' en asignaturas.csv para el año {año}")

        # Si no se encontraron mapeos por año, usar la columna 'nombre' como fallback
        if not mapa_asignaturas:
            self.logger.warning("No se encontraron mapeos por año, usando columna 'nombre' como fallback")
            for _, fila in asignaturas_bd.iterrows():
                nombre = fila["nombre"]
                if pd.notna(nombre) and str(nombre).strip():
                    nombre_normalizado = str(nombre).strip().lower()
                    if nombre_normalizado not in mapa_asignaturas:
                        mapa_asignaturas[nombre_normalizado] = int(fila["id_asignatura"])

        self.logger.info(f"Total de asignaturas mapeadas: {len(mapa_asignaturas)}")

        calificaciones["asignatura_normalizada"] = calificaciones["Asignatura"].astype(str).str.strip().str.lower()
        calificaciones["id_asignatura"] = calificaciones["asignatura_normalizada"].map(mapa_asignaturas)

        # Verificar que no haya asignaturas sin mapear
        asignaturas_sin_mapear = calificaciones[calificaciones['id_asignatura'].isna()]
        if not asignaturas_sin_mapear.empty:
            asignaturas_unicas_sin_mapear = asignaturas_sin_mapear['Asignatura'].unique()
            self.logger.error(f"❌ Se encontraron {len(asignaturas_unicas_sin_mapear)} asignaturas sin mapear:")
            for asignatura in asignaturas_unicas_sin_mapear:
                self.logger.error(f"  - '{asignatura}'")
            raise ValueError(f"Se encontraron {len(asignaturas_unicas_sin_mapear)} asignaturas sin mapear. "
                           f"Revisa el mapeo de asignaturas en el script.")

        # Agregar columna de intensidad horaria (valor por defecto para 2023-2025)
        calificaciones['intensidad_horaria'] = None

        # Seleccionar las columnas necesarias para formato short
        columns_to_keep = ['Sede', 'Estudiante', 'Grado', 'Periodo', 'Año', 
                          'Cog', 'Proc', 'Act', 'Axi', 'Resultado', 'Identificación', 'id_asignatura', 'intensidad_horaria']
        calificaciones_selected = calificaciones[columns_to_keep].copy()

        # Renombrar para el formato final esperado
        calificaciones_selected = calificaciones_selected.rename(columns={
            'Sede': 'sede',
            'Estudiante': 'estudiante',
            'Grado': 'grado',
            'Periodo': 'periodo',
            'Año': 'año',
            'Cog': 'cognitivo',
            'Proc': 'procedimental',
            'Act': 'actitudinal',
            'Axi': 'axiológico',
            'Resultado': 'resultado',
            'Identificación': 'identificación',
            'id_asignatura': 'asignatura'
        })

        self.logger.info(f"Calificaciones 2023-2025 procesadas: {len(calificaciones_selected)} registros")
        return calificaciones_selected

    def classify_learning_level(self, valor: float) -> Optional[str]:
        """
        Clasifica el nivel de aprendizaje basado en el valor numérico.

        Args:
            valor: Valor numérico de la calificación

        Returns:
            Nivel de aprendizaje o None si el valor es nulo
        """
        if pd.isna(valor):
            return None
        elif valor < 60:
            return "Bajo"
        elif valor < 80:
            return "Básico"
        elif valor < 95:
            return "Alto"
        else:
            return "Superior"

    def consolidate_datasets(self, df_2021_2022: pd.DataFrame, df_2023_2025: pd.DataFrame) -> pd.DataFrame:
        """
        Consolida los datasets de diferentes períodos.

        Args:
            df_2021_2022: DataFrame de calificaciones 2021-2022
            df_2023_2025: DataFrame de calificaciones 2023-2025

        Returns:
            DataFrame consolidado
        """
        self.logger.info("Consolidando datasets...")

        # Asegurar que las columnas sean consistentes
        columnas_esperadas = ['sede', 'año', 'periodo', 'grado', 'identificación', 'estudiante', 
                            'asignatura', 'cognitivo', 'procedimental', 'axiológico', 'actitudinal', 'comunicativo', 'resultado', 'intensidad_horaria']

        # Verificar que ambos datasets tengan las columnas necesarias
        for df, nombre in [(df_2021_2022, "2021-2022"), (df_2023_2025, "2023-2025")]:
            columnas_faltantes = set(columnas_esperadas) - set(df.columns)
            if columnas_faltantes:
                self.logger.warning(f"Columnas faltantes en dataset {nombre}: {columnas_faltantes}")

        # Concatenar datasets
        df_consolidado = pd.concat([df_2021_2022, df_2023_2025], ignore_index=True, sort=False)

        # Ordenar por año, sede, estudiante, asignatura, período
        df_consolidado = df_consolidado.sort_values(['año', 'sede', 'identificación', 'asignatura', 'periodo'])

        self.logger.info(f"Dataset consolidado: {len(df_consolidado)} registros totales")
        return df_consolidado

    def run(self):
        """Ejecuta el proceso completo de procesamiento de calificaciones históricas."""
        try:
            self.logger.info("Iniciando procesamiento de calificaciones históricas...")

            # Procesar calificaciones 2021-2022
            df_2021_2022 = self.process_grades_2021_2022()

            # Procesar calificaciones 2023-2025
            df_2023_2025 = self.process_grades_2023_2025()

            # Consolidar datasets
            df_consolidado = self.consolidate_datasets(df_2021_2022, df_2023_2025)

            # Guardar dataset consolidado
            output_path = os.path.join(self.interim_dir, 'calificaciones', 'calificaciones_2021-2025.csv')
            self.save_to_csv(df_consolidado, output_path)

            # Estadísticas finales
            self.logger.info("=== RESUMEN DEL PROCESAMIENTO ===")
            self.logger.info(f"Total de registros procesados: {len(df_consolidado)}")
            self.logger.info(f"Período cubierto: 2021-2025")
            self.logger.info(f"Archivo guardado en: {output_path}")

            # Estadísticas por año
            if 'año' in df_consolidado.columns:
                stats_por_año = df_consolidado['año'].value_counts().sort_index()
                self.logger.info("Registros por año:")
                for año, count in stats_por_año.items():
                    self.logger.info(f"  {año}: {count} registros")

            # Estadísticas por sede
            if 'sede' in df_consolidado.columns:
                stats_por_sede = df_consolidado['sede'].value_counts()
                self.logger.info("Registros por sede:")
                for sede, count in stats_por_sede.items():
                    self.logger.info(f"  {sede}: {count} registros")

            self.logger.info("Procesamiento completado exitosamente.")

        except Exception as e:
            self.logger.error(f"Error durante el procesamiento: {str(e)}")
            raise


def main():
    """Función principal para ejecutar el script."""
    processor = ProcessHistoricGrades()
    try:
        processor.run()
    finally:
        processor.close()


if __name__ == "__main__":
    main()
