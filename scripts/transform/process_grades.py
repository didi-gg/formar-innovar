import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.base_script import BaseScript


class GradesProcessor(BaseScript):

    def verificar_valores_nulos(self, df, nombre_dataset):
        """
        Verifica si hay valores nulos en cualquier columna del DataFrame
        Args:
            df: DataFrame a verificar
            nombre_dataset: Nombre descriptivo del dataset para el mensaje de error
        """
        columnas_con_nulos = df.columns[df.isna().any()].tolist()

        if columnas_con_nulos:
            total_nulos_por_columna = df[columnas_con_nulos].isna().sum()

            self.logger.error(f"❌ Se encontraron valores nulos en el dataset '{nombre_dataset}':")
            for columna in columnas_con_nulos:
                cantidad_nulos = total_nulos_por_columna[columna]
                self.logger.error(f"   - Columna '{columna}': {cantidad_nulos} valores nulos")

            # Obtener todos los registros con nulos
            todos_los_nulos = df[df[columnas_con_nulos].isna().any(axis=1)]

            # Mostrar algunos ejemplos de registros con nulos en el logger
            self.logger.error(f"\n{'='*80}")
            self.logger.error(f"EJEMPLOS DE REGISTROS CON VALORES NULOS (mostrando primeros 10):")
            self.logger.error(f"{'='*80}")

            registros_muestra = todos_los_nulos.head(10)
            for idx, row in registros_muestra.iterrows():
                self.logger.error(f"\nRegistro #{idx}:")
                for col in df.columns:
                    valor = row[col]
                    if pd.isna(valor):
                        self.logger.error(f"  {col}: NULL ⚠️")
                    else:
                        self.logger.error(f"  {col}: {valor}")

            self.logger.error(f"\n{'='*80}\n")

            raise ValueError(f"Se encontraron valores nulos en {len(columnas_con_nulos)} columna(s) del dataset '{nombre_dataset}'. "
                           f"Total de registros con nulos: {len(todos_los_nulos)}. ")
        else:
            self.logger.info(f"✓ No se encontraron valores nulos en el dataset '{nombre_dataset}'")
            return True

    def verificar_duplicados(self, df, clave_columns, asignatura_mapping=None):
        """
        Verifica si hay duplicados por la llave especificada
        Args:
            df: DataFrame a verificar
            clave_columns: Lista de columnas que forman la llave
            asignatura_mapping: Diccionario opcional con el mapeo nombre->id de asignaturas
        """
        # Contar duplicados por la llave
        duplicados = df.groupby(clave_columns).size().reset_index(name='count')
        duplicados_encontrados = duplicados[duplicados['count'] > 1]

        if not duplicados_encontrados.empty:
            # Obtener todos los registros duplicados
            condicion_todos_duplicados = df.set_index(clave_columns).index.isin(
                duplicados_encontrados.set_index(clave_columns).index
            )
            todos_los_duplicados = df[condicion_todos_duplicados].copy()

            # Agregar el nombre de la asignatura si se proporciona el mapeo
            if asignatura_mapping is not None and 'id_asignatura' in todos_los_duplicados.columns:
                # Crear mapeo inverso (id -> nombre)
                asignatura_id_to_name = {v: k for k, v in asignatura_mapping.items()}

                # Mapear el nombre de la asignatura
                todos_los_duplicados['nombre_asignatura'] = todos_los_duplicados['id_asignatura'].map(asignatura_id_to_name)

                # Reordenar columnas para que el nombre de la asignatura aparezca junto al id
                columns = list(todos_los_duplicados.columns)
                if 'nombre_asignatura' in columns:
                    # Insertar nombre_asignatura después de id_asignatura
                    id_asignatura_idx = columns.index('id_asignatura')
                    columns.insert(id_asignatura_idx + 1, columns.pop(columns.index('nombre_asignatura')))
                    todos_los_duplicados = todos_los_duplicados[columns]

            # Crear directorio logs si no existe
            logs_dir = "logs"
            os.makedirs(logs_dir, exist_ok=True)

            # Generar nombre de archivo con timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archivo_duplicados = os.path.join(logs_dir, f"duplicados_calificaciones_{timestamp}.csv")

            # Ordenar los duplicados por la llave para facilitar el análisis
            todos_los_duplicados = todos_los_duplicados.sort_values(clave_columns)

            # Guardar todos los duplicados en un archivo CSV
            todos_los_duplicados.to_csv(archivo_duplicados, index=False)

            # Lanzar excepción con información del archivo de log
            raise ValueError(f"Se encontraron {len(duplicados_encontrados)} combinaciones de llave duplicadas. "
                           f"Revisa el archivo de log: {archivo_duplicados}")
        else:
            self.logger.info("✓ No se encontraron duplicados por la llave especificada")
            return True

    def process_grades(self):
        grades = pd.read_csv("data/interim/calificaciones/data_imputed_notes_2023_2025.csv")
        grades_filtered = grades[grades['Año'].isin([2024, 2025])].copy()

        # Eliminar filas con valores nulos en columnas críticas
        columnas_criticas = ['Sede', 'Estudiante', 'Proc', 'Cog', 'Act', 'Axi', 'Resultado']
        filas_iniciales = len(grades_filtered)

        # Identificar filas con valores nulos en columnas críticas
        mask_filas_vacias = grades_filtered[columnas_criticas].isna().any(axis=1)
        filas_a_eliminar = grades_filtered[mask_filas_vacias]

        if len(filas_a_eliminar) > 0:
            self.logger.warning(f"⚠️  Se encontraron {len(filas_a_eliminar)} filas con valores nulos en columnas críticas.")
            self.logger.warning(f"Columnas críticas: {columnas_criticas}")

            # Mostrar resumen de las filas a eliminar
            estudiantes_afectados = filas_a_eliminar['Identificación'].nunique()
            self.logger.warning(f"Estudiantes afectados: {estudiantes_afectados}")

            # Eliminar las filas con valores nulos en columnas críticas
            grades_filtered = grades_filtered[~mask_filas_vacias].copy()
            filas_despues = len(grades_filtered)

            self.logger.info(f"✓ Se eliminaron {filas_iniciales - filas_despues} filas incompletas.")
            self.logger.info(f"Registros restantes: {filas_despues}")

        # Verificar y reportar filas con Sede nula (después de eliminar filas críticas)
        filas_sin_sede_inicial = grades_filtered['Sede'].isna().sum()
        if filas_sin_sede_inicial > 0:
            self.logger.warning(f"⚠️  Se encontraron {filas_sin_sede_inicial} filas con Sede nula.")
            self.logger.info("Intentando rellenar la Sede basándose en otros registros del mismo estudiante...")

            # Crear un mapeo de Identificación -> Sede basado en registros no nulos
            estudiante_sede_map = grades_filtered[grades_filtered['Sede'].notna()].groupby('Identificación')['Sede'].first().to_dict()

            # Rellenar los valores nulos usando el mapeo
            mask_sede_nula = grades_filtered['Sede'].isna()
            grades_filtered.loc[mask_sede_nula, 'Sede'] = grades_filtered.loc[mask_sede_nula, 'Identificación'].map(estudiante_sede_map)

            # Verificar cuántos se pudieron rellenar
            filas_sin_sede_despues = grades_filtered['Sede'].isna().sum()
            filas_rellenadas = filas_sin_sede_inicial - filas_sin_sede_despues

            if filas_rellenadas > 0:
                self.logger.info(f"✓ Se rellenaron {filas_rellenadas} filas con la sede del mismo estudiante.")

            if filas_sin_sede_despues > 0:
                estudiantes_sin_sede = grades_filtered[grades_filtered['Sede'].isna()]['Identificación'].unique()
                self.logger.error(f"❌ Aún quedan {filas_sin_sede_despues} filas con Sede nula para {len(estudiantes_sin_sede)} estudiante(s).")
                self.logger.error(f"Estos estudiantes no tienen ningún registro con Sede: {list(estudiantes_sin_sede)}")

        grades_filtered.loc[:, 'Sede'] = grades_filtered['Sede'].replace({
            'FUSAGASUGÁ': 'Fusagasugá',
            'GIRARDOT': 'Girardot'
        })

        asignatura_mapping = {
            'CIENCIAS NATURALES': 1,
            'CIENCIAS SOCIALES': 2,
            'MATEMÁTICAS': 3,
            'LENGUA CASTELLANA': 4,
            'LECTURA CRÍTICA': 9,
            'INGLÉS': 5,
            'EDUCACIÓN FÍSICA': 8,
            'CREATIVIDAD E INNOVACIÓN': 6,
            'APRENDIZAJE BASADO EN PROYECTOS': 7,
            'ARTES': 10,
            'TECNOLOGÍAS INFORMÁTICAS': 11,
            'INTEGRALIDAD': 12,
            'FRANCÉS': 17,
            'FILOSOFÍA': 16,
            'PLAN DE INVERSIÓN': 15,
            'INNOVACIÓN Y EMPRENDIMIENTO': 13,
            'APRENDIZAJE BASADO EN INVESTIGACIÓN': 14,
            'CENTRO DE INTERÉS ARTÍSTICO': 10,
            'FÍSICA': 23
        }

        grades_filtered.loc[:, 'id_asignatura'] = grades_filtered['Asignatura'].map(asignatura_mapping)

        # Seleccionar las columnas necesarias
        columns_to_keep = ['Sede', 'Estudiante', 'Grado', 'Periodo', 'Año', 
                          'Cog', 'Proc', 'Act', 'Axi', 'Resultado', 'Identificación', 'id_asignatura']
        grades_selected = grades_filtered[columns_to_keep].copy()

        # Renombrar columnas para el procesamiento
        grades_selected = grades_selected.rename(columns={
            'Sede': 'sede',
            'Estudiante': 'estudiante',
            'Grado': 'id_grado',
            'Periodo': 'period',
            'Año': 'year',
            'Identificación': 'documento_identificación'
        })

        # Verificar duplicados ANTES de la transformación a formato largo
        clave_duplicados = ['documento_identificación', 'id_asignatura', 'id_grado', 'period', 'year', 'sede']
        self.logger.info("Verificando duplicados antes de la transformación a formato largo...")

        try:
            self.verificar_duplicados(grades_selected, clave_duplicados, asignatura_mapping)
        except ValueError as e:
            self.logger.error(f"❌ Error: {e}")
            raise

        # Crear dataset en formato largo
        # Primero crear registros para proc, cog, act, axi
        dimensiones_base = ['Proc', 'Cog', 'Act', 'Axi']

        # Transformar a formato largo usando melt
        grades_long = pd.melt(grades_selected, 
                             id_vars=['documento_identificación', 'id_asignatura', 'id_grado', 
                                     'period', 'year', 'sede', 'estudiante', 'Resultado'],
                             value_vars=dimensiones_base,
                             var_name='dimensión',
                             value_name='resultado')

        # Convertir los nombres de dimensiones a minúsculas
        grades_long['dimensión'] = grades_long['dimensión'].str.lower()

        # Crear registros para la dimensión 'final' usando el valor de 'Resultado'
        grades_final_dimension = grades_selected[['documento_identificación', 'id_asignatura', 'id_grado', 
                                                 'period', 'year', 'sede', 'estudiante', 'Resultado']].copy()
        grades_final_dimension['dimensión'] = 'final'
        grades_final_dimension['resultado'] = grades_final_dimension['Resultado']

        # Eliminar la columna 'Resultado' temporal
        grades_final_dimension = grades_final_dimension.drop('Resultado', axis=1)
        grades_long = grades_long.drop('Resultado', axis=1)

        # Combinar ambos datasets
        grades_complete = pd.concat([grades_long, grades_final_dimension], ignore_index=True)

        # Función para calcular el nivel basado en el resultado
        def calcular_nivel(resultado):
            if pd.isna(resultado):
                return None
            elif resultado >= 95:
                return 'Superior'
            elif resultado >= 80:
                return 'Alto'
            elif resultado >= 60:
                return 'Básico'
            else:
                return 'Bajo'

        # Calcular el nivel
        grades_complete['nivel'] = grades_complete['resultado'].apply(calcular_nivel)

        # Reordenar las columnas según lo solicitado
        column_order = ['documento_identificación', 'id_asignatura', 'id_grado', 
                       'period', 'year', 'sede', 'estudiante', 'dimensión', 'resultado', 'nivel']
        grades_final = grades_complete[column_order]

        # Ordenar por documento_identificación, id_asignatura, period, year, dimensión
        grades_final = grades_final.sort_values(['documento_identificación', 'id_asignatura', 
                                               'period', 'year', 'dimensión'])

        # Verificar que no haya valores nulos antes de guardar
        self.logger.info("Verificando valores nulos en dataset largo...")
        try:
            self.verificar_valores_nulos(grades_final, "calificaciones_2024_2025_long")
        except ValueError as e:
            self.logger.error(f"❌ Error: {e}")
            raise

        self.save_to_csv(grades_final, "data/interim/calificaciones/calificaciones_2024_2025_long.csv")

        # Crear dataset en formato corto (filas originales)
        self.logger.info("Creando dataset en formato corto...")

        # Renombrar columnas para el dataset corto
        grades_short = grades_selected.rename(columns={
            'Proc': 'proc',
            'Cog': 'cog', 
            'Act': 'act',
            'Axi': 'axi',
            'Resultado': 'nota_final'
        })

        # Calcular el nivel basado en la nota final
        grades_short['nivel'] = grades_short['nota_final'].apply(calcular_nivel)

        # Reordenar las columnas para el dataset corto
        short_column_order = ['documento_identificación', 'id_asignatura', 'id_grado', 
                             'period', 'year', 'sede', 'estudiante', 'proc', 'cog', 
                             'act', 'axi', 'nota_final', 'nivel']
        grades_short_final = grades_short[short_column_order]

        # Ordenar por documento_identificación, id_asignatura, period, year
        grades_short_final = grades_short_final.sort_values(['documento_identificación', 'id_asignatura', 
                                                           'period', 'year'])

        # Verificar que no haya valores nulos antes de guardar
        self.logger.info("Verificando valores nulos en dataset corto...")
        try:
            self.verificar_valores_nulos(grades_short_final, "calificaciones_2024_2025_short")
        except ValueError as e:
            self.logger.error(f"❌ Error: {e}")
            raise

        self.save_to_csv(grades_short_final, "data/interim/calificaciones/calificaciones_2024_2025_short.csv")
        self.logger.info("Procesamiento de calificaciones completado.")


if __name__ == "__main__":
    processor = GradesProcessor()
    processor.process_grades()
    processor.logger.info("Grades processed successfully.")
    processor.close()
