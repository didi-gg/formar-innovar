import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.base_script import BaseScript


class GradesProcessor(BaseScript):

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
        grades_filtered = grades[grades['Año'].isin([2024, 2025])]
        
        grades_filtered['Sede'] = grades_filtered['Sede'].replace({
            'FUSAGASUGÁ': 'Fusagasugá',
            'GIRARDOT': 'Girardot'
        })

        grades_filtered = grades_filtered.copy()

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

        grades_filtered['id_asignatura'] = grades_filtered['Asignatura'].map(asignatura_mapping)

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

        self.save_to_csv(grades_final, "data/interim/calificaciones/calificaciones_2024_2025_long.csv")
        self.logger.info("Procesamiento de calificaciones completado.")


if __name__ == "__main__":
    processor = GradesProcessor()
    processor.process_grades()
    processor.logger.info("Grades processed successfully.")
    processor.close()
