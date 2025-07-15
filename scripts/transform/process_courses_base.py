import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.base_script import BaseScript


class CoursesBaseProcessor(BaseScript):

    def process_course_data(self):
        # Cargar archivo de carga horaria
        academic_load_path = "data/raw/tablas_maestras/carga_horaria.csv"
        academic_load = pd.read_csv(academic_load_path)
        self.logger.info(f"Cargados {len(academic_load)} registros de carga horaria")
        
        # Filtrar solo años 2024 y 2025
        academic_load = academic_load[academic_load['year'].isin([2024, 2025])]
        self.logger.info(f"Después de filtrar años 2024-2025: {len(academic_load)} registros")
        
        # Verificar duplicados en los datos originales antes del merge
        original_duplicate_keys = ['sede', 'year', 'id_grado', 'id_asignatura']
        original_duplicates = academic_load.duplicated(subset=original_duplicate_keys).sum()
        if original_duplicates > 0:
            self.logger.warning(f"Se encontraron {original_duplicates} duplicados en los datos originales de carga horaria")
            # Mostrar algunos ejemplos de duplicados
            duplicate_examples = academic_load[academic_load.duplicated(subset=original_duplicate_keys, keep=False)].head(10)
            self.logger.warning("Ejemplos de duplicados encontrados:")
            for _, row in duplicate_examples.iterrows():
                self.logger.warning(f"  {row['sede']}, {row['year']}, {row['id_grado']}, {row['id_asignatura']}, {row['docente']}")
        
        # Cargar archivo de teachers featured
        teachers_featured_path = "data/interim/moodle/teachers_featured.csv"
        teachers_featured = pd.read_csv(teachers_featured_path)
        self.logger.info(f"Cargados {len(teachers_featured)} registros de teachers featured")
        
        # Verificar la distribución de teachers_featured por año
        teachers_by_year = teachers_featured.groupby('year').size()
        self.logger.info("Distribución de teachers_featured por año:")
        for year, count in teachers_by_year.items():
            self.logger.info(f"  Año {year}: {count} registros")
        
        # Seleccionar las columnas específicas de teachers_featured
        columns_to_merge = [
            'id_docente', 'year', 'nombre', 'rol_adicional', 'nivel_educativo', 
            'total_subjects', 'total_hours', 'unique_students_count', 
            'update_events_count', 'years_experience_ficc', 
            'years_experience_total', 'age'
        ]
        
        teachers_subset = teachers_featured[columns_to_merge].copy()
        
        # Hacer el merge por id_docente Y year para evitar duplicados
        merged_data = academic_load.merge(
            teachers_subset, 
            on=['id_docente', 'year'], 
            how='inner'
        )
        
        self.logger.info(f"Merge completado. Resultado: {len(merged_data)} registros")
        self.logger.info(f"Registros originales filtrados: {len(academic_load)}")
        self.logger.info(f"Registros después del merge: {len(merged_data)}")
        
        # Verificar que no hay registros sin match (debería ser 0 con inner join)
        no_match_count = len(academic_load) - len(merged_data)
        if no_match_count > 0:
            self.logger.warning(f"Se perdieron {no_match_count} registros sin match en teachers_featured")
            
            # Identificar qué registros no hicieron match
            merged_keys = merged_data[['id_docente', 'year']].drop_duplicates()
            academic_keys = academic_load[['id_docente', 'year']].drop_duplicates()
            
            # Encontrar las combinaciones que no hicieron match
            no_match_keys = academic_keys.merge(merged_keys, on=['id_docente', 'year'], how='left', indicator=True)
            no_match_keys = no_match_keys[no_match_keys['_merge'] == 'left_only'][['id_docente', 'year']]
            
            self.logger.warning(f"Registros sin match (combinaciones id_docente-year):")
            for _, row in no_match_keys.iterrows():
                self.logger.warning(f"  id_docente: {row['id_docente']}, year: {row['year']}")
            
            # Mostrar ejemplos de registros completos que no hicieron match
            no_match_examples = academic_load.merge(no_match_keys, on=['id_docente', 'year'], how='inner')
            self.logger.warning(f"Ejemplos de registros completos sin match:")
            for _, row in no_match_examples.head(10).iterrows():
                self.logger.warning(f"  {row['sede']}, {row['year']}, {row['id_grado']}, {row['id_asignatura']}, {row['docente']}, id_docente: {row['id_docente']}")
        else:
            self.logger.info("Todos los registros hicieron match exitosamente")
        
        # Separar datos por año antes de crear períodos
        data_2024 = merged_data[merged_data['year'] == 2024].copy()
        data_2025 = merged_data[merged_data['year'] == 2025].copy()
        
        # Para 2025, agregar period = 1
        data_2025['period'] = 1
        
        # Para 2024, crear registros para cada período (1, 2, 3, 4)
        periods_2024 = []
        for period in [1, 2, 3, 4]:
            period_data = data_2024.copy()
            period_data['period'] = period
            periods_2024.append(period_data)
        
        # Combinar todos los datos
        all_2024_data = pd.concat(periods_2024, ignore_index=True)
        final_data = pd.concat([all_2024_data, data_2025], ignore_index=True)
        
        # Asegurar que period sea de tipo int
        final_data['period'] = final_data['period'].astype(int)
        
        self.logger.info(f"Datos finales con períodos: {len(final_data)} registros")
        self.logger.info(f"Tipo de datos de period: {final_data['period'].dtype}")
        self.logger.info(f"Distribución por año y período:")
        period_distribution = final_data.groupby(['year', 'period']).size()
        for (year, period), count in period_distribution.items():
            self.logger.info(f"  Año {year}, Período {period}: {count} registros")
        
        # Verificar que no hay duplicados en la llave compuesta FINAL
        duplicate_keys = ['sede', 'year', 'id_grado', 'id_asignatura', 'period']
        duplicates_check = final_data.duplicated(subset=duplicate_keys).sum()
        
        if duplicates_check > 0:
            self.logger.error(f"ERROR CRÍTICO: Se encontraron {duplicates_check} duplicados en la llave compuesta final")
            
            # Mostrar ejemplos de duplicados para debugging
            duplicate_examples = final_data[final_data.duplicated(subset=duplicate_keys, keep=False)]
            self.logger.error("Ejemplos de duplicados encontrados:")
            for _, row in duplicate_examples.head(10).iterrows():
                self.logger.error(f"  {row['sede']}, {row['year']}, {row['id_grado']}, {row['id_asignatura']}, {row['period']}")
            
            # Lanzar excepción para detener el procesamiento
            raise ValueError(f"Se encontraron {duplicates_check} duplicados en la llave compuesta [{', '.join(duplicate_keys)}]. "
                           f"Esto indica un problema en la lógica del script que debe ser corregido.")
        else:
            self.logger.info("✓ Verificación exitosa: No hay duplicados en la llave compuesta")
        
        # Guardar resultado
        output_path = "data/interim/moodle/courses_base.csv"
        self.save_to_csv(final_data, output_path)
        self.logger.info(f"Procesamiento de cursos completado. Archivo guardado en: {output_path}")
        return final_data

if __name__ == "__main__":
    processor = CoursesBaseProcessor()
    result = processor.process_course_data()
    processor.logger.info("Courses base processed successfully.")
    processor.close()
