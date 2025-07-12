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
        
        # Cargar archivo de teachers featured
        teachers_featured_path = "data/interim/moodle/teachers_featured.csv"
        teachers_featured = pd.read_csv(teachers_featured_path)
        self.logger.info(f"Cargados {len(teachers_featured)} registros de teachers featured")
        
        # Seleccionar las columnas específicas de teachers_featured
        columns_to_merge = [
            'id_docente', 'nombre', 'rol_adicional', 'nivel_educativo', 
            'total_subjects', 'total_hours', 'unique_students_count', 
            'update_events_count', 'years_experience_ficc', 
            'years_experience_total', 'age'
        ]
        
        teachers_subset = teachers_featured[columns_to_merge].copy()
        
        # Hacer el merge por id_docente
        merged_data = academic_load.merge(
            teachers_subset, 
            on='id_docente', 
            how='inner'
        )
        
        self.logger.info(f"Merge completado. Resultado: {len(merged_data)} registros")
        self.logger.info(f"Registros originales filtrados: {len(academic_load)}")
        self.logger.info(f"Registros después del merge: {len(merged_data)}")
        
        # Verificar que no hay registros sin match (debería ser 0 con inner join)
        no_match_count = len(academic_load) - len(merged_data)
        if no_match_count > 0:
            self.logger.warning(f"Se perdieron {no_match_count} registros sin match en teachers_featured")
        else:
            self.logger.info("Todos los registros hicieron match exitosamente")
        
        # Agregar columna period
        # Para 2024: valores 1, 2, 3, 4
        # Para 2025: solo valor 1
        merged_data['period'] = merged_data['year'].apply(
            lambda x: 1 if x == 2025 else 1
        )
        
        # Para 2024, crear registros duplicados para cada período
        data_2024 = merged_data[merged_data['year'] == 2024].copy()
        data_2025 = merged_data[merged_data['year'] == 2025].copy()
        
        # Crear múltiples registros para 2024 (períodos 1, 2, 3, 4)
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
