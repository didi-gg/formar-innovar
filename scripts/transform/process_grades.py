import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.base_script import BaseScript


class GradesProcessor(BaseScript):

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

        column_mapping = {
            'Sede': 'sede',
            'Estudiante': 'estudiante', 
            'Grado': 'id_grado',
            'Periodo': 'periodo',
            'Año': 'año',
            'Cog': 'cog',
            'Proc': 'proc',
            'Act': 'act',
            'Axi': 'axi',
            'Resultado': 'resultado',
            'Nivel': 'nivel',
            'Identificación': 'documento_identificación'
        }

        grades_final = grades_filtered[list(column_mapping.keys())].rename(columns=column_mapping)
        self.save_to_csv(grades_final, "data/interim/calificaciones/calificaciones_2024_2025.csv")
        self.logger.info("Procesamiento de calificaciones completado.")


if __name__ == "__main__":
    processor = GradesProcessor()
    processor.process_grades()
    processor.logger.info("Grades processed successfully.")
    processor.close()
