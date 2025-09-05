import pandas as pd
import os
import sys
import numpy as np
from difflib import SequenceMatcher
from itertools import combinations

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.base_script import BaseScript


class SequenceAnalyzer(BaseScript):
    """
    Analiza secuencias de actividades comparando secuencias reales vs. ideales.
    Genera características derivadas para análisis de patrones de aprendizaje.
    """

    def __init__(self):
        super().__init__()
        self.logger.info("Inicializando SequenceAnalyzer...")

    @staticmethod
    def levenshtein_distance(seq1, seq2):
        """
        Calcula la distancia de Levenshtein entre dos secuencias.
        Número mínimo de operaciones (inserción, borrado, substitución).
        """
        if not seq1 and not seq2:
            return 0
        if not seq1:
            return len(seq2)
        if not seq2:
            return len(seq1)

        # Convertir secuencias a listas si son strings
        if isinstance(seq1, str):
            seq1 = seq1.split(',') if seq1 else []
        if isinstance(seq2, str):
            seq2 = seq2.split(',') if seq2 else []

        # Crear matriz de distancias
        rows = len(seq1) + 1
        cols = len(seq2) + 1
        
        # Inicializar matriz
        dist = [[0 for _ in range(cols)] for _ in range(rows)]

        # Llenar primera fila y columna
        for i in range(1, rows):
            dist[i][0] = i
        for j in range(1, cols):
            dist[0][j] = j

        # Llenar matriz
        for i in range(1, rows):
            for j in range(1, cols):
                if seq1[i-1] == seq2[j-1]:
                    cost = 0
                else:
                    cost = 1

                dist[i][j] = min(
                    dist[i-1][j] + 1,      # borrado
                    dist[i][j-1] + 1,      # inserción
                    dist[i-1][j-1] + cost  # substitución
                )
        return dist[rows-1][cols-1]

    @staticmethod
    def substring_similarity(seq1, seq2):
        """
        Calcula la similitud de substring usando SequenceMatcher.
        Puntaje de la subcadena mejor alineada.
        """
        if not seq1 and not seq2:
            return 1.0
        if not seq1 or not seq2:
            return 0.0

        # Convertir a strings para SequenceMatcher
        if isinstance(seq1, list):
            seq1 = ','.join(seq1)
        if isinstance(seq2, list):
            seq2 = ','.join(seq2)
        return SequenceMatcher(None, seq1, seq2).ratio()

    @staticmethod
    def get_ngrams(sequence, n):
        """
        Genera n-gramas de una secuencia.
        """
        if isinstance(sequence, str):
            sequence = sequence.split(',') if sequence else []

        if len(sequence) < n:
            return []

        return [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]

    @staticmethod
    def common_ngrams_count(seq1, seq2, n):
        """
        Cuenta n-gramas comunes entre dos secuencias.
        """
        ngrams1 = set(SequenceAnalyzer.get_ngrams(seq1, n))
        ngrams2 = set(SequenceAnalyzer.get_ngrams(seq2, n))

        return len(ngrams1.intersection(ngrams2))

    def analyze_sequence_features(self, real_sequence, ideal_sequence):
        """
        Analiza una secuencia individual y genera todas las características.
        """
        # Convertir secuencias a listas
        real_list = real_sequence.split(',') if real_sequence else []
        ideal_list = ideal_sequence.split(',') if ideal_sequence else []

        # Remover elementos vacíos
        real_list = [x.strip() for x in real_list if x.strip()]
        ideal_list = [x.strip() for x in ideal_list if x.strip()]

        features = {}

        # a) Longitud de la secuencia
        features['real_sequence_length'] = len(real_list)
        features['ideal_sequence_length'] = len(ideal_list)
        features['total_accesses'] = len(real_list)

        # b) Distancia de Levenshtein
        features['levenshtein_distance'] = self.levenshtein_distance(real_list, ideal_list)

        # Distancia normalizada (0-1)
        max_len = max(len(real_list), len(ideal_list), 1)
        features['levenshtein_normalized'] = features['levenshtein_distance'] / max_len

        # c) Substring similarity
        features['substring_similarity'] = self.substring_similarity(real_list, ideal_list)
        
        # d) N-gramas comunes
        features['common_bigrams'] = self.common_ngrams_count(real_list, ideal_list, 2)
        features['common_trigrams'] = self.common_ngrams_count(real_list, ideal_list, 3)

        # Métricas adicionales útiles
        features['sequence_match_ratio'] = len(set(real_list).intersection(set(ideal_list))) / max(len(set(ideal_list)), 1)
        features['extra_activities'] = max(0, len(real_list) - len(ideal_list))
        features['missing_activities'] = max(0, len(ideal_list) - len(real_list))

        # Orden correcto (actividades en la posición correcta)
        correct_order = sum(1 for i, (r, i_item) in enumerate(zip(real_list, ideal_list)) if r == i_item)
        features['correct_order_count'] = correct_order
        features['correct_order_ratio'] = correct_order / max(len(ideal_list), 1)

        return features

    def load_data(self):
        """
        Carga los datos necesarios para el análisis.
        """
        self.logger.info("Cargando datos...")

        # Cargar secuencias reales
        real_sequences = pd.read_csv("data/interim/moodle/student_activity_sequences.csv")

        # Cargar secuencias ideales
        ideal_courses = pd.read_csv("data/interim/moodle/courses.csv")

        return real_sequences, ideal_courses

    def process_sequence_analysis(self):
        """
        Proceso principal para analizar secuencias y generar características.
        """
        # Cargar datos
        real_sequences, ideal_courses = self.load_data()

        self.logger.info(f"Secuencias reales: {len(real_sequences)} registros")
        self.logger.info(f"Cursos ideales: {len(ideal_courses)} registros")

        # Hacer merge para combinar secuencias reales con ideales
        merged_data = real_sequences.merge(
            ideal_courses[['id_asignatura', 'id_grado', 'year', 'period', 'sede', 'sequence']],
            on=['id_asignatura', 'id_grado', 'year', 'period', 'sede'],
            how='left',
            suffixes=('_real', '_ideal')
        )

        self.logger.info(f"Registros después del merge: {len(merged_data)}")
        self.logger.info(f"Registros con secuencia ideal: {merged_data['sequence'].notna().sum()}")
        
        # Filtrar solo registros que tienen secuencia ideal
        merged_data = merged_data.dropna(subset=['sequence'])

        self.logger.info(f"Registros para análisis: {len(merged_data)}")

        # Analizar cada secuencia
        self.logger.info("Analizando secuencias...")

        all_features = []
        for idx, row in merged_data.iterrows():
            if idx % 1000 == 0:
                self.logger.info(f"Procesando registro {idx}/{len(merged_data)}")

            # Analizar secuencia
            features = self.analyze_sequence_features(
                row['activity_sequence'], 
                row['sequence']
            )

            # Agregar información base
            features['documento_identificación'] = row['documento_identificación']
            features['id_asignatura'] = row['id_asignatura']
            features['id_grado'] = row['id_grado']
            features['sede'] = row['sede']
            features['year'] = row['year']
            features['period'] = row['period']
            features['platforms_used'] = row['platforms_used']
            features['real_sequence'] = row['activity_sequence']
            features['ideal_sequence'] = row['sequence']
            all_features.append(features)

        # Crear DataFrame con resultados
        results_df = pd.DataFrame(all_features)

        # Guardar resultados
        output_file = "data/interim/moodle/sequence_analysis_features.csv"
        self.save_to_csv(results_df, output_file)

        # Estadísticas finales
        self.logger.info("=== ESTADÍSTICAS DE ANÁLISIS ===")
        self.logger.info(f"Total de secuencias analizadas: {len(results_df)}")
        self.logger.info(f"Distancia Levenshtein promedio: {results_df['levenshtein_distance'].mean():.2f}")
        self.logger.info(f"Similitud substring promedio: {results_df['substring_similarity'].mean():.3f}")
        self.logger.info(f"Bi-gramas comunes promedio: {results_df['common_bigrams'].mean():.2f}")
        self.logger.info(f"Tri-gramas comunes promedio: {results_df['common_trigrams'].mean():.2f}")
        self.logger.info(f"Ratio de orden correcto promedio: {results_df['correct_order_ratio'].mean():.3f}")
        return results_df
    
    def show_example_analysis(self):
        """
        Muestra un ejemplo de cómo funcionan las métricas de análisis.
        """
        print("\n=== EJEMPLO DE ANÁLISIS DE SECUENCIAS ===")

        # Ejemplo 1: Secuencias similares
        real1 = "A1,Q2,F3,H4"
        ideal1 = "A1,Q2,F3,H4"
        
        print(f"\nEjemplo 1 - Secuencias idénticas:")
        print(f"Real:  {real1}")
        print(f"Ideal: {ideal1}")

        features1 = self.analyze_sequence_features(real1, ideal1)
        print(f"Distancia Levenshtein: {features1['levenshtein_distance']}")
        print(f"Similitud substring: {features1['substring_similarity']:.3f}")
        print(f"Bi-gramas comunes: {features1['common_bigrams']}")
        print(f"Orden correcto: {features1['correct_order_ratio']:.3f}")

        # Ejemplo 2: Secuencias con orden diferente
        real2 = "A1,F3,Q2,H4"
        ideal2 = "A1,Q2,F3,H4"

        print(f"\nEjemplo 2 - Orden diferente:")
        print(f"Real:  {real2}")
        print(f"Ideal: {ideal2}")

        features2 = self.analyze_sequence_features(real2, ideal2)
        print(f"Distancia Levenshtein: {features2['levenshtein_distance']}")
        print(f"Similitud substring: {features2['substring_similarity']:.3f}")
        print(f"Bi-gramas comunes: {features2['common_bigrams']}")
        print(f"Orden correcto: {features2['correct_order_ratio']:.3f}")

        # Ejemplo 3: Secuencias mixtas Moodle+Edukrea
        real3 = "A1,eH2,Q3,eF4"
        ideal3 = "A1,Q3,eH2,eF4"

        print(f"\nEjemplo 3 - Secuencia mixta (Moodle + Edukrea):")
        print(f"Real:  {real3}")
        print(f"Ideal: {ideal3}")

        features3 = self.analyze_sequence_features(real3, ideal3)
        print(f"Distancia Levenshtein: {features3['levenshtein_distance']}")
        print(f"Similitud substring: {features3['substring_similarity']:.3f}")
        print(f"Bi-gramas comunes: {features3['common_bigrams']}")
        print(f"Orden correcto: {features3['correct_order_ratio']:.3f}")


if __name__ == "__main__":
    analyzer = SequenceAnalyzer()
    
    # Mostrar ejemplo de análisis
    analyzer.show_example_analysis()
    
    # Ejecutar análisis completo
    analyzer.process_sequence_analysis()
    analyzer.logger.info("Análisis de secuencias completado exitosamente.")
    analyzer.close()
