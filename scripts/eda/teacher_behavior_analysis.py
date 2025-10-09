"""
Script para análisis del comportamiento de docentes en calificaciones
Analiza el comportamiento de las calificaciones por docente en área, asignatura, grado y género
"""

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Agregar el directorio padre al path para importar utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.eda_analysis_base import EDAAnalysisBase

class TeacherBehaviorAnalysis(EDAAnalysisBase):

    def _initialize_analysis_attributes(self):
        pass

    def load_and_prepare_data(self) -> pd.DataFrame:
        # Cargar el dataset principal que se pasa como parámetro
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"No se encontró el dataset en: {self.dataset_path}")

        df = pd.read_csv(self.dataset_path)

        # Verificar que tenga las columnas necesarias
        required_columns = ['id_asignatura', 'nota_final', 'id_docente', 'id_grado']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Columnas requeridas faltantes en el dataset: {missing_columns}")

        self.df_merged = df
        return df


    def create_simple_tecreate_teacher_subject_boxplotsacher_boxplots(self, output_dir: str):
        self.logger.info("Creando boxplots simples por docente...")

        # Obtener todos los docentes con datos
        teacher_counts = self.df_merged.groupby('id_docente').size()
        teachers_with_data = teacher_counts[teacher_counts >= 3].index.tolist()  # Al menos 3 calificaciones

        if len(teachers_with_data) == 0:
            self.logger.warning("No hay docentes con suficientes datos")
            return

        # Calcular número de filas y columnas dinámicamente
        n_teachers = len(teachers_with_data)
        n_cols = 4  # 4 columnas fijas
        n_rows = (n_teachers + n_cols - 1) // n_cols  # Redondear hacia arriba

        # Crear figura con subplots dinámicos
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows + 2))  # +2 para espacio del título
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for i, teacher_id in enumerate(teachers_with_data):
            if i >= len(axes):
                break

            # Datos del docente
            df_teacher = self.df_merged[self.df_merged['id_docente'] == teacher_id]

            # Crear boxplot por asignatura
            unique_subjects_teacher = sorted(df_teacher['id_asignatura'].unique())
            n_subjects_teacher = len(unique_subjects_teacher)
            colors_teacher = self.get_beautiful_palette(n_subjects_teacher, 'tab20b')

            sns.boxplot(data=df_teacher, x='id_asignatura', y='nota_final', palette=colors_teacher, ax=axes[i])
            axes[i].set_title(f'Docente ID: {teacher_id}\n({df_teacher["id_asignatura"].nunique()} asignatura{"s" if df_teacher["id_asignatura"].nunique() > 1 else ""})')
            axes[i].set_xlabel('ID Asignatura')
            axes[i].set_ylabel('Calificaciones')

            # Agregar promedio general del docente
            overall_mean = df_teacher['nota_final'].mean()
            axes[i].axhline(y=overall_mean, color='red', linestyle='--', alpha=0.7, label=f'Promedio: {overall_mean:.1f}')
            axes[i].legend(fontsize=8)

        # Ocultar subplots vacíos
        for i in range(len(teachers_with_data), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('Boxplots de Calificaciones por Docente y Asignatura', fontsize=14, y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.93])  # Dejar espacio para el título
        plt.savefig(f"{output_dir}/boxplots_docentes_por_asignatura.png", dpi=300, bbox_inches='tight')
        plt.close()


    def create_general_boxplot_by_subject(self, output_dir: str):
        """Crear boxplot general mostrando todas las asignaturas en el eje X."""
        self.logger.info("Creando boxplot general por asignatura...")

        # Crear figura
        plt.figure(figsize=(15, 8))

        # Obtener asignaturas únicas para crear paleta de colores
        unique_subjects = sorted(self.df_merged['id_asignatura'].unique())
        n_subjects = len(unique_subjects)

        colors = self.get_beautiful_palette(n_subjects, 'tab20b')

        # Crear boxplot por asignatura con colores diferentes
        sns.boxplot(data=self.df_merged, x='id_asignatura', y='nota_final', palette=colors)
        plt.title('Distribución de Calificaciones por Asignatura', fontsize=16)
        plt.xlabel('ID Asignatura', fontsize=12)
        plt.ylabel('Calificaciones', fontsize=12)
        plt.xticks(rotation=45)

        # Agregar línea de promedio general
        overall_mean = self.df_merged['nota_final'].mean()
        plt.axhline(y=overall_mean, color='red', linestyle='--', alpha=0.7, label=f'Promedio General: {overall_mean:.1f}')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/boxplot_general_por_asignatura.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_teacher_grade_boxplots(self, output_dir: str):
        """Crear boxplots por docente mostrando calificaciones por grado."""
        self.logger.info("Creando boxplots por docente por grado...")

        # Obtener todos los docentes con datos
        teacher_counts = self.df_merged.groupby('id_docente').size()
        teachers_with_data = teacher_counts[teacher_counts >= 3].index.tolist()  # Al menos 3 calificaciones

        if len(teachers_with_data) == 0:
            self.logger.warning("No hay docentes con suficientes datos")
            return

        # Calcular número de filas y columnas dinámicamente
        n_teachers = len(teachers_with_data)
        n_cols = 4  # 4 columnas fijas
        n_rows = (n_teachers + n_cols - 1) // n_cols  # Redondear hacia arriba

        # Crear figura con subplots dinámicos
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows + 2))  # +2 para espacio del título
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for i, teacher_id in enumerate(teachers_with_data):
            if i >= len(axes):
                break

            # Datos del docente
            df_teacher = self.df_merged[self.df_merged['id_docente'] == teacher_id]

            # Crear boxplot por grado
            unique_grades_teacher = sorted(df_teacher['id_grado'].unique())
            n_grades_teacher = len(unique_grades_teacher)
            colors_teacher = self.get_beautiful_palette(n_grades_teacher, 'tab20b')

            sns.boxplot(data=df_teacher, x='id_grado', y='nota_final', palette=colors_teacher, ax=axes[i])
            axes[i].set_title(f'Docente ID: {teacher_id}\n({df_teacher["id_grado"].nunique()} grado{"s" if df_teacher["id_grado"].nunique() > 1 else ""})')
            axes[i].set_xlabel('ID Grado')
            axes[i].set_ylabel('Calificaciones')

            # Agregar promedio general del docente
            overall_mean = df_teacher['nota_final'].mean()
            axes[i].axhline(y=overall_mean, color='red', linestyle='--', alpha=0.7, label=f'Promedio: {overall_mean:.1f}')
            axes[i].legend(fontsize=8)

        # Ocultar subplots vacíos
        for i in range(len(teachers_with_data), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('Boxplots de Calificaciones por Docente y Grado', fontsize=14, y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.93])  # Dejar espacio para el título
        plt.savefig(f"{output_dir}/boxplots_docentes_por_grado.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_teacher_subject_grade_boxplots(self, output_dir: str):
        """Crear boxplots por combinación docente-asignatura mostrando calificaciones por grado."""
        self.logger.info("Creando boxplots por docente-asignatura por grado...")

        # Obtener todas las combinaciones docente-asignatura con datos suficientes
        teacher_subject_counts = self.df_merged.groupby(['id_docente', 'id_asignatura']).size()
        combinations_with_data = teacher_subject_counts[teacher_subject_counts >= 3].index.tolist()  # Al menos 3 calificaciones

        if len(combinations_with_data) == 0:
            self.logger.warning("No hay combinaciones docente-asignatura con suficientes datos")
            return

        # Calcular número de filas y columnas dinámicamente
        n_combinations = len(combinations_with_data)
        n_cols = 4  # 4 columnas fijas
        n_rows = (n_combinations + n_cols - 1) // n_cols  # Redondear hacia arriba

        # Crear figura con subplots dinámicos
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows + 2))  # +2 para espacio del título
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for i, (teacher_id, subject_id) in enumerate(combinations_with_data):
            if i >= len(axes):
                break

            # Datos de la combinación docente-asignatura
            df_combination = self.df_merged[
                (self.df_merged['id_docente'] == teacher_id) & 
                (self.df_merged['id_asignatura'] == subject_id)
            ]

            # Crear boxplot por grado
            unique_grades = sorted(df_combination['id_grado'].unique())
            n_grades = len(unique_grades)
            colors_combination = self.get_beautiful_palette(n_grades, 'tab20b')

            sns.boxplot(data=df_combination, x='id_grado', y='nota_final', palette=colors_combination, ax=axes[i])
            axes[i].set_title(f'Docente {teacher_id} - Asignatura {subject_id}\n({n_grades} grado{"s" if n_grades > 1 else ""})')
            axes[i].set_xlabel('ID Grado')
            axes[i].set_ylabel('Calificaciones')

            # Agregar promedio general de la combinación
            overall_mean = df_combination['nota_final'].mean()
            axes[i].axhline(y=overall_mean, color='red', linestyle='--', alpha=0.7, label=f'Promedio: {overall_mean:.1f}')
            axes[i].legend(fontsize=8)

        # Ocultar subplots vacíos
        for i in range(len(combinations_with_data), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('Boxplots de Calificaciones por Docente-Asignatura y Grado', fontsize=14, y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.93])  # Dejar espacio para el título
        plt.savefig(f"{output_dir}/boxplots_docentes_asignatura_grado.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_visualizations(self, output_dir: str):
        """Crear todas las visualizaciones."""
        self.logger.info("Creando visualizaciones...")

        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)

        # Crear boxplots simples
        self.create_teacher_subject_boxplots(output_dir)

        # Crear un boxplot general por asignatura
        self.create_general_boxplot_by_subject(output_dir)

        # Crear boxplots por docente por grado
        self.create_teacher_grade_boxplots(output_dir)

        # Crear boxplots por docente, asignatura y grado
        self.create_teacher_subject_grade_boxplots(output_dir)


    def run_analysis(self):
        """Ejecutar análisis completo."""
        self.logger.info("Iniciando análisis de comportamiento docente...")
        self.logger.info(f"Resultados se guardarán en: {self.results_path}")

        # Crear directorio de resultados si no existe
        self.create_results_directory()

        # Cargar y preparar datos
        df = self.load_and_prepare_data()

        # Crear visualizaciones
        self.create_visualizations(self.results_path)


def main():
    """Función principal."""
    import argparse

    parser = argparse.ArgumentParser(description='Análisis de comportamiento de docentes en calificaciones')
    parser.add_argument('--dataset', '-d', type=str, required=True,
                       help='Ruta al archivo CSV del dataset (requerido)')
    parser.add_argument('--results', '-r', type=str, required=True,
                       help='Nombre del folder para guardar resultados (requerido, se creará en reports/)')

    args = parser.parse_args()

    # Crear y ejecutar analizador siguiendo el patrón estándar
    analyzer = TeacherBehaviorAnalysis(dataset_path=args.dataset, results_folder=args.results)

    try:
        analyzer.run_analysis()
        analyzer.logger.info("✅ Análisis completado exitosamente")
    except FileNotFoundError as e:
        analyzer.logger.error(f"❌ Error: {e}")
        raise
    except ValueError as e:
        analyzer.logger.error(f"❌ Error de validación: {e}")
        raise
    except Exception as e:
        analyzer.logger.error(f"❌ Error inesperado: {e}")
        raise
    finally:
        analyzer.close()

if __name__ == "__main__":
    main()