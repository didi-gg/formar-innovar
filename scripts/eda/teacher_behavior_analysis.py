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

    def _create_teacher_subject_boxplot_common(self, df_data, teacher_id, ax, is_sede=False, sede=None):
        """Función común para crear boxplot por docente y asignatura."""
        # Datos del docente
        df_teacher = df_data[df_data['id_docente'] == teacher_id]

        # Crear boxplot por asignatura
        unique_subjects_teacher = sorted(df_teacher['id_asignatura'].unique())
        n_subjects_teacher = len(unique_subjects_teacher)
        colors_teacher = self.get_beautiful_palette(n_subjects_teacher, 'tab20b')

        # Preparar etiquetas de asignaturas
        if 'nombre_asignatura' in df_teacher.columns:
            subject_labels = df_teacher.groupby('id_asignatura')['nombre_asignatura'].first()
            # Truncar nombres largos para mejor visualización
            subject_labels = [name[:12] + '...' if len(name) > 12 else name for name in subject_labels]
        else:
            subject_labels = [f'Asig {idx}' for idx in sorted(df_teacher['id_asignatura'].unique())]

        sns.boxplot(data=df_teacher, x='id_asignatura', y='nota_final', palette=colors_teacher, ax=ax)
        teacher_name = df_teacher['nombre'].iloc[0] if 'nombre' in df_teacher.columns else f'Docente {teacher_id}'

        # Título con información de sede si aplica
        title_suffix = f' - {sede}' if is_sede else ''
        ax.set_title(f'{teacher_name}\n({df_teacher["id_asignatura"].nunique()} asignatura{"s" if df_teacher["id_asignatura"].nunique() > 1 else ""}){title_suffix}')
        ax.set_xlabel('Asignatura')
        ax.set_ylabel('Calificaciones')
        ax.set_xticklabels(subject_labels, rotation=45, ha='right')

        # Agregar promedio general del docente
        overall_mean = df_teacher['nota_final'].mean()
        ax.axhline(y=overall_mean, color='red', linestyle='--', alpha=0.7, label=f'Promedio: {overall_mean:.1f}')
        ax.legend(fontsize=8)

    def _create_teacher_grade_boxplot_common(self, df_data, teacher_id, ax, is_sede=False, sede=None):
        """Función común para crear boxplot por docente y grado."""
        # Datos del docente
        df_teacher = df_data[df_data['id_docente'] == teacher_id]

        # Crear boxplot por grado
        unique_grades_teacher = sorted(df_teacher['id_grado'].unique())
        n_grades_teacher = len(unique_grades_teacher)
        colors_teacher = self.get_beautiful_palette(n_grades_teacher, 'tab20b')

        sns.boxplot(data=df_teacher, x='id_grado', y='nota_final', palette=colors_teacher, ax=ax)
        teacher_name = df_teacher['nombre'].iloc[0] if 'nombre' in df_teacher.columns else f'Docente {teacher_id}'

        # Título con información de sede si aplica
        title_suffix = f' - {sede}' if is_sede else ''
        ax.set_title(f'{teacher_name}\n({df_teacher["id_grado"].nunique()} grado{"s" if df_teacher["id_grado"].nunique() > 1 else ""}){title_suffix}')
        ax.set_xlabel('ID Grado')
        ax.set_ylabel('Calificaciones')

        # Agregar promedio general del docente
        overall_mean = df_teacher['nota_final'].mean()
        ax.axhline(y=overall_mean, color='red', linestyle='--', alpha=0.7, label=f'Promedio: {overall_mean:.1f}')
        ax.legend(fontsize=8)

    def _create_teacher_subject_grade_boxplot_common(self, df_data, teacher_id, subject_id, ax, is_sede=False, sede=None):
        """Función común para crear boxplot por docente-asignatura-grado."""
        # Datos de la combinación docente-asignatura
        df_combination = df_data[
            (df_data['id_docente'] == teacher_id) & 
            (df_data['id_asignatura'] == subject_id)
        ]

        # Crear boxplot por grado
        unique_grades = sorted(df_combination['id_grado'].unique())
        n_grades = len(unique_grades)
        colors_combination = self.get_beautiful_palette(n_grades, 'tab20b')

        # Debug: Log información sobre los datos
        log_suffix = f" (Sede {sede})" if is_sede else ""
        for grado in sorted(df_combination['id_grado'].unique()):
            df_grado = df_combination[df_combination['id_grado'] == grado]

        # Crear boxplot con todos los grados, sin filtrar
        sns.boxplot(data=df_combination, x='id_grado', y='nota_final', 
                   palette=colors_combination, ax=ax,
                   showfliers=True,  # Mostrar outliers
                   whis=1.5)  # Rango de whiskers

        teacher_name = df_combination['nombre'].iloc[0] if 'nombre' in df_combination.columns else f'Docente {teacher_id}'
        subject_name = df_combination['nombre_asignatura'].iloc[0] if 'nombre_asignatura' in df_combination.columns else f'Asignatura {subject_id}'
        # Truncar nombres para mejor visualización
        teacher_name_short = teacher_name[:15] + '...' if len(teacher_name) > 15 else teacher_name
        subject_name_short = subject_name[:20] + '...' if len(subject_name) > 20 else subject_name

        # Título con información de sede si aplica
        title_suffix = f' - {sede}' if is_sede else ''
        ax.set_title(f'{teacher_name_short}\n{subject_name_short}{title_suffix}')
        ax.set_xlabel('ID Grado')
        ax.set_ylabel('Calificaciones')

        # Agregar promedio general de la combinación
        overall_mean = df_combination['nota_final'].mean()
        ax.axhline(y=overall_mean, color='red', linestyle='--', alpha=0.7, label=f'Promedio: {overall_mean:.1f}')
        ax.legend(fontsize=8)

    def _create_general_boxplot_by_subject_common(self, df_data, output_path, is_sede=False, sede=None):
        """Función común para crear boxplot general por asignatura."""
        # Crear figura
        plt.figure(figsize=(15, 8))

        # Obtener asignaturas únicas para crear paleta de colores
        unique_subjects = sorted(df_data['id_asignatura'].unique())
        n_subjects = len(unique_subjects)

        colors = self.get_beautiful_palette(n_subjects, 'tab20b')

        # Preparar etiquetas de asignaturas
        if 'nombre_asignatura' in df_data.columns:
            subject_labels = df_data.groupby('id_asignatura')['nombre_asignatura'].first()
            # Truncar nombres largos para mejor visualización
            subject_labels = [name[:15] + '...' if len(name) > 15 else name for name in subject_labels]
        else:
            subject_labels = [f'Asig {idx}' for idx in unique_subjects]

        # Crear boxplot por asignatura con colores diferentes
        sns.boxplot(data=df_data, x='id_asignatura', y='nota_final', palette=colors)

        # Título con información de sede si aplica
        title_suffix = f' - {sede}' if is_sede else ''
        plt.title(f'Distribución de Calificaciones por Asignatura{title_suffix}', fontsize=16)
        plt.xlabel('Asignatura', fontsize=12)
        plt.ylabel('Calificaciones', fontsize=12)
        plt.xticks(range(len(unique_subjects)), subject_labels, rotation=45, ha='right')

        # Agregar línea de promedio general
        overall_mean = df_data['nota_final'].mean()
        plt.axhline(y=overall_mean, color='red', linestyle='--', alpha=0.7, label=f'Promedio General: {overall_mean:.1f}')
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def load_and_prepare_data(self) -> pd.DataFrame:
        # Cargar el dataset principal que se pasa como parámetro
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"No se encontró el dataset en: {self.dataset_path}")

        df = pd.read_csv(self.dataset_path)

        # Verificar que tenga las columnas necesarias
        required_columns = ['id_asignatura', 'nota_final', 'id_docente', 'id_grado', 'sede']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Columnas requeridas faltantes en el dataset: {missing_columns}")

        # Cargar información de docentes
        docentes_path = os.path.join(os.path.dirname(self.dataset_path), "..", "raw", "tablas_maestras", "docentes.csv")
        if not os.path.exists(docentes_path):
            raise FileNotFoundError(f"No se encontró el archivo de docentes en: {docentes_path}")

        df_docentes = pd.read_csv(docentes_path)

        # Verificar que tenga las columnas necesarias
        required_docente_columns = ['id_docente', 'primer_nombre', 'primer_apellido']
        missing_docente_columns = [col for col in required_docente_columns if col not in df_docentes.columns]
        if missing_docente_columns:
            raise ValueError(f"Columnas requeridas faltantes en docentes.csv: {missing_docente_columns}")

        # Crear nombre concatenando primer_nombre + primer_apellido
        df_docentes['nombre'] = df_docentes['primer_nombre'] + ' ' + df_docentes['primer_apellido']
        # Hacer join con la información de docentes
        df = df.merge(df_docentes[['id_docente', 'nombre']], on='id_docente', how='left')

        # Cargar información de asignaturas
        asignaturas_path = os.path.join(os.path.dirname(self.dataset_path), "..", "raw", "tablas_maestras", "asignaturas.csv")
        if not os.path.exists(asignaturas_path):
            raise FileNotFoundError(f"No se encontró el archivo de asignaturas en: {asignaturas_path}")

        df_asignaturas = pd.read_csv(asignaturas_path)

        # Verificar que tenga las columnas necesarias
        required_asignatura_columns = ['id_asignatura', 'nombre']
        missing_asignatura_columns = [col for col in required_asignatura_columns if col not in df_asignaturas.columns]
        if missing_asignatura_columns:
            raise ValueError(f"Columnas requeridas faltantes en asignaturas.csv: {missing_asignatura_columns}")

        # Hacer join con la información de asignaturas
        df = df.merge(df_asignaturas[['id_asignatura', 'nombre']], on='id_asignatura', how='left', suffixes=('_docente', '_asignatura'))
        # Renombrar las columnas para mantener consistencia
        df = df.rename(columns={'nombre_docente': 'nombre', 'nombre_asignatura': 'nombre_asignatura'})

        self.df_merged = df
        return df

    def create_teacher_subject_boxplots(self, df_data: pd.DataFrame, output_dir: str, sede: str = None):
        """Crear boxplots por docente y asignatura."""
        sede_suffix = f" - {sede}" if sede else ""
        sede_file_suffix = f"_{sede.lower()}" if sede else ""
        self.logger.info(f"Creando boxplots simples por docente{sede_suffix}...")

        # Obtener todos los docentes con datos
        teacher_counts = df_data.groupby('id_docente').size()
        teachers_with_data = teacher_counts[teacher_counts >= 3].index.tolist()  # Al menos 3 calificaciones

        if len(teachers_with_data) == 0:
            self.logger.warning(f"No hay docentes con suficientes datos{sede_suffix}")
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
            self._create_teacher_subject_boxplot_common(df_data, teacher_id, axes[i], is_sede=(sede is not None), sede=sede)

        # Ocultar subplots vacíos
        for i in range(len(teachers_with_data), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f'Boxplots de Calificaciones por Docente y Asignatura{sede_suffix}', fontsize=14, y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.93])  # Dejar espacio para el título
        plt.savefig(f"{output_dir}/boxplots_docentes_por_asignatura{sede_file_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()


    def create_general_boxplot_by_subject(self, df_data: pd.DataFrame, output_dir: str, sede: str = None):
        """Crear boxplot general mostrando todas las asignaturas en el eje X."""
        sede_suffix = f" - {sede}" if sede else ""
        sede_file_suffix = f"_{sede.lower()}" if sede else ""
        self.logger.info(f"Creando boxplot general por asignatura{sede_suffix}...")
        output_path = f"{output_dir}/boxplot_general_por_asignatura{sede_file_suffix}.png"
        self._create_general_boxplot_by_subject_common(df_data, output_path, is_sede=(sede is not None), sede=sede)

    def create_teacher_grade_boxplots(self, df_data: pd.DataFrame, output_dir: str, sede: str = None):
        """Crear boxplots por docente mostrando calificaciones por grado."""
        sede_suffix = f" - {sede}" if sede else ""
        sede_file_suffix = f"_{sede.lower()}" if sede else ""
        self.logger.info(f"Creando boxplots por docente por grado{sede_suffix}...")

        # Obtener todos los docentes con datos
        teacher_counts = df_data.groupby('id_docente').size()
        teachers_with_data = teacher_counts[teacher_counts >= 3].index.tolist()  # Al menos 3 calificaciones

        if len(teachers_with_data) == 0:
            self.logger.warning(f"No hay docentes con suficientes datos{sede_suffix}")
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
            self._create_teacher_grade_boxplot_common(df_data, teacher_id, axes[i], is_sede=(sede is not None), sede=sede)

        # Ocultar subplots vacíos
        for i in range(len(teachers_with_data), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f'Boxplots de Calificaciones por Docente y Grado{sede_suffix}', fontsize=14, y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.93])  # Dejar espacio para el título
        plt.savefig(f"{output_dir}/boxplots_docentes_por_grado{sede_file_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_teacher_subject_grade_boxplots(self, df_data: pd.DataFrame, output_dir: str, sede: str = None):
        """Crear boxplots por combinación docente-asignatura mostrando calificaciones por grado."""
        sede_suffix = f" - {sede}" if sede else ""
        sede_file_suffix = f"_{sede.lower()}" if sede else ""
        self.logger.info(f"Creando boxplots por docente-asignatura por grado{sede_suffix}...")

        # Obtener todas las combinaciones docente-asignatura con datos suficientes
        teacher_subject_counts = df_data.groupby(['id_docente', 'id_asignatura']).size()
        combinations_with_data = teacher_subject_counts[teacher_subject_counts >= 3].index.tolist()  # Al menos 3 calificaciones

        if len(combinations_with_data) == 0:
            self.logger.warning(f"No hay combinaciones docente-asignatura con suficientes datos{sede_suffix}")
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
            self._create_teacher_subject_grade_boxplot_common(df_data, teacher_id, subject_id, axes[i], is_sede=(sede is not None), sede=sede)

        # Ocultar subplots vacíos
        for i in range(len(combinations_with_data), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f'Boxplots de Calificaciones por Docente-Asignatura y Grado{sede_suffix}', fontsize=14, y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.93])  # Dejar espacio para el título
        plt.savefig(f"{output_dir}/boxplots_docentes_asignatura_grado{sede_file_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_teacher_grading_patterns(self, df_data: pd.DataFrame, output_dir: str, sede: str = None):
        """Analizar patrones de calificación de docentes para identificar los que no colocan notas altas."""
        sede_suffix = f" - {sede}" if sede else ""
        sede_file_suffix = f"_{sede.lower()}" if sede else ""
        self.logger.info(f"Analizando patrones de calificación de docentes{sede_suffix}...")

        # Calcular estadísticas por docente
        teacher_stats = df_data.groupby('id_docente')['nota_final'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max',
            lambda x: (x >= 80).sum(),  # Notas altas (>=80)
            lambda x: (x >= 90).sum(),  # Notas muy altas (>=90)
            lambda x: (x < 60).sum()    # Notas bajas (<60)
        ]).round(2)

        # Renombrar columnas
        teacher_stats.columns = ['total_calificaciones', 'promedio', 'mediana', 'desviacion_std', 
                               'nota_minima', 'nota_maxima', 'notas_altas_80', 'notas_muy_altas_90', 'notas_bajas_60']

        # Agregar nombres de docentes y sede
        teacher_info = df_data.groupby('id_docente')[['nombre', 'sede']].first()
        teacher_stats = teacher_stats.merge(teacher_info, left_index=True, right_index=True, how='left')

        # Calcular porcentajes
        teacher_stats['porcentaje_notas_altas'] = (teacher_stats['notas_altas_80'] / teacher_stats['total_calificaciones'] * 100).round(2)
        teacher_stats['porcentaje_notas_muy_altas'] = (teacher_stats['notas_muy_altas_90'] / teacher_stats['total_calificaciones'] * 100).round(2)
        teacher_stats['porcentaje_notas_bajas'] = (teacher_stats['notas_bajas_60'] / teacher_stats['total_calificaciones'] * 100).round(2)

        # Identificar docentes con patrones de calificación bajos
        # Criterios: promedio < 75, porcentaje notas altas < 30%, o nota máxima < 95
        teacher_stats['patron_bajo'] = (
            (teacher_stats['promedio'] < 75) | 
            (teacher_stats['porcentaje_notas_altas'] < 30) | 
            (teacher_stats['nota_maxima'] < 95)
        )

        # Ordenar por sede y luego por promedio para agrupar por sede (si hay múltiples sedes)
        if sede is None:
            teacher_stats_sorted = teacher_stats.sort_values(['sede', 'promedio'])
        else:
            teacher_stats_sorted = teacher_stats.sort_values('promedio')

        # Crear visualización de patrones de calificación
        fig, axes = plt.subplots(2, 4, figsize=(32, 12))

        # Preparar etiquetas de docentes con información de sede
        teacher_labels = []
        bar_positions = []
        bar_values = []
        bar_colors = []
        current_sede = None
        position = 0

        for idx, (_, row) in enumerate(teacher_stats_sorted.iterrows()):
            if sede is None and current_sede != row['sede']:
                # Agregar separador de sede solo si es análisis general
                teacher_labels.append(f"--- {row['sede']} ---")
                bar_positions.append(position)
                bar_values.append(0)  # Valor 0 para el separador
                bar_colors.append('lightgray')  # Color gris para separadores
                position += 1
                current_sede = row['sede']
            # Agregar nombre del docente
            name = row['nombre'][:15] + '...' if len(row['nombre']) > 15 else row['nombre']
            teacher_labels.append(name)
            bar_positions.append(position)
            bar_values.append(row['promedio'])
            bar_colors.append('red' if row['patron_bajo'] else 'skyblue')
            position += 1

        # 1. Distribución de promedios por docente
        bars1 = axes[0, 0].bar(bar_positions, bar_values, color=bar_colors)
        axes[0, 0].set_title('Promedio de Calificaciones por Docente', fontsize=12)
        axes[0, 0].set_xlabel('Docentes (ordenados por promedio)')
        axes[0, 0].set_ylabel('Promedio de Calificaciones')
        axes[0, 0].set_xticks(range(len(teacher_labels)))
        axes[0, 0].set_xticklabels(teacher_labels, rotation=45, ha='right')
        axes[0, 0].axhline(y=75, color='orange', linestyle='--', alpha=0.7, label='Umbral 75')
        axes[0, 0].legend()

        # 2. Porcentaje de notas altas por docente
        bar_values_2 = []
        bar_colors_2 = []
        current_sede_2 = None
        for idx, (_, row) in enumerate(teacher_stats_sorted.iterrows()):
            if sede is None and current_sede_2 != row['sede']:
                bar_values_2.append(0)
                bar_colors_2.append('lightgray')
                current_sede_2 = row['sede']
            bar_values_2.append(row['porcentaje_notas_altas'])
            bar_colors_2.append('red' if row['patron_bajo'] else 'lightgreen')

        bars2 = axes[0, 1].bar(bar_positions, bar_values_2, color=bar_colors_2)
        axes[0, 1].set_title('Porcentaje de Notas Altas (≥80) por Docente', fontsize=12)
        axes[0, 1].set_xlabel('Docentes (agrupados por sede)')
        axes[0, 1].set_ylabel('Porcentaje de Notas Altas')
        axes[0, 1].set_xticks(range(len(teacher_labels)))
        axes[0, 1].set_xticklabels(teacher_labels, rotation=45, ha='right')
        axes[0, 1].axhline(y=30, color='orange', linestyle='--', alpha=0.7, label='Umbral 30%')
        axes[0, 1].legend()

        # 3. Nota máxima por docente
        bar_values_3 = []
        bar_colors_3 = []
        current_sede_3 = None
        for idx, (_, row) in enumerate(teacher_stats_sorted.iterrows()):
            if sede is None and current_sede_3 != row['sede']:
                bar_values_3.append(0)
                bar_colors_3.append('lightgray')
                current_sede_3 = row['sede']
            bar_values_3.append(row['nota_maxima'])
            bar_colors_3.append('red' if row['patron_bajo'] else 'gold')

        bars3 = axes[0, 2].bar(bar_positions, bar_values_3, color=bar_colors_3)
        axes[0, 2].set_title('Nota Máxima por Docente', fontsize=12)
        axes[0, 2].set_xlabel('Docentes (agrupados por sede)')
        axes[0, 2].set_ylabel('Nota Máxima')
        axes[0, 2].set_xticks(range(len(teacher_labels)))
        axes[0, 2].set_xticklabels(teacher_labels, rotation=45, ha='right')
        axes[0, 2].axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='Umbral 95')
        axes[0, 2].legend()

        # 4. Nota mínima por docente
        bar_values_4 = []
        bar_colors_4 = []
        current_sede_4 = None
        for idx, (_, row) in enumerate(teacher_stats_sorted.iterrows()):
            if sede is None and current_sede_4 != row['sede']:
                bar_values_4.append(0)
                bar_colors_4.append('lightgray')
                current_sede_4 = row['sede']
            bar_values_4.append(row['nota_minima'])
            bar_colors_4.append('red' if row['patron_bajo'] else 'lightcoral')

        bars4 = axes[1, 0].bar(bar_positions, bar_values_4, color=bar_colors_4)
        axes[1, 0].set_title('Nota Mínima por Docente', fontsize=12)
        axes[1, 0].set_xlabel('Docentes (agrupados por sede)')
        axes[1, 0].set_ylabel('Nota Mínima')
        axes[1, 0].set_xticks(range(len(teacher_labels)))
        axes[1, 0].set_xticklabels(teacher_labels, rotation=45, ha='right')
        axes[1, 0].axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='Umbral 60')
        axes[1, 0].legend()

        # 5. Mediana por docente
        bar_values_5 = []
        bar_colors_5 = []
        current_sede_5 = None
        for idx, (_, row) in enumerate(teacher_stats_sorted.iterrows()):
            if sede is None and current_sede_5 != row['sede']:
                bar_values_5.append(0)
                bar_colors_5.append('lightgray')
                current_sede_5 = row['sede']
            bar_values_5.append(row['mediana'])
            bar_colors_5.append('red' if row['patron_bajo'] else 'lightsteelblue')

        bars5 = axes[1, 1].bar(bar_positions, bar_values_5, color=bar_colors_5)
        axes[1, 1].set_title('Mediana de Calificaciones por Docente', fontsize=12)
        axes[1, 1].set_xlabel('Docentes (agrupados por sede)')
        axes[1, 1].set_ylabel('Mediana de Calificaciones')
        axes[1, 1].set_xticks(range(len(teacher_labels)))
        axes[1, 1].set_xticklabels(teacher_labels, rotation=45, ha='right')
        axes[1, 1].axhline(y=75, color='orange', linestyle='--', alpha=0.7, label='Umbral 75')
        axes[1, 1].legend()

        # 6. Porcentaje de notas bajas por docente
        bar_values_6 = []
        bar_colors_6 = []
        current_sede_6 = None
        for idx, (_, row) in enumerate(teacher_stats_sorted.iterrows()):
            if sede is None and current_sede_6 != row['sede']:
                bar_values_6.append(0)
                bar_colors_6.append('lightgray')
                current_sede_6 = row['sede']
            bar_values_6.append(row['porcentaje_notas_bajas'])
            bar_colors_6.append('red' if row['patron_bajo'] else 'orange')

        bars6 = axes[1, 2].bar(bar_positions, bar_values_6, color=bar_colors_6)
        axes[1, 2].set_title('Porcentaje de Notas Bajas (<60) por Docente', fontsize=12)
        axes[1, 2].set_xlabel('Docentes (agrupados por sede)')
        axes[1, 2].set_ylabel('Porcentaje de Notas Bajas')
        axes[1, 2].set_xticks(range(len(teacher_labels)))
        axes[1, 2].set_xticklabels(teacher_labels, rotation=45, ha='right')
        axes[1, 2].axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Umbral 20%')
        axes[1, 2].legend()

        # 7. Desviación estándar por docente
        bar_values_7 = []
        bar_colors_7 = []
        current_sede_7 = None
        for idx, (_, row) in enumerate(teacher_stats_sorted.iterrows()):
            if sede is None and current_sede_7 != row['sede']:
                bar_values_7.append(0)
                bar_colors_7.append('lightgray')
                current_sede_7 = row['sede']
            bar_values_7.append(row['desviacion_std'])
            # Colores basados en la variabilidad: baja variabilidad (consistente) = verde, alta variabilidad = rojo
            if row['desviacion_std'] < 10:
                bar_colors_7.append('green')  # Muy consistente
            elif row['desviacion_std'] < 15:
                bar_colors_7.append('lightgreen')  # Consistente
            elif row['desviacion_std'] < 20:
                bar_colors_7.append('orange')  # Moderadamente variable
            else:
                bar_colors_7.append('red')  # Muy variable

        bars7 = axes[1, 3].bar(bar_positions, bar_values_7, color=bar_colors_7)
        axes[1, 3].set_title('Desviación Estándar por Docente', fontsize=12)
        axes[1, 3].set_xlabel('Docentes (agrupados por sede)')
        axes[1, 3].set_ylabel('Desviación Estándar')
        axes[1, 3].set_xticks(range(len(teacher_labels)))
        axes[1, 3].set_xticklabels(teacher_labels, rotation=45, ha='right')
        axes[1, 3].axhline(y=15, color='red', linestyle='--', alpha=0.7, label='Umbral 15')
        axes[1, 3].legend()

        # 8. Distribución de promedios (histograma)
        axes[0, 3].hist(teacher_stats['promedio'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 3].axvline(x=75, color='red', linestyle='--', alpha=0.7, label='Umbral 75')
        axes[0, 3].set_title('Distribución de Promedios de Docentes', fontsize=12)
        axes[0, 3].set_xlabel('Promedio de Calificaciones')
        axes[0, 3].set_ylabel('Frecuencia')
        axes[0, 3].legend()

        plt.suptitle(f'Análisis de Patrones de Calificación de Docentes{sede_suffix}', fontsize=16, y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(f"{output_dir}/analisis_patrones_calificacion_docentes{sede_file_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Identificar docentes con patrones problemáticos
        docentes_problematicos = teacher_stats[teacher_stats['patron_bajo']].copy()
        docentes_problematicos = docentes_problematicos.sort_values('promedio')

        # Crear tabla resumen
        if len(docentes_problematicos) > 0:
            self.logger.warning(f"Se identificaron {len(docentes_problematicos)} docentes con patrones de calificación bajos{sede_suffix}")

            # Crear visualización específica de docentes problemáticos
            fig, ax = plt.subplots(figsize=(12, 8))

            # Crear tabla visual
            table_data = []
            for idx, (teacher_id, row) in enumerate(docentes_problematicos.iterrows()):
                teacher_name = row.get('nombre', f'Docente {teacher_id}')
                table_data.append([
                    teacher_name,
                    f"{row['promedio']:.1f}",
                    f"{row['porcentaje_notas_altas']:.1f}%",
                    f"{row['nota_maxima']:.1f}",
                    f"{row['total_calificaciones']}"
                ])

            table = ax.table(cellText=table_data,
                           colLabels=['Docente', 'Promedio', '% Notas Altas', 'Nota Máx', 'Total Calif'],
                           cellLoc='center',
                           loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)

            ax.set_title(f'Docentes con Patrones de Calificación Bajos{sede_suffix}', fontsize=14, pad=20)
            ax.axis('off')

            plt.tight_layout()
            plt.savefig(f"{output_dir}/tabla_docentes_patrones_bajos{sede_file_suffix}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            self.logger.info(f"No se identificaron docentes con patrones de calificación problemáticos{sede_suffix}")

        return teacher_stats


    def create_visualizations(self, output_dir: str):
        """Crear todas las visualizaciones."""
        self.logger.info("Creando visualizaciones...")

        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)

        # Obtener sedes únicas
        sedes_unicas = sorted(self.df_merged['sede'].unique())
        self.logger.info(f"Creando visualizaciones para {len(sedes_unicas)} sedes: {sedes_unicas}")

        # Crear visualizaciones para cada sede
        for sede in sedes_unicas:
            self.logger.info(f"Procesando sede: {sede}")

            # Filtrar datos para esta sede
            df_sede = self.df_merged[self.df_merged['sede'] == sede].copy()

            if len(df_sede) == 0:
                self.logger.warning(f"No hay datos para la sede {sede}")
                continue

            # Crear subdirectorio para esta sede
            sede_dir = os.path.join(output_dir, sede.lower())
            os.makedirs(sede_dir, exist_ok=True)

            # Crear visualizaciones para esta sede usando las funciones unificadas
            self.create_teacher_subject_boxplots(df_sede, sede_dir, sede)
            self.create_general_boxplot_by_subject(df_sede, sede_dir, sede)
            self.create_teacher_grade_boxplots(df_sede, sede_dir, sede)
            self.create_teacher_subject_grade_boxplots(df_sede, sede_dir, sede)
            self.analyze_teacher_grading_patterns(df_sede, sede_dir, sede)

        # Crear visualizaciones generales (todas las sedes juntas)
        self.logger.info("Creando visualizaciones generales...")
        self.create_teacher_subject_boxplots(self.df_merged, output_dir)
        self.create_general_boxplot_by_subject(self.df_merged, output_dir)
        self.create_teacher_grade_boxplots(self.df_merged, output_dir)
        self.create_teacher_subject_grade_boxplots(self.df_merged, output_dir)
        self.analyze_teacher_grading_patterns(self.df_merged, output_dir)


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