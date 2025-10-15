"""
Script para análisis de cursos en Moodle
Analiza la composición y características de los cursos con mapas de calor curso-asignatura
"""

import os
import sys
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Agregar el directorio padre al path para importar utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.eda_analysis_base import EDAAnalysisBase

class CourseAnalysis(EDAAnalysisBase):
    """
    Clase para análisis de cursos en Moodle.
    Genera mapas de calor y análisis detallados de la composición de cursos.
    """

    def _initialize_analysis_attributes(self):
        """Inicializar atributos específicos del análisis de cursos."""
        self.df_courses = None
        self.df_courses_base = None
        self.df_merged = None
        self.df_modules_active = None
        self.df_modules_featured = None
        self.df_asignaturas = None  # Cache para nombres de asignaturas
        self.subject_filter = [1, 2, 3, 4]  # Asignaturas a analizar
    
    # ==================== MÉTODOS HELPER REUTILIZABLES ====================
    
    def _filter_by_subjects(self, df: pd.DataFrame, subjects: list = None) -> pd.DataFrame:
        """Filtrar DataFrame por lista de asignaturas"""
        if subjects is None:
            subjects = self.subject_filter
        
        if 'id_asignatura' in df.columns and not df.empty:
            return df[df['id_asignatura'].isin(subjects)].copy()
        return df
    
    def _get_sede_suffixes(self, sede: str = None) -> tuple:
        """Obtener sufijos para título y nombre de archivo según sede"""
        if sede:
            return f" - {sede}", f"_{sede.lower()}"
        return "", ""
    
    def _load_asignaturas_names(self) -> pd.DataFrame:
        """Cargar nombres de asignaturas desde archivo maestro (con cache)"""
        if self.df_asignaturas is not None:
            return self.df_asignaturas
        
        asignaturas_path = os.path.join(
            os.path.dirname(self.dataset_path), 
            "..", "raw", "tablas_maestras", "asignaturas.csv"
        )
        
        if os.path.exists(asignaturas_path):
            self.df_asignaturas = pd.read_csv(asignaturas_path)
            self.logger.info(f"Nombres de asignaturas cargados: {len(self.df_asignaturas)} asignaturas")
            return self.df_asignaturas
        
        self.logger.warning(f"No se encontró archivo de asignaturas en: {asignaturas_path}")
        self.df_asignaturas = pd.DataFrame()
        return self.df_asignaturas
    
    def _get_asignatura_name(self, id_asignatura: int) -> str:
        """Obtener nombre de asignatura por su ID"""
        df_asig = self._load_asignaturas_names()
        
        if not df_asig.empty and 'nombre' in df_asig.columns:
            asig_row = df_asig[df_asig['id_asignatura'] == id_asignatura]
            if not asig_row.empty:
                return str(asig_row['nombre'].iloc[0]).strip()
        
        return f"Asignatura {id_asignatura}"
    
    def _create_asignatura_label(self, row: pd.Series, max_length: int = 30) -> str:
        """Crear etiqueta legible de asignatura-grado"""
        asig_name = row.get('asignatura', None)
        if pd.isna(asig_name) or not isinstance(asig_name, str):
            asig_name = self._get_asignatura_name(row['id_asignatura'])
        
        if len(asig_name) > max_length:
            asig_name = asig_name[:max_length-3] + '...'
        
        return f"{asig_name} - Grado {row['id_grado']}"
    
    def _filter_by_sede(self, df: pd.DataFrame, sede: str = None) -> pd.DataFrame:
        """Filtrar DataFrame por sede"""
        if sede and 'sede' in df.columns:
            return df[df['sede'] == sede].copy()
        return df.copy()
    
    def _calculate_text_contrast_color(self, color_rgba: tuple) -> str:
        """Calcular color de texto (blanco/negro) según luminosidad del fondo"""
        r, g, b = color_rgba[:3]
        luminosidad = 0.299 * r + 0.587 * g + 0.114 * b
        return 'white' if luminosidad < 0.5 else 'black'
    
    def _merge_asignatura_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agregar nombres de asignaturas a un DataFrame"""
        df_asig = self._load_asignaturas_names()
        
        if not df_asig.empty and 'id_asignatura' in df.columns and 'nombre' in df_asig.columns:
            df = df.merge(
                df_asig[['id_asignatura', 'nombre']], 
                on='id_asignatura', 
                how='left'
            )
            if 'nombre' in df.columns:
                df = df.rename(columns={'nombre': 'asignatura'})
        
        return df
    
    # ==================== FIN MÉTODOS HELPER ====================

    def load_and_prepare_data(self) -> pd.DataFrame:
        """Cargar y preparar los datos de cursos"""
        # Cargar el dataset principal
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"No se encontró el dataset en: {self.dataset_path}")

        df = pd.read_csv(self.dataset_path)
        self.logger.info(f"Dataset principal cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

        # Filtrar por asignaturas
        df = self._filter_by_subjects(df)
        self.logger.info(f"Dataset filtrado a asignaturas {self.subject_filter}: {df.shape[0]} filas")

        # Cargar información de cursos (courses.csv)
        courses_path = os.path.join(os.path.dirname(self.dataset_path), "..", "interim", "moodle", "courses.csv")
        if os.path.exists(courses_path):
            self.df_courses = pd.read_csv(courses_path)
            self.logger.info(f"Información de cursos cargada: {self.df_courses.shape[0]} cursos")
            
            # Filtrar y agregar nombres de asignaturas
            self.df_courses = self._filter_by_subjects(self.df_courses)
            self.df_courses = self._merge_asignatura_names(self.df_courses)
            self.logger.info(f"Cursos filtrados: {self.df_courses.shape[0]} cursos")
        else:
            self.logger.warning(f"No se encontró el archivo de cursos en: {courses_path}")
            self.df_courses = pd.DataFrame()

        # Cargar información base de cursos (courses_base.csv)
        courses_base_path = os.path.join(os.path.dirname(self.dataset_path), "..", "interim", "moodle", "courses_base.csv")
        if os.path.exists(courses_base_path):
            self.df_courses_base = pd.read_csv(courses_base_path)
            self.df_courses_base = self._filter_by_subjects(self.df_courses_base)
            self.logger.info(f"Cursos base filtrados: {self.df_courses_base.shape[0]} registros")
        else:
            self.logger.warning(f"No se encontró el archivo de cursos base en: {courses_base_path}")
            self.df_courses_base = pd.DataFrame()

        # Cargar información de módulos activos
        modules_active_path = os.path.join(os.path.dirname(self.dataset_path), "..", "interim", "moodle", "modules_active.csv")
        if os.path.exists(modules_active_path):
            self.df_modules_active = pd.read_csv(modules_active_path)
            self.df_modules_active = self._filter_by_subjects(self.df_modules_active)
            self.logger.info(f"Módulos activos filtrados: {self.df_modules_active.shape[0]} módulos")
        else:
            self.logger.warning(f"No se encontró el archivo de módulos activos en: {modules_active_path}")
            self.df_modules_active = pd.DataFrame()

        # Cargar información de módulos destacados
        modules_featured_path = os.path.join(os.path.dirname(self.dataset_path), "..", "interim", "moodle", "modules_featured.csv")
        if os.path.exists(modules_featured_path):
            self.df_modules_featured = pd.read_csv(modules_featured_path)
            self.df_modules_featured = self._filter_by_subjects(self.df_modules_featured)
            self.df_modules_featured = self._merge_asignatura_names(self.df_modules_featured)
            # Renombrar para consistencia con honeycomb charts
            if 'asignatura' in self.df_modules_featured.columns:
                self.df_modules_featured = self.df_modules_featured.rename(columns={'asignatura': 'asignatura_name'})
            self.logger.info(f"Módulos destacados filtrados: {self.df_modules_featured.shape[0]} módulos")
        else:
            self.logger.warning(f"No se encontró el archivo de módulos destacados en: {modules_featured_path}")
            self.df_modules_featured = pd.DataFrame()

        # Combinar datasets
        self.df_merged = self._merge_datasets(df)

        return self.df_merged

    def _merge_datasets(self, df_main: pd.DataFrame) -> pd.DataFrame:
        """Combinar el dataset principal con información de cursos"""
        df_result = df_main.copy()

        # Merge con courses si está disponible
        if not self.df_courses.empty:
            # Las columnas clave para el merge
            merge_cols = ['id_asignatura', 'id_grado', 'year', 'period', 'sede']

            # Verificar que existan las columnas necesarias
            missing_cols = [col for col in merge_cols if col not in df_result.columns]
            if missing_cols:
                self.logger.warning(f"Columnas faltantes para merge con courses: {missing_cols}")
            else:
                df_result = df_result.merge(
                    self.df_courses,
                    on=merge_cols,
                    how='left',
                    suffixes=('', '_course')
                )
                self.logger.info(f"Merge con courses completado: {df_result.shape}")

        # Merge con courses_base si está disponible
        if not self.df_courses_base.empty:
            merge_cols_base = ['sede', 'year', 'id_grado', 'id_asignatura', 'period']

            missing_cols_base = [col for col in merge_cols_base if col not in df_result.columns]
            if missing_cols_base:
                self.logger.warning(f"Columnas faltantes para merge con courses_base: {missing_cols_base}")
            else:
                # Seleccionar columnas de courses_base que no estén ya en df_result
                cols_to_add = [col for col in self.df_courses_base.columns if col not in df_result.columns or col in merge_cols_base]
                df_result = df_result.merge(
                    self.df_courses_base[cols_to_add],
                    on=merge_cols_base,
                    how='left',
                    suffixes=('', '_base')
                )
                self.logger.info(f"Merge con courses_base completado: {df_result.shape}")

        return df_result

    def create_course_composition_heatmap(self, output_dir: str, sede: str = None):
        """Crear mapa de calor de porcentajes de composición de cursos. Muestra la mediana por combinación ASIGNATURA - GRADO del dataset principal."""
        sede_suffix, sede_file_suffix = self._get_sede_suffixes(sede)
        self.logger.info(f"Creando mapa de calor de composición de cursos{sede_suffix}...")

        # Filtrar por sede
        df_main_data = self._filter_by_sede(self.df_merged, sede)

        if df_main_data.empty:
            self.logger.warning(f"No hay datos en el dataset principal{sede_suffix}")
            return

        # Obtener combinaciones únicas de asignatura-grado del dataset principal
        combinaciones_principales = df_main_data[['id_asignatura', 'id_grado']].drop_duplicates()
        self.logger.info(f"Combinaciones asignatura-grado en dataset principal: {len(combinaciones_principales)}")

        # Filtrar courses por sede
        df_courses_data = self._filter_by_sede(self.df_courses, sede)

        if df_courses_data.empty:
            self.logger.warning(f"No hay datos de cursos para crear mapa de calor{sede_suffix}")
            return

        # Filtrar solo las combinaciones que están en el dataset principal
        df_courses_data = df_courses_data.merge(
            combinaciones_principales,
            on=['id_asignatura', 'id_grado'],
            how='inner'
        )

        if df_courses_data.empty:
            self.logger.warning(f"No hay datos de cursos que coincidan con el dataset principal{sede_suffix}")
            return

        self.logger.info(f"Cursos después de filtrar por dataset principal: {len(df_courses_data)}")

        # Columnas de porcentajes
        percentage_columns = {
            'percent_evaluation': 'Evaluación (%)',
            'percent_collaboration': 'Colaboración (%)',
            'percent_content': 'Contenido (%)',
            'percent_in_english': 'En Inglés (%)',
            'percent_interactive': 'Interactivo (%)',
            'percent_updated': 'Actualizado (%)'
        }

        # Verificar que tengamos las columnas necesarias
        available_cols = [col for col in percentage_columns.keys() if col in df_courses_data.columns]

        if len(available_cols) == 0:
            self.logger.warning(f"No hay columnas de porcentajes disponibles{sede_suffix}")
            return

        # Crear identificador legible de ASIGNATURA - GRADO
        df_courses_data['curso_label'] = df_courses_data.apply(self._create_asignatura_label, axis=1)

        # Agrupar por asignatura-grado y calcular la MEDIANA
        group_cols = ['curso_label'] + available_cols
        matrix_data = df_courses_data[group_cols].groupby('curso_label').median()

        # Convertir a porcentaje 0-100
        matrix_data = matrix_data * 100

        # Ordenar por nombre
        matrix_data = matrix_data.sort_index()

        self.logger.info(f"Mostrando todas las {len(matrix_data)} combinaciones asignatura-grado")

        # Renombrar columnas
        matrix_data.columns = [percentage_columns[col] for col in available_cols]

        # Crear figura
        fig_height = max(10, len(matrix_data) * 0.4)
        fig, ax = plt.subplots(figsize=(14, fig_height))

        # Crear paleta de colores monocromática con transparencia
        # Usamos un gradiente de azul con diferentes intensidades
        from matplotlib.colors import LinearSegmentedColormap
        colors_gradient = ['#e6f2ff', '#cce5ff', '#99ccff', '#66b3ff', '#3399ff', '#0080ff', '#0066cc']
        n_bins = 100
        cmap_mono = LinearSegmentedColormap.from_list('mono_blue', colors_gradient, N=n_bins)

        # Crear mapa de calor con paleta monocromática
        sns.heatmap(
            matrix_data,
            annot=True,
            fmt='.1f',
            cmap=cmap_mono,
            cbar_kws={'label': 'Porcentaje (%)'},
            linewidths=0.5,
            linecolor='gray',
            ax=ax,
            vmin=0,
            vmax=100
        )

        # Configurar etiquetas
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)

        # Títulos y etiquetas
        ax.set_title(f'Mapa de Calor: Mediana de Composición por Asignatura-Grado (Tipo de Contenido){sede_suffix}', 
                    fontsize=14, pad=20, weight='bold')
        ax.set_xlabel('Tipo de Contenido', fontsize=11)
        ax.set_ylabel('Asignatura - Grado', fontsize=11)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/heatmap_composicion_cursos{sede_file_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Mapa de calor de composición de cursos creado{sede_suffix}")

    def create_subject_grade_count_bar_chart(self, output_dir: str, sede: str = None):
        """Crear gráfico de barras apiladas con el conteo de registros por ASIGNATURA-GRADO y SEDE. Genera una gráfica general mostrando la distribución por sede en barras apiladas."""
        self.logger.info(f"Creando gráfico de conteo por asignatura-grado (todas las sedes)...")

        df_main_data = self.df_merged.copy()

        if df_main_data.empty:
            self.logger.warning(f"No hay datos en el dataset principal")
            return

        # Verificar que tengamos la columna sede
        if 'sede' not in df_main_data.columns:
            self.logger.warning("No hay columna 'sede' en el dataset")
            return

        # Contar por id_asignatura, id_grado y sede
        conteo_ids = df_main_data.groupby(['id_asignatura', 'id_grado', 'sede']).size().reset_index(name='count')

        self.logger.info(f"Total de combinaciones asignatura-grado: {len(conteo_ids)}")
        self.logger.info(f"Total de registros: {conteo_ids['count'].sum()}")

        # Agregar nombres de asignaturas y crear etiquetas
        conteo_ids = self._merge_asignatura_names(conteo_ids)
        conteo_ids['asignatura_grado'] = conteo_ids.apply(self._create_asignatura_label, axis=1)

        # Ordenar por grado (de menor a mayor) y luego por nombre de asignatura
        if 'asignatura' in conteo_ids.columns:
            conteo_ids = conteo_ids.sort_values(['id_grado', 'asignatura'])
        else:
            conteo_ids = conteo_ids.sort_values(['id_grado', 'id_asignatura'])

        # Crear pivot table con sedes como columnas
        pivot_data = conteo_ids.pivot_table(
            index='asignatura_grado',
            columns='sede',
            values='count',
            aggfunc='sum',
            fill_value=0
        )

        # Mantener el orden de las etiquetas
        pivot_data = pivot_data.reindex(conteo_ids['asignatura_grado'].unique())

        # Obtener sedes únicas y colores (usar paleta tab20b para consistencia)
        sedes = pivot_data.columns.tolist()
        colores_sedes = plt.cm.tab20b(np.linspace(0, 1, len(sedes)))

        # Calcular totales por fila
        totales = pivot_data.sum(axis=1).values

        # Crear figura
        n_rows = len(pivot_data)
        fig_width = 14
        fig_height = max(6, n_rows * 0.3)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Crear barras apiladas horizontales
        y_pos = np.arange(n_rows)
        left = np.zeros(n_rows)

        bars_by_sede = []
        for i, sede in enumerate(sedes):
            bars = ax.barh(y_pos, pivot_data[sede].values, left=left, 
                          color=colores_sedes[i], edgecolor='white', linewidth=0.5,
                          label=sede)
            bars_by_sede.append(bars)

            # Añadir valores en las barras si son significativos
            for j, (bar, value) in enumerate(zip(bars, pivot_data[sede].values)):
                if value > 0:  # Solo mostrar si hay datos
                    text_x = left[j] + value / 2
                    text_color = self._calculate_text_contrast_color(colores_sedes[i])
                    
                    ax.text(text_x, bar.get_y() + bar.get_height()/2, 
                           f'{int(value)}', va='center', ha='center', 
                           fontsize=6, weight='bold', color=text_color)

            left += pivot_data[sede].values

        # Añadir totales al final de cada barra
        max_total = totales.max()
        for j, (y, total) in enumerate(zip(y_pos, totales)):
            ax.text(total + max_total * 0.01, y, 
                   f'{int(total)}', va='center', ha='left', 
                   fontsize=7, weight='bold', color='darkblue')

        # Configurar etiquetas
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pivot_data.index, fontsize=8)
        ax.set_xlabel('Número de Registros', fontsize=11, weight='bold')
        ax.set_ylabel('Asignatura - Grado', fontsize=11, weight='bold')
        ax.set_title('Conteo de Registros por Asignatura-Grado y Sede', 
                     fontsize=13, pad=15, weight='bold')

        # Añadir leyenda
        ax.legend(loc='best', fontsize=9, title='Sede', title_fontsize=10, framealpha=0.9)

        # Añadir grid para facilitar lectura
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/conteo_registros_asignatura_grado.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Gráfico de conteo por asignatura-grado creado (todas las sedes)")

    def analyze_course_composition(self, output_dir: str, sede: str = None):
        """Analizar la composición de cursos (módulos, actividades, estudiantes)."""
        sede_suffix, sede_file_suffix = self._get_sede_suffixes(sede)
        self.logger.info(f"Analizando composición de cursos{sede_suffix}...")

        # Filtrar por sede
        df_courses_data = self._filter_by_sede(self.df_courses, sede)

        if df_courses_data.empty:
            self.logger.warning(f"No hay datos de cursos para analizar{sede_suffix}")
            return

        # Crear figura con 2x2 subplots (solo gráficas activas)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # 1. Distribución de número de módulos por curso
        if 'num_modules' in df_courses_data.columns:
            axes[0].hist(df_courses_data['num_modules'].dropna(), bins=20, color='skyblue', edgecolor='black')
            axes[0].set_title('Distribución de Número de Módulos por Curso', fontsize=12)
            axes[0].set_xlabel('Número de Módulos')
            axes[0].set_ylabel('Frecuencia')
            axes[0].axvline(df_courses_data['num_modules'].median(), color='red', linestyle='--', 
                           label=f'Mediana: {df_courses_data["num_modules"].median():.1f}')
            axes[0].legend()
        else:
            axes[0].text(0.5, 0.5, 'Datos no disponibles', ha='center', va='center', transform=axes[0].transAxes)

        # 2. Porcentaje de módulos actualizados
        if 'percent_updated' in df_courses_data.columns:
            axes[1].hist(df_courses_data['percent_updated'].dropna() * 100, bins=20, color='orange', edgecolor='black')
            axes[1].set_title('Porcentaje de Módulos Actualizados', fontsize=12)
            axes[1].set_xlabel('Porcentaje (%)')
            axes[1].set_ylabel('Frecuencia')
            axes[1].axvline(df_courses_data['percent_updated'].median() * 100, color='red', linestyle='--',
                           label=f'Mediana: {df_courses_data["percent_updated"].median() * 100:.1f}%')
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, 'Datos no disponibles', ha='center', va='center', transform=axes[1].transAxes)

        # 3. Composición de contenido (evaluation, collaboration, content, etc.)
        content_cols = ['count_evaluation', 'count_collaboration', 'count_content', 'count_interactive']
        if all(col in df_courses_data.columns for col in content_cols):
            content_means = df_courses_data[content_cols].mean()
            axes[2].bar(range(len(content_means)), content_means.values, 
                       color=['red', 'blue', 'green', 'purple'], alpha=0.7)
            axes[2].set_title('Promedio de Tipo de Contenido por Curso', fontsize=12)
            axes[2].set_xticks(range(len(content_means)))
            axes[2].set_xticklabels(['Evaluación', 'Colaboración', 'Contenido', 'Interactivo'], rotation=45, ha='right')
            axes[2].set_ylabel('Promedio')
        else:
            axes[2].text(0.5, 0.5, 'Datos no disponibles', ha='center', va='center', transform=axes[2].transAxes)

        # 4. Actividad docente (vistas antes de inicio planificado)
        if 'num_teacher_views_before_planned_start_date' in df_courses_data.columns:
            axes[3].hist(df_courses_data['num_teacher_views_before_planned_start_date'].dropna(), 
                        bins=20, color='coral', edgecolor='black')
            axes[3].set_title('Vistas de Docente Antes de Inicio Planificado', fontsize=12)
            axes[3].set_xlabel('Número de Vistas')
            axes[3].set_ylabel('Frecuencia')
            median_val = df_courses_data['num_teacher_views_before_planned_start_date'].median()
            axes[3].axvline(median_val, color='red', linestyle='--',
                           label=f'Mediana: {median_val:.1f}')
            axes[3].legend()
        else:
            axes[3].text(0.5, 0.5, 'Datos no disponibles', ha='center', va='center', transform=axes[3].transAxes)

        plt.suptitle(f'Análisis de Composición de Cursos{sede_suffix}', fontsize=16, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(f"{output_dir}/analisis_composicion_cursos{sede_file_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Análisis de composición de cursos completado{sede_suffix}")

    def create_honeycomb_module_charts(self, output_dir: str, asignaturas: list = None, sede: str = None):
        """
        Crear gráficas de panal de abeja para visualizar módulos por asignatura.

        Características de la visualización:
        - Eje X: año-periodo
        - Eje Y: módulos
        - Forma: estrella para módulos interactivos, círculo para solo lectura
        - Color: rosado si es en inglés, azul si es en español
        - Tamaño: número de estudiantes que accedieron al recurso
        - Transparencia: updates del docente (más opaco = más updates)

        Args:
            output_dir: Directorio de salida
            asignaturas: Lista de IDs de asignaturas a visualizar (por defecto self.subject_filter)
            sede: Filtrar por sede específica (opcional)
        """
        import time
        
        if asignaturas is None:
            asignaturas = self.subject_filter
            
        sede_suffix, sede_file_suffix = self._get_sede_suffixes(sede)
        self.logger.info(f"Creando {len(asignaturas)} gráficas de panal de abeja{sede_suffix}...")

        # Verificar que tengamos los datos de módulos
        if self.df_modules_featured.empty:
            self.logger.warning(f"No hay datos de módulos para crear gráficas de panal de abeja{sede_suffix}")
            return

        # Filtrar por sede
        df_modules = self._filter_by_sede(self.df_modules_featured, sede)

        if df_modules.empty:
            self.logger.warning(f"No hay datos de módulos después de filtrar por sede{sede_suffix}")
            return

        # Convertir columnas a valores numéricos si no lo son (una sola vez)
        if 'is_interactive' in df_modules.columns:
            df_modules['is_interactive'] = pd.to_numeric(df_modules['is_interactive'], errors='coerce').fillna(0).astype(int)
        if 'is_in_english' in df_modules.columns:
            df_modules['is_in_english'] = pd.to_numeric(df_modules['is_in_english'], errors='coerce').fillna(0).astype(int)

        # Crear una gráfica por cada asignatura con progreso
        start_time = time.time()
        for i, id_asignatura in enumerate(asignaturas, 1):
            chart_start = time.time()
            self.logger.info(f"[{i}/{len(asignaturas)}] Generando gráfica para asignatura {id_asignatura}...")
            
            self._create_single_honeycomb_chart(df_modules, id_asignatura, output_dir, sede_suffix, sede_file_suffix)
            
            chart_time = time.time() - chart_start
            self.logger.info(f"  ✓ Completada en {chart_time:.1f}s")

        total_time = time.time() - start_time
        self.logger.info(f"✅ {len(asignaturas)} gráficas de panal de abeja creadas en {total_time:.1f}s{sede_suffix}")

    def _create_single_honeycomb_chart(self, df_modules: pd.DataFrame, id_asignatura: int, 
                                       output_dir: str, sede_suffix: str, sede_file_suffix: str):
        """
        Crear una gráfica de panal de abeja para una asignatura específica.

        Args:
            df_modules: DataFrame con información de módulos
            id_asignatura: ID de la asignatura
            output_dir: Directorio de salida
            sede_suffix: Sufijo para el título
            sede_file_suffix: Sufijo para el archivo
        """
        # Filtrar por asignatura
        df_asig = df_modules[df_modules['id_asignatura'] == id_asignatura].copy()

        if df_asig.empty:
            self.logger.warning(f"No hay datos para asignatura {id_asignatura}{sede_suffix}")
            return

        # Obtener nombre de la asignatura
        asignatura_name = self._get_asignatura_name(id_asignatura)
        asignatura_slug = re.sub(r'[^a-zA-Z0-9]+', '_', asignatura_name.lower()).strip('_')

        self.logger.info(f"Creando gráfica para {asignatura_name} con {len(df_asig)} módulos{sede_suffix}")

        # Verificar que tengamos la columna de grado
        if 'id_grado' not in df_asig.columns:
            self.logger.warning(f"No hay columna id_grado para asignatura {id_asignatura}{sede_suffix}")
            return

        # Crear etiqueta año-periodo para eje X
        df_asig['year_period'] = df_asig['year'].astype(str) + '-P' + df_asig['period'].astype(str)

        # Ordenar por year_period, grado, y orden dentro del periodo
        if 'order' in df_asig.columns:
            df_asig = df_asig.sort_values(['year_period', 'id_grado', 'order'])
        else:
            df_asig = df_asig.sort_values(['year_period', 'id_grado', 'week'])

        # Obtener todas las combinaciones únicas de año-periodo
        year_periods = sorted(df_asig['year_period'].unique())

        # Para cada año-periodo, contar cuántos módulos hay por grado
        # y asignar posiciones X distribuidas dentro del periodo
        df_asig['x_pos'] = 0.0
        df_asig['y_pos'] = 0.0

        # Calcular densidad máxima para ajustar visualización
        max_modulos_por_grupo = 0

        for i, yp in enumerate(year_periods):
            mask = df_asig['year_period'] == yp
            df_periodo = df_asig[mask]

            # Agrupar por grado dentro de este periodo
            for grado in df_periodo['id_grado'].unique():
                mask_grado = (df_asig['year_period'] == yp) & (df_asig['id_grado'] == grado)
                n_modulos_grado = mask_grado.sum()

                if n_modulos_grado > max_modulos_por_grupo:
                    max_modulos_por_grupo = n_modulos_grado

                if n_modulos_grado > 0:
                    # Distribuir horizontalmente los módulos de este grado en este periodo
                    # Usar un rango más amplio para mejor dispersión
                    horizontal_range = 0.9  # Más ancho que antes
                    positions_x = np.linspace(i - horizontal_range/2, i + horizontal_range/2, n_modulos_grado)

                    # Añadir jitter vertical para evitar superposición
                    # Balance entre separación y agrupación visual por grado
                    if n_modulos_grado > 20:
                        vertical_jitter = 0.32  # Balance óptimo para alta densidad
                    elif n_modulos_grado > 10:
                        vertical_jitter = 0.25  # Balance óptimo para media densidad
                    else:
                        vertical_jitter = 0.18  # Balance óptimo para baja densidad

                    # Generar posiciones Y con jitter aleatorio alrededor del grado
                    np.random.seed(42 + grado + i)  # Seed para reproducibilidad
                    positions_y = grado + np.random.uniform(-vertical_jitter, vertical_jitter, n_modulos_grado)

                    df_asig.loc[mask_grado, 'x_pos'] = positions_x
                    df_asig.loc[mask_grado, 'y_pos'] = positions_y

        self.logger.info(f"Densidad máxima: {max_modulos_por_grupo} módulos en un grado/periodo")

        # Etiquetas para el eje Y (grados)
        grados_unicos = sorted(df_asig['id_grado'].unique())
        y_labels = [f"Grado {g}" for g in grados_unicos]

        # Preparar características visuales

        # 1. Tamaño: basado en estudiantes únicos que accedieron (normalizado) y ajustado por densidad
        if 'students_who_viewed' in df_asig.columns:
            df_asig['students_who_viewed_clean'] = pd.to_numeric(df_asig['students_who_viewed'], errors='coerce').fillna(0)
            max_students = df_asig['students_who_viewed_clean'].max()
            min_students = df_asig['students_who_viewed_clean'].min()

            # Ajustar rango de tamaños según densidad máxima
            if max_modulos_por_grupo > 30:
                # Muchos módulos: usar puntos más pequeños
                size_min, size_max = 30, 200
            elif max_modulos_por_grupo > 15:
                size_min, size_max = 40, 300
            else:
                size_min, size_max = 50, 400

            if max_students > min_students:
                df_asig['point_size'] = size_min + (df_asig['students_who_viewed_clean'] - min_students) / (max_students - min_students) * (size_max - size_min)
            else:
                df_asig['point_size'] = (size_min + size_max) / 2
        else:
            # Tamaño por defecto basado en densidad
            if max_modulos_por_grupo > 30:
                df_asig['point_size'] = 60
            else:
                df_asig['point_size'] = 100

        # 2. Transparencia: basada en teacher_total_updates, ajustada por densidad
        # Más updates = más opaco (menos transparente) = más trabajo docente
        if 'teacher_total_updates' in df_asig.columns:
            df_asig['teacher_total_updates_clean'] = pd.to_numeric(df_asig['teacher_total_updates'], errors='coerce').fillna(0)
            max_updates = df_asig['teacher_total_updates_clean'].max()

            # Ajustar transparencia según densidad
            if max_modulos_por_grupo > 30:
                # Alta densidad: usar más transparencia base para ver superposiciones
                alpha_min, alpha_max = 0.4, 0.8
            elif max_modulos_por_grupo > 15:
                alpha_min, alpha_max = 0.4, 0.9
            else:
                alpha_min, alpha_max = 0.5, 1.0

            if max_updates > 0:
                df_asig['alpha'] = alpha_min + (df_asig['teacher_total_updates_clean'] / max_updates) * (alpha_max - alpha_min)
            else:
                df_asig['alpha'] = (alpha_min + alpha_max) / 2
        else:
            # Transparencia por defecto basada en densidad
            if max_modulos_por_grupo > 30:
                df_asig['alpha'] = 0.5
            else:
                df_asig['alpha'] = 0.7

        # 3. Color: basado en is_in_english (vectorizado para performance)
        # Rosado para inglés, azul para español
        df_asig['color'] = np.where(df_asig['is_in_english'] == 1, '#FF69B4', '#4169E1')

        # 4. Forma: basada en is_interactive (vectorizado para performance)
        # Estrella para interactivo, círculo para solo lectura
        df_asig['marker'] = np.where(df_asig['is_interactive'] == 1, '*', 'o')

        # Crear figura con tamaño apropiado para visualización por grados
        # Ahora el eje Y es por grado (máximo 11), mucho más manejable
        n_grados = len(grados_unicos)
        fig_width = max(14, len(year_periods) * 2)  # Ancho proporcional al número de periodos
        fig_height = max(8, n_grados * 1.5)  # Altura proporcional al número de grados

        dpi_to_use = 300  # Siempre usar alta calidad ya que ahora es manejable

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Ajustar grosor de borde según densidad
        if max_modulos_por_grupo > 30:
            edge_width = 0.3
        elif max_modulos_por_grupo > 15:
            edge_width = 0.4
        else:
            edge_width = 0.5

        # Agrupar por forma y dibujar con arrays (vectorizado para performance máxima)
        for marker_type in df_asig['marker'].unique():
            df_marker = df_asig[df_asig['marker'] == marker_type]
            
            # Dibujar todos los puntos del mismo marker de una vez (10-20x más rápido)
            # Matplotlib puede manejar arrays de colores, tamaños y alphas
            scatter = ax.scatter(
                df_marker['x_pos'].values,
                df_marker['y_pos'].values,
                s=df_marker['point_size'].values,
                c=df_marker['color'].values,
                marker=marker_type,
                alpha=df_marker['alpha'].values,
                edgecolors='black',
                linewidths=edge_width
            )

        # Configurar ejes
        # Eje X: año-periodo con marcas principales
        ax.set_xticks(range(len(year_periods)))
        ax.set_xticklabels(year_periods, rotation=45, ha='right', fontsize=11)
        ax.set_xlabel('Año - Periodo', fontsize=13, weight='bold')

        # Añadir líneas verticales para separar periodos
        for i in range(len(year_periods)):
            ax.axvline(x=i, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

        # Eje Y: grados (1-11)
        ax.set_yticks(grados_unicos)
        ax.set_yticklabels(y_labels, fontsize=11)
        ax.set_ylabel('Grado', fontsize=13, weight='bold')

        # Ajustar límites del eje Y para dar espacio
        ax.set_ylim(min(grados_unicos) - 0.5, max(grados_unicos) + 0.5)

        # Añadir líneas horizontales para separar grados
        for grado in grados_unicos:
            ax.axhline(y=grado, color='gray', linestyle='-', alpha=0.15, linewidth=0.5)

        # Título con información de densidad
        titulo_base = f'Visualización de Módulos por Grado - {asignatura_name}{sede_suffix}'
        if max_modulos_por_grupo > 30:
            densidad_info = f' (Máx: {max_modulos_por_grupo} módulos en un grado/periodo)'
        else:
            densidad_info = ''

        ax.set_title(f'{titulo_base}{densidad_info}\n'
                    f'Total: {len(df_asig)} modulos | Forma: Estrella=Interactivo / Circulo=Lectura\n'
                    f'Color: Rosa=Ingles / Azul=Español | Tamano=N Estudiantes | Opacidad=Updates Docente',
                    fontsize=11, weight='bold', pad=15)

        # Añadir grid para facilitar lectura
        ax.grid(axis='both', alpha=0.2, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        # Crear leyenda personalizada
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markersize=12, 
                   label='Interactivo', markeredgecolor='black', linewidth=0),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, 
                   label='Solo lectura', markeredgecolor='black', linewidth=0),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF69B4', markersize=8, 
                   label='En inglés', markeredgecolor='black', linewidth=0),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#4169E1', markersize=8, 
                   label='En español', markeredgecolor='black', linewidth=0),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/honeycomb_modulos_{asignatura_slug}{sede_file_suffix}.png", 
                   dpi=dpi_to_use, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Gráfica de panal de abeja para {asignatura_name} creada: {len(df_asig)} módulos en {len(grados_unicos)} grados{sede_suffix}")

    def create_visualizations(self, output_dir: str):
        """
        Crear visualizaciones de análisis de cursos por sede.

        Por defecto solo genera visualizaciones por sede individual.
        Las visualizaciones generales (todas las sedes juntas) están desactivadas
        y deben llamarse explícitamente si se necesitan.

        Args:
            output_dir: Directorio de salida
        """
        self.logger.info("Creando visualizaciones de cursos...")

        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)

        # Crear gráfica de conteo general (todas las sedes en barras apiladas)
        self.logger.info("Creando gráfica de conteo general...")
        self.create_subject_grade_count_bar_chart(output_dir)

        # Obtener sedes únicas
        if 'sede' in self.df_merged.columns:
            sedes_unicas = sorted(self.df_merged['sede'].unique())
            self.logger.info(f"Creando visualizaciones para {len(sedes_unicas)} sedes: {sedes_unicas}")

            # Crear visualizaciones para cada sede
            for sede in sedes_unicas:
                self.logger.info(f"Procesando sede: {sede}")

                # Crear subdirectorio para esta sede
                sede_dir = os.path.join(output_dir, sede.lower())
                os.makedirs(sede_dir, exist_ok=True)

                # Crear visualizaciones para esta sede (sin conteo, que ya es general)
                self.create_course_composition_heatmap(sede_dir, sede)
                self.analyze_course_composition(sede_dir, sede)
                self.create_honeycomb_module_charts(sede_dir, sede=sede)


    def create_general_visualizations(self, output_dir: str):
        """
        Crear visualizaciones generales (todas las sedes juntas).

        Este método debe llamarse explícitamente si se desean
        visualizaciones agregadas de todas las sedes.

        Args:
            output_dir: Directorio de salida
        """
        self.logger.info("Creando visualizaciones generales (todas las sedes)...")
        self.create_course_composition_heatmap(output_dir)
        self.create_subject_grade_count_bar_chart(output_dir)
        self.analyze_course_composition(output_dir)
        self.create_honeycomb_module_charts(output_dir)
        self.logger.info("Visualizaciones generales completadas")

    def run_analysis(self):
        """Ejecutar análisis completo de cursos."""
        import time
        
        total_start = time.time()
        self.logger.info("="*60)
        self.logger.info("🚀 Iniciando análisis de cursos...")
        self.logger.info(f"📁 Resultados: {self.results_path}")
        self.logger.info("="*60)

        # Crear directorio de resultados si no existe
        self.create_results_directory()

        # Cargar y preparar datos
        self.logger.info("📊 Cargando y preparando datos...")
        load_start = time.time()
        df = self.load_and_prepare_data()
        load_time = time.time() - load_start
        self.logger.info(f"✓ Datos cargados en {load_time:.1f}s")

        # Verificar que tenemos datos
        if df.empty:
            self.logger.error("❌ No hay datos para analizar")
            return

        # Crear visualizaciones
        self.logger.info("📈 Generando visualizaciones...")
        viz_start = time.time()
        self.create_visualizations(self.results_path)
        viz_time = time.time() - viz_start
        self.logger.info(f"✓ Visualizaciones completadas en {viz_time:.1f}s")

        total_time = time.time() - total_start
        self.logger.info("="*60)
        self.logger.info(f"✅ Análisis completado en {total_time:.1f}s ({total_time/60:.1f} min)")
        self.logger.info("="*60)


def main():
    """Función principal."""
    import argparse

    parser = argparse.ArgumentParser(description='Análisis de cursos en Moodle')
    parser.add_argument('--dataset', '-d', type=str, required=True,
                       help='Ruta al archivo CSV del dataset (requerido)')
    parser.add_argument('--results', '-r', type=str, required=True,
                       help='Nombre del folder para guardar resultados (requerido, se creará en reports/)')

    args = parser.parse_args()

    # Crear y ejecutar analizador siguiendo el patrón estándar
    analyzer = CourseAnalysis(dataset_path=args.dataset, results_folder=args.results)

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

