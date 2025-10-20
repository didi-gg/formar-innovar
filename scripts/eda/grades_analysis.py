"""Script para análisis de calificaciones históricas 2021-2025."""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para mejor rendimiento en paralelo
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging

# Configuración de warnings y logging
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)

# Silenciar mensajes de debug adicionales
matplotlib.set_loglevel("WARNING")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.eda_analysis_base import EDAAnalysisBase


class GradesAnalysis(EDAAnalysisBase):
    """Analizador de calificaciones históricas."""

    # Constantes de clase para evitar duplicación
    PERFORMANCE_COLORS = {
        'Bajo': '#e74c3c',
        'Básico': '#f39c12',
        'Alto': '#3498db',
        'Superior': '#2ecc71',
        'Sin Datos': '#95a5a6'
    }

    PERFORMANCE_LEVELS = ['Bajo', 'Básico', 'Alto', 'Superior', 'Sin Datos']

    COMPETENCIAS_NOMBRES = {
        'axiológico': 'Axiológica',
        'cognitivo': 'Cognitiva',
        'procedimental': 'Procedimental',
        'actitudinal': 'Actitudinal'
    }

    COMPETENCIAS_COLORS = {
        'cognitivo': '#3498db',
        'procedimental': '#9b59b6',
        'actitudinal': '#2ecc71',
        'axiológico': '#e74c3c'
    }

    # Colores para cada asignatura
    ASIGNATURAS_COLORS = {
        1: '#e74c3c',  # Rojo
        2: '#3498db',  # Azul
        3: '#2ecc71',  # Verde
        4: '#f39c12'   # Naranja
    }

    def _initialize_analysis_attributes(self):
        """Inicializar atributos específicos del análisis de calificaciones."""
        self.df = None
        self.results = {}
        self._asignaturas_map = None  # Cache para la tabla de asignaturas

    def load_grades_data(self):
        """Carga los datos de calificaciones."""
        self.logger.info(f"Cargando datos de calificaciones desde: {self.dataset_path}")

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"No se encontró el dataset en: {self.dataset_path}")

        # Optimizar carga con tipos de datos específicos para reducir memoria
        df = pd.read_csv(self.dataset_path, 
                        dtype={'año': 'int16', 'periodo': 'int8'},
                        low_memory=False)
        self.logger.info(f"Dataset cargado: {df.shape[0]:,} registros, {df.shape[1]} columnas")

        # Validar columnas necesarias
        required_columns = ['sede', 'año', 'periodo', 'identificación']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Columnas requeridas faltantes: {missing_columns}")

        self.df = df
        return df

    def create_periodo_column(self):
        """Crea una columna combinada año-periodo para el análisis temporal."""
        self.logger.info("Creando columna año-periodo...")

        # Crear columna año-periodo (ej: "2021-1", "2021-2")
        self.df['año_periodo'] = self.df['año'].astype(str) + '-' + self.df['periodo'].astype(str)

        # Ordenar por año y periodo
        self.df['año_num'] = self.df['año'].astype(int)
        self.df['periodo_num'] = self.df['periodo'].astype(int)
        self.df = self.df.sort_values(['año_num', 'periodo_num'])

        unique_periodos = self.df['año_periodo'].nunique()
        self.logger.info(f"Períodos únicos identificados: {unique_periodos}")

        return self.df

    # ============================================================================
    # MÉTODOS HELPER PARA EVITAR DUPLICACIÓN
    # ============================================================================

    @staticmethod
    def classify_grade(grade):
        """Clasifica una calificación en nivel de desempeño."""
        if pd.isna(grade):
            return 'Sin Datos'
        elif grade < 60:
            return 'Bajo'
        elif grade < 70:
            return 'Básico'
        elif grade < 90:
            return 'Alto'
        else:
            return 'Superior'

    @staticmethod
    def calc_reprobacion(group):
        """Calcula porcentaje de reprobación de un grupo."""
        total = len(group)
        reprobados = (group < 60).sum()
        return (reprobados / total * 100) if total > 0 else 0

    def _setup_subplots(self, n_items, n_cols=2, figsize_per_item=(12, 7)):
        n_cols = min(n_cols, n_items)
        n_rows = (n_items + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, 
            figsize=(figsize_per_item[0]*n_cols, figsize_per_item[1]*n_rows)
        )

        # Normalizar axes a array 2D
        if n_items == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        return fig, axes

    def _hide_empty_subplots(self, axes, n_used, n_cols):
        """Oculta subplots vacíos."""
        n_rows = axes.shape[0]
        total_subplots = n_rows * n_cols

        for idx in range(n_used, total_subplots):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

    def _load_asignaturas_map(self):
        """Carga y cachea el mapeo de asignaturas (id -> nombre)."""
        if self._asignaturas_map is not None:
            return self._asignaturas_map

        asignaturas_path = 'data/raw/tablas_maestras/asignaturas.csv'
        try:
            asignaturas_df = pd.read_csv(asignaturas_path)
            asig_map = {}
            for _, row in asignaturas_df.iterrows():
                asig_map[int(row['id_asignatura'])] = row['nombre']
                asig_map[str(row['id_asignatura'])] = row['nombre']

            self._asignaturas_map = asig_map
            self.logger.info(f"Tabla de asignaturas cargada: {len(asignaturas_df)} asignaturas")
        except Exception as e:
            self.logger.warning(f"No se pudo cargar tabla de asignaturas: {e}. Usando nombres originales.")
            self._asignaturas_map = {}

        return self._asignaturas_map

    def _get_competencias_disponibles(self):
        """Retorna lista de competencias disponibles en el dataset."""
        competencias = ['axiológico', 'cognitivo', 'procedimental', 'actitudinal']
        return [comp for comp in competencias if comp in self.df.columns]

    # ============================================================================
    # MÉTODOS DE GRAFICACIÓN
    # ============================================================================

    def plot_performance_levels_by_period(self):
        """Genera gráfico de barras apiladas con niveles de desempeño por período y sede."""
        self.logger.info("Generando gráfico de niveles de desempeño por período y sede...")

        sedes_list = self.results['sedes_list']

        # Clasificar calificaciones usando método helper
        self.df['nivel_desempeño'] = self.df['resultado'].apply(self.classify_grade)

        # Usar constantes de clase
        colors = self.PERFORMANCE_COLORS
        niveles_orden = self.PERFORMANCE_LEVELS

        # Setup de subplots usando helper
        n_sedes = len(sedes_list)
        n_cols = 2
        fig, axes = self._setup_subplots(n_sedes, n_cols=n_cols, figsize_per_item=(12, 7))

        # Crear un subplot por sede
        for idx, sede in enumerate(sedes_list):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # Filtrar datos para esta sede
            sede_df = self.df[self.df['sede'] == sede]

            # Obtener períodos ordenados solo para esta sede
            sede_periodos = sorted(sede_df[['año_num', 'periodo_num', 'año_periodo']].drop_duplicates().values.tolist())
            sede_periodos_labels = [p[2] for p in sede_periodos]

            # Si no hay datos para esta sede, continuar
            if len(sede_periodos_labels) == 0:
                ax.text(0.5, 0.5, f'Sin datos disponibles\npara {sede}', 
                       ha='center', va='center', fontsize=12, transform=ax.transAxes)
                ax.axis('off')
                continue

            # Contar estudiantes por nivel y período para esta sede
            nivel_counts = []
            for periodo in sede_periodos_labels:
                periodo_data = sede_df[sede_df['año_periodo'] == periodo]
                counts = periodo_data['nivel_desempeño'].value_counts()
                total = len(periodo_data)

                nivel_counts.append({
                    'periodo': periodo,
                    'Bajo': (counts.get('Bajo', 0) / total * 100) if total > 0 else 0,
                    'Básico': (counts.get('Básico', 0) / total * 100) if total > 0 else 0,
                    'Alto': (counts.get('Alto', 0) / total * 100) if total > 0 else 0,
                    'Superior': (counts.get('Superior', 0) / total * 100) if total > 0 else 0,
                    'Sin Datos': (counts.get('Sin Datos', 0) / total * 100) if total > 0 else 0
                })

            nivel_df = pd.DataFrame(nivel_counts)

            # Crear barras apiladas
            x_pos = range(len(sede_periodos_labels))
            bottom = np.zeros(len(sede_periodos_labels))

            for nivel in niveles_orden:
                values = nivel_df[nivel].values
                bars = ax.bar(x_pos, values, bottom=bottom, 
                             label=nivel, color=colors[nivel], 
                             edgecolor='white', linewidth=0.5, alpha=0.9)

                # Agregar porcentajes en las barras (solo si es > 5%)
                for i, (bar, value) in enumerate(zip(bars, values)):
                    if value > 5:  # Solo mostrar si es mayor a 5%
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., bottom[i] + height/2.,
                               f'{value:.1f}%',
                               ha='center', va='center', fontsize=8, 
                               fontweight='bold', color='white')

                bottom += values

            # Configuración del subplot
            ax.set_xlabel('Año - Período', fontsize=11, fontweight='bold')
            ax.set_ylabel('Porcentaje de Estudiantes (%)', fontsize=11, fontweight='bold')

            # Agregar información de años disponibles en el título
            años_range = f"{sede_periodos[0][0]}-{sede_periodos[-1][0]}"
            ax.set_title(f'Sede: {sede} ({años_range})', fontsize=13, fontweight='bold', pad=10)

            # Configurar eje X
            ax.set_xticks(x_pos)
            ax.set_xticklabels(sede_periodos_labels, rotation=45, ha='right', fontsize=9)

            # Leyenda solo en el primer subplot
            if idx == 0:
                ax.legend(title='Nivel de Desempeño', fontsize=9, title_fontsize=10,
                         loc='upper left', bbox_to_anchor=(1, 1), framealpha=0.95)

            # Grid horizontal
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
            ax.set_ylim([0, 100])

        # Ocultar subplots vacíos usando helper
        self._hide_empty_subplots(axes, n_sedes, n_cols)

        # Título general
        fig.suptitle('Distribución de Niveles de Desempeño por Período y Sede', 
                    fontsize=16, fontweight='bold', y=0.995)

        # Ajustar layout
        plt.tight_layout()

        # Guardar
        output_path = f'{self.results_path}/performance_levels_by_period.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✅ Gráfico guardado: {output_path}")

        return output_path

    def plot_grades_distribution_by_periodo_sede(self):
        """Genera boxplots de la distribución de calificaciones por período y sede."""
        self.logger.info("Generando boxplots de distribución de calificaciones por período y sede...")

        sedes_list = self.results['sedes_list']

        # Obtener todos los períodos únicos ordenados
        periodos_ordenados = sorted(self.df[['año_num', 'periodo_num', 'año_periodo']].drop_duplicates().values.tolist())
        periodos_labels = [p[2] for p in periodos_ordenados]

        # Preparar datos para boxplot
        # Crear una lista de datos por período
        data_by_periodo = []
        positions = []
        colors_list = []
        labels_list = []

        # Obtener paleta de colores para las sedes
        sede_colors = self.get_beautiful_palette(len(sedes_list), palette_name='tab20b')
        sede_color_map = {sede: sede_colors[i] for i, sede in enumerate(sedes_list)}

        # Configurar ancho de las cajas y espaciado
        box_width = 0.35
        spacing = 0.15
        total_width = len(sedes_list) * box_width + (len(sedes_list) - 1) * spacing

        # Crear figura
        fig, ax = plt.subplots(figsize=(max(16, len(periodos_labels) * 1.2), 10))

        # Para cada período, crear boxplots para cada sede
        for periodo_idx, periodo in enumerate(periodos_labels):
            periodo_data = self.df[self.df['año_periodo'] == periodo]

            # Calcular posición base para este período
            base_position = periodo_idx * (total_width + 1)

            for sede_idx, sede in enumerate(sedes_list):
                sede_periodo_data = periodo_data[periodo_data['sede'] == sede]['resultado'].dropna()

                if len(sede_periodo_data) > 0:
                    data_by_periodo.append(sede_periodo_data)
                    # Calcular posición específica para esta sede en este período
                    position = base_position + sede_idx * (box_width + spacing)
                    positions.append(position)
                    colors_list.append(sede_color_map[sede])

                    # Agregar etiqueta solo en el primer período para la leyenda
                    if periodo_idx == 0:
                        labels_list.append(sede)

        # Crear boxplots
        bp = ax.boxplot(data_by_periodo, 
                       positions=positions,
                       widths=box_width,
                       patch_artist=True,
                       showfliers=True,
                       notch=False,
                       boxprops=dict(linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5),
                       medianprops=dict(linewidth=2, color='darkred'),
                       flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5))

        # Colorear las cajas
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Configurar eje X con los períodos
        # Calcular posiciones centrales para cada período
        periodo_positions = []
        for periodo_idx in range(len(periodos_labels)):
            base_position = periodo_idx * (total_width + 1)
            center_position = base_position + (total_width / 2) - (box_width / 2)
            periodo_positions.append(center_position)

        ax.set_xticks(periodo_positions)
        ax.set_xticklabels(periodos_labels, rotation=45, ha='right', fontsize=10)

        # Configuración del gráfico
        ax.set_xlabel('Año - Período', fontsize=14, fontweight='bold')
        ax.set_ylabel('Calificación Final', fontsize=14, fontweight='bold')
        ax.set_title('Distribución de Calificaciones Finales por Período y Sede (2021-2025)', 
                    fontsize=16, fontweight='bold', pad=20)

        # Agregar líneas de referencia para los niveles de desempeño
        ax.axhline(y=60, color='red', linestyle='--', linewidth=1, alpha=0.3, label='Bajo (60)')
        ax.axhline(y=70, color='orange', linestyle='--', linewidth=1, alpha=0.3, label='Básico (70)')
        ax.axhline(y=80, color='green', linestyle='--', linewidth=1, alpha=0.3, label='Alto (80)')
        ax.axhline(y=90, color='blue', linestyle='--', linewidth=1, alpha=0.3, label='Superior (90)')

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')

        # Crear leyenda personalizada para las sedes
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=sede_color_map[sede], alpha=0.7, label=sede) 
                          for sede in sedes_list]

        # Agregar leyenda en una posición que no interfiera
        ax.legend(handles=legend_elements, title='Sede', fontsize=11, 
                 title_fontsize=12, loc='upper right', framealpha=0.95)

        # Ajustar límites del eje Y
        ax.set_ylim([0, 105])

        # Ajustar layout
        plt.tight_layout()

        # Guardar
        output_path = f'{self.results_path}/grades_distribution_by_periodo_sede.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✅ Gráfico de distribución de calificaciones guardado: {output_path}")

        return output_path

    def plot_competencias_density_by_sede(self):
        """Genera gráficas de densidad para axiológico, cognitivo, procedimental y actitudinal por sede."""
        self.logger.info("Generando gráficas de densidad de competencias por sede...")

        sedes_list = self.results['sedes_list']

        # Usar helper para obtener competencias disponibles
        competencias_disponibles = self._get_competencias_disponibles()

        if not competencias_disponibles:
            self.logger.warning("No se encontraron columnas de competencias en el dataset")
            return None

        self.logger.info(f"Competencias encontradas: {', '.join(competencias_disponibles)}")

        # Obtener paleta de colores para las sedes
        sede_colors = self.get_beautiful_palette(len(sedes_list), palette_name='tab20b')
        sede_color_map = {sede: sede_colors[i] for i, sede in enumerate(sedes_list)}

        # Setup de subplots usando helper
        n_comps = len(competencias_disponibles)
        n_cols = 2
        fig, axes = self._setup_subplots(n_comps, n_cols=n_cols, figsize_per_item=(8, 6))

        # Crear un subplot por competencia
        for idx, comp in enumerate(competencias_disponibles):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # Graficar densidad para cada sede
            for sede_idx, sede in enumerate(sedes_list):
                sede_data = self.df[self.df['sede'] == sede][comp].dropna()

                if len(sede_data) > 0:
                    # Calcular estadísticas
                    media = sede_data.mean()
                    mediana = sede_data.median()

                    # Graficar densidad
                    sede_data.plot(kind='kde', ax=ax, color=sede_color_map[sede], 
                                  linewidth=2.5, alpha=0.8, label=f'{sede}')

                    # Agregar línea vertical para la media
                    ax.axvline(media, color=sede_color_map[sede], 
                              linestyle='--', linewidth=1.5, alpha=0.6)

            # Agregar tabla con estadísticas
            stats_text = []
            for sede in sedes_list:
                sede_data = self.df[self.df['sede'] == sede][comp].dropna()
                if len(sede_data) > 0:
                    media = sede_data.mean()
                    mediana = sede_data.median()
                    stats_text.append(f'{sede}: μ={media:.1f}  M={mediana:.1f}')

            # Colocar texto en la esquina superior izquierda
            text_content = '\n'.join(stats_text)
            ax.text(0.03, 0.97, text_content,
                   transform=ax.transAxes,
                   fontsize=9,
                   verticalalignment='top',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', 
                           facecolor='white', 
                           edgecolor='gray',
                           alpha=0.9))

            # Configuración del subplot
            ax.set_xlabel('Calificación', fontsize=12, fontweight='bold')
            ax.set_ylabel('Densidad', fontsize=12, fontweight='bold')
            ax.set_title(f'Competencia {self.COMPETENCIAS_NOMBRES.get(comp, comp)}', 
                        fontsize=14, fontweight='bold', pad=10)

            # Leyenda
            ax.legend(title='Sede', fontsize=10, title_fontsize=11,
                     loc='upper right', framealpha=0.95)

            # Grid
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

            # Límites del eje X
            ax.set_xlim([0, 100])

        # Ocultar subplots vacíos usando helper
        self._hide_empty_subplots(axes, n_comps, n_cols)

        # Título general
        fig.suptitle('Distribución de Densidad de Competencias por Sede', 
                    fontsize=18, fontweight='bold', y=0.995)

        # Ajustar layout
        plt.tight_layout()

        # Guardar
        output_path = f'{self.results_path}/competencias_density_by_sede.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✅ Gráfico guardado: {output_path}")

        # Guardar estadísticas de competencias por sede
        stats_comp_data = []
        for sede in sedes_list:
            sede_data = self.df[self.df['sede'] == sede]
            for comp in competencias_disponibles:
                comp_data = sede_data[comp].dropna()
                if len(comp_data) > 0:
                    stats_comp_data.append({
                        'Sede': sede,
                        'Competencia': self.COMPETENCIAS_NOMBRES.get(comp, comp),
                        'Media': comp_data.mean(),
                        'Mediana': comp_data.median(),
                        'Desv_Std': comp_data.std(),
                        'Min': comp_data.min(),
                        'Max': comp_data.max(),
                        'Q25': comp_data.quantile(0.25),
                        'Q75': comp_data.quantile(0.75)
                    })

        if stats_comp_data:
            stats_comp_df = pd.DataFrame(stats_comp_data)
            #stats_comp_df.to_csv(f'{self.results_path}/estadisticas_competencias_por_sede.csv', index=False)
            self.logger.info(f"✅ Estadísticas de competencias guardadas")

        return output_path

    def plot_heatmap_reprobacion_grado_sede(self):
        """Genera heatmap de proporción de reprobación (% estudiantes con nota < 60) por grado y sede."""
        self.logger.info("Generando heatmap de proporción de reprobación por grado y sede...")

        sedes_list = self.results['sedes_list']

        # Usar método helper para calcular reprobación
        heatmap_data = self.df.groupby(['grado', 'sede'])['resultado'].apply(self.calc_reprobacion).unstack(fill_value=0)
        heatmap_data = heatmap_data.sort_index()

        # Crear figura
        fig, ax = plt.subplots(figsize=(12, max(8, len(heatmap_data) * 0.4)))

        # Crear heatmap con escala de 0 a max (rojo = más reprobación)
        vmax = max(heatmap_data.values.max(), 10)  # Al menos hasta 10% para ver diferencias
        im = ax.imshow(heatmap_data.values, cmap='Reds', aspect='auto', vmin=0, vmax=vmax)

        # Configurar ejes
        ax.set_xticks(np.arange(len(heatmap_data.columns)))
        ax.set_yticks(np.arange(len(heatmap_data.index)))
        ax.set_xticklabels(heatmap_data.columns, fontsize=11)
        ax.set_yticklabels(heatmap_data.index, fontsize=10)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Agregar valores en cada celda
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                value = heatmap_data.iloc[i, j]
                text_color = 'white' if value > vmax/2 else 'black'
                text = ax.text(j, i, f'{value:.1f}%',
                             ha="center", va="center", color=text_color,
                             fontsize=9, fontweight='bold')

        # Etiquetas
        ax.set_xlabel('Sede', fontsize=13, fontweight='bold')
        ax.set_ylabel('Grado', fontsize=13, fontweight='bold')
        ax.set_title('Porcentaje de Reprobación por Grado y Sede\n(% Estudiantes con nota < 60)', 
                    fontsize=16, fontweight='bold', pad=20)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('% Reprobación', rotation=270, labelpad=25, fontsize=12, fontweight='bold')

        # Agregar líneas divisorias
        for i in range(len(heatmap_data.index) + 1):
            ax.axhline(i - 0.5, color='white', linewidth=1)
        for j in range(len(heatmap_data.columns) + 1):
            ax.axvline(j - 0.5, color='white', linewidth=1)

        plt.tight_layout()

        # Guardar
        output_path = f'{self.results_path}/heatmap_reprobacion_grado_sede.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✅ Heatmap de reprobación guardado: {output_path}")

        return output_path

    def plot_heatmap_reprobacion_asignatura_grado_sede(self):
        """Genera heatmap de proporción de reprobación (% estudiantes con nota < 60) por asignatura-grado y sede."""
        self.logger.info("Generando heatmap de proporción de reprobación por asignatura-grado y sede...")

        sedes_list = self.results['sedes_list']

        # Usar helper para cargar tabla de asignaturas
        asig_map = self._load_asignaturas_map()

        # Obtener asignaturas únicas del dataframe y mapear sus nombres
        asignaturas_unicas = self.df['asignatura'].unique()
        self.logger.info(f"Asignaturas únicas en el dataset: {sorted(asignaturas_unicas)}")

        # Crear columna con nombres mapeados en el dataframe original
        df_temp = self.df.copy()
        if asig_map:
            df_temp['asignatura_nombre'] = df_temp['asignatura'].apply(
                lambda x: asig_map.get(int(x) if isinstance(x, (int, float)) else x, str(x))
            )
            self.logger.info(f"Nombres mapeados: {df_temp['asignatura_nombre'].unique()[:10]}")
        else:
            df_temp['asignatura_nombre'] = df_temp['asignatura'].astype(str)

        # Calcular reprobación usando helper
        reprobacion_data = df_temp.groupby(['asignatura_nombre', 'grado', 'sede'])['resultado'].apply(self.calc_reprobacion).reset_index()
        reprobacion_data.columns = ['asignatura_nombre', 'grado', 'sede', 'reprobacion']

        # Crear columna combinada asignatura-grado
        reprobacion_data['asignatura_grado'] = reprobacion_data['asignatura_nombre'] + ' - ' + reprobacion_data['grado'].astype(str)

        # Contar registros por combinación asignatura-grado
        count_data = df_temp.groupby(['asignatura_nombre', 'grado']).size().reset_index()
        count_data.columns = ['asignatura_nombre', 'grado', 'count']
        count_data['asignatura_grado'] = count_data['asignatura_nombre'] + ' - ' + count_data['grado'].astype(str)

        # Filtrar combinaciones con al menos 20 registros en total
        combinaciones_validas = count_data[count_data['count'] >= 20]['asignatura_grado'].tolist()
        reprobacion_filtered = reprobacion_data[reprobacion_data['asignatura_grado'].isin(combinaciones_validas)]

        # Crear pivot table usando asignatura-grado
        heatmap_data = reprobacion_filtered.pivot(index='asignatura_grado', columns='sede', values='reprobacion')

        # Ordenar por promedio de reprobación (descendente para ver las peores primero)
        heatmap_data['_promedio'] = heatmap_data.mean(axis=1)
        heatmap_data = heatmap_data.sort_values('_promedio', ascending=False)
        heatmap_data = heatmap_data.drop('_promedio', axis=1)

        # Limitar a top 40 combinaciones con mayor reprobación
        if len(heatmap_data) > 40:
            self.logger.info(f"Mostrando top 40 combinaciones asignatura-grado con mayor reprobación de {len(heatmap_data)}")
            heatmap_data = heatmap_data.head(40)

        # Crear figura (ajustar altura según número de combinaciones)
        fig, ax = plt.subplots(figsize=(14, max(12, len(heatmap_data) * 0.35)))

        # Calcular escala
        valid_data = heatmap_data.values[~np.isnan(heatmap_data.values)]
        if len(valid_data) > 0:
            vmax = max(valid_data.max(), 10)  # Al menos hasta 10%
        else:
            vmax = 10

        # Crear heatmap con escala de 0 a max (rojo = más reprobación)
        im = ax.imshow(heatmap_data.values, cmap='Reds', aspect='auto', vmin=0, vmax=vmax)

        # Configurar ejes
        ax.set_xticks(np.arange(len(heatmap_data.columns)))
        ax.set_yticks(np.arange(len(heatmap_data.index)))
        ax.set_xticklabels(heatmap_data.columns, fontsize=12, fontweight='bold')
        ax.set_yticklabels(heatmap_data.index, fontsize=9)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Agregar valores en cada celda
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                value = heatmap_data.iloc[i, j]
                if not np.isnan(value):
                    text_color = 'white' if value > vmax/2 else 'black'
                    text = ax.text(j, i, f'{value:.1f}%',
                                 ha="center", va="center", color=text_color,
                                 fontsize=7, fontweight='bold')

        # Etiquetas
        ax.set_xlabel('Sede', fontsize=13, fontweight='bold')
        ax.set_ylabel('Asignatura - Grado', fontsize=13, fontweight='bold')

        # Título
        n_combinaciones_mostradas = len(heatmap_data)
        n_combinaciones_totales = len(combinaciones_validas)
        title = 'Porcentaje de Reprobación por Asignatura-Grado y Sede'
        subtitle = '(% Estudiantes con nota < 60)'
        if n_combinaciones_mostradas < n_combinaciones_totales:
            subtitle += f' - Top {n_combinaciones_mostradas} de {n_combinaciones_totales} combinaciones'

        ax.set_title(f'{title}\n{subtitle}', fontsize=15, fontweight='bold', pad=20)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('% Reprobación', rotation=270, labelpad=25, fontsize=12, fontweight='bold')

        # Agregar líneas divisorias
        for i in range(len(heatmap_data.index) + 1):
            ax.axhline(i - 0.5, color='white', linewidth=0.5)
        for j in range(len(heatmap_data.columns) + 1):
            ax.axvline(j - 0.5, color='white', linewidth=1)

        plt.tight_layout()

        # Guardar
        output_path = f'{self.results_path}/heatmap_reprobacion_asignatura_grado_sede.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✅ Heatmap de reprobación por asignatura-grado guardado: {output_path}")

        return output_path

    def plot_heatmap_excelencia_grado_sede(self):
        """Genera heatmap de proporción de excelencia (% estudiantes con nota ≥ 90) por grado y sede."""
        self.logger.info("Generando heatmap de proporción de excelencia por grado y sede...")

        sedes_list = self.results['sedes_list']

        # Calcular porcentaje de nivel superior por grado y sede
        def calc_excelencia(group):
            total = len(group)
            superiores = (group >= 90).sum()
            return (superiores / total * 100) if total > 0 else 0

        heatmap_data = self.df.groupby(['grado', 'sede'])['resultado'].apply(calc_excelencia).unstack(fill_value=0)
        heatmap_data = heatmap_data.sort_index()

        # Crear figura
        fig, ax = plt.subplots(figsize=(12, max(8, len(heatmap_data) * 0.4)))

        # Crear heatmap con escala de 0 a max (verde = más excelencia)
        vmax = max(heatmap_data.values.max(), 20)  # Al menos hasta 20% para ver diferencias
        im = ax.imshow(heatmap_data.values, cmap='Greens', aspect='auto', vmin=0, vmax=vmax)

        # Configurar ejes
        ax.set_xticks(np.arange(len(heatmap_data.columns)))
        ax.set_yticks(np.arange(len(heatmap_data.index)))
        ax.set_xticklabels(heatmap_data.columns, fontsize=11)
        ax.set_yticklabels(heatmap_data.index, fontsize=10)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Agregar valores en cada celda
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                value = heatmap_data.iloc[i, j]
                text_color = 'white' if value > vmax/2 else 'black'
                text = ax.text(j, i, f'{value:.1f}%',
                             ha="center", va="center", color=text_color,
                             fontsize=9, fontweight='bold')

        # Etiquetas
        ax.set_xlabel('Sede', fontsize=13, fontweight='bold')
        ax.set_ylabel('Grado', fontsize=13, fontweight='bold')
        ax.set_title('Porcentaje de Excelencia por Grado y Sede\n(% Estudiantes con nota ≥ 90)', 
                    fontsize=16, fontweight='bold', pad=20)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('% Nivel Superior', rotation=270, labelpad=25, fontsize=12, fontweight='bold')

        # Agregar líneas divisorias
        for i in range(len(heatmap_data.index) + 1):
            ax.axhline(i - 0.5, color='white', linewidth=1)
        for j in range(len(heatmap_data.columns) + 1):
            ax.axvline(j - 0.5, color='white', linewidth=1)

        plt.tight_layout()

        # Guardar
        output_path = f'{self.results_path}/heatmap_excelencia_grado_sede.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✅ Heatmap de excelencia guardado: {output_path}")

        return output_path

    def plot_asignaturas_evolution_by_sede(self):
        """Genera gráfico de líneas mostrando evolución temporal de asignaturas 1,2,3,4 por sede."""
        self.logger.info("Generando gráfico de evolución de asignaturas por sede...")

        sedes_list = self.results['sedes_list']

        # Asignaturas a analizar
        asignaturas_ids = [1, 2, 3, 4]

        # Usar helper para cargar asignaturas
        asig_map = self._load_asignaturas_map()

        # Mapear nombres de asignaturas
        asignaturas_nombres = {}
        for asig_id in asignaturas_ids:
            asignaturas_nombres[asig_id] = asig_map.get(asig_id, f"Asignatura {asig_id}")

        # Filtrar datos para las asignaturas seleccionadas
        df_filtered = self.df[self.df['asignatura'].isin(asignaturas_ids)].copy()

        if len(df_filtered) == 0:
            self.logger.warning("No se encontraron datos para las asignaturas 1, 2, 3, 4")
            return None

        # Setup de subplots usando helper
        n_sedes = len(sedes_list)
        n_cols = 2
        fig, axes = self._setup_subplots(n_sedes, n_cols=n_cols, figsize_per_item=(12, 7))

        # Crear un subplot por sede
        for idx, sede in enumerate(sedes_list):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # Filtrar datos para esta sede
            sede_df = df_filtered[df_filtered['sede'] == sede]

            if len(sede_df) == 0:
                ax.text(0.5, 0.5, f'Sin datos disponibles\npara {sede}', 
                       ha='center', va='center', fontsize=12, transform=ax.transAxes)
                ax.axis('off')
                continue

            # Obtener períodos únicos para esta sede SOLAMENTE
            periodos_sede_ordenados = sorted(sede_df[['año_num', 'periodo_num', 'año_periodo']].drop_duplicates().values.tolist())
            periodos_sede_labels = [p[2] for p in periodos_sede_ordenados]

            # Graficar línea para cada asignatura
            for asig_id in asignaturas_ids:
                asig_data = sede_df[sede_df['asignatura'] == asig_id]

                if len(asig_data) == 0:
                    continue

                # Calcular promedio por período
                promedios = asig_data.groupby('año_periodo')['resultado'].mean().reset_index()
                promedios = promedios.sort_values('año_periodo')

                # Obtener períodos disponibles para esta asignatura
                periodos_asig = promedios['año_periodo'].tolist()
                valores = promedios['resultado'].tolist()

                # Graficar línea
                ax.plot(periodos_asig, valores, 
                       marker='o', markersize=8, linewidth=2.5, 
                       color=self.ASIGNATURAS_COLORS[asig_id], alpha=0.8,
                       label=asignaturas_nombres[asig_id])

            # Configuración del subplot
            ax.set_xlabel('Año - Período', fontsize=11, fontweight='bold')
            ax.set_ylabel('Calificación Promedio', fontsize=11, fontweight='bold')
            ax.set_title(f'Sede: {sede}', fontsize=13, fontweight='bold', pad=10)

            # Configurar eje X con los períodos de esta sede
            ax.set_xticks(range(len(periodos_sede_labels)))
            ax.set_xticklabels(periodos_sede_labels, rotation=45, ha='right', fontsize=9)

            # Leyenda
            ax.legend(title='Asignatura', fontsize=9, title_fontsize=10,
                     loc='best', framealpha=0.95)

            # Grid horizontal
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

            # Límites del eje Y
            ax.set_ylim([0, 100])

            # Agregar líneas de referencia
            ax.axhline(y=60, color='red', linestyle=':', linewidth=1, alpha=0.3)
            ax.axhline(y=70, color='orange', linestyle=':', linewidth=1, alpha=0.3)
            ax.axhline(y=80, color='green', linestyle=':', linewidth=1, alpha=0.3)

        # Ocultar subplots vacíos usando helper
        self._hide_empty_subplots(axes, n_sedes, n_cols)

        # Título general
        fig.suptitle('Evolución Temporal de Calificaciones por Asignatura y Sede', 
                    fontsize=16, fontweight='bold', y=0.995)

        # Ajustar layout
        plt.tight_layout()

        # Guardar
        output_path = f'{self.results_path}/asignaturas_evolution_by_sede.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✅ Gráfico de evolución de asignaturas guardado: {output_path}")

        return output_path

    def plot_competencias_vs_resultado(self):
        """Genera scatter plots mostrando relación entre cada competencia y la nota final por sede."""
        self.logger.info("Generando scatter plots de competencias vs resultado final por sede...")

        sedes_list = self.results['sedes_list']

        # Usar helper para obtener competencias disponibles
        competencias_disponibles = self._get_competencias_disponibles()

        if not competencias_disponibles or 'resultado' not in self.df.columns:
            self.logger.warning("No se encontraron columnas de competencias o resultado en el dataset")
            return None

        self.logger.info(f"Competencias encontradas: {', '.join(competencias_disponibles)}")

        # Usar constantes de clase
        comp_colors = self.COMPETENCIAS_COLORS

        # Crear una figura por sede
        n_comps = len(competencias_disponibles)
        n_cols = 2

        # Procesar cada sede
        for sede in sedes_list:
            sede_df = self.df[self.df['sede'] == sede].copy()

            if len(sede_df) == 0:
                self.logger.warning(f"No hay datos para la sede {sede}")
                continue

            # Setup de subplots usando helper
            fig, axes = self._setup_subplots(n_comps, n_cols=n_cols, figsize_per_item=(8, 7))
            axes = axes.flatten()

            # Crear scatter plot para cada competencia
            for idx, comp in enumerate(competencias_disponibles):
                ax = axes[idx]

                # Filtrar datos sin valores nulos
                df_comp = sede_df[[comp, 'resultado']].dropna()

                if len(df_comp) == 0:
                    ax.text(0.5, 0.5, f'Sin datos disponibles\npara {self.COMPETENCIAS_NOMBRES[comp]}', 
                           ha='center', va='center', fontsize=12, transform=ax.transAxes)
                    ax.axis('off')
                    continue

                x = df_comp[comp].values
                y = df_comp['resultado'].values

                # Calcular correlación
                correlation = np.corrcoef(x, y)[0, 1]

                # Scatter plot con transparencia
                ax.scatter(x, y, alpha=0.3, s=30, color=comp_colors[comp], edgecolors='none')

                # Línea de tendencia (regresión lineal)
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), color='darkred', linewidth=3, 
                       linestyle='--', label=f'Tendencia (r={correlation:.3f})')

                # Región sombreada: Baja competencia pero alta nota final
                # Esto validaría tu hipótesis
                # Región: competencia < 65 Y resultado >= 70
                ax.axvspan(0, 65, ymin=0.7, ymax=1.0, alpha=0.15, color='yellow', 
                          label='Zona de interés:\nBaja comp. + Alta nota')
                ax.axhline(y=70, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
                ax.axvline(x=65, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)

                # Contar casos en la zona de interés (para todas las competencias)
                casos_especiales = len(df_comp[(df_comp[comp] < 65) & (df_comp['resultado'] >= 70)])
                total_casos = len(df_comp)
                porcentaje = (casos_especiales / total_casos * 100) if total_casos > 0 else 0

                # Agregar texto con el conteo
                ax.text(0.05, 0.95, 
                       f'Casos zona de interés:\n{casos_especiales} ({porcentaje:.1f}%)',
                       transform=ax.transAxes,
                       fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.5', 
                               facecolor='yellow', 
                               edgecolor='orange',
                               alpha=0.7))

                # Agregar estadísticas
                stats_text = f'n = {len(df_comp):,}\n'
                stats_text += f'Correlación: {correlation:.3f}\n'
                stats_text += f'Media {comp}: {x.mean():.1f}\n'
                stats_text += f'Media resultado: {y.mean():.1f}'

                ax.text(0.95, 0.05, stats_text,
                       transform=ax.transAxes,
                       fontsize=9,
                       verticalalignment='bottom',
                       horizontalalignment='right',
                       fontfamily='monospace',
                       bbox=dict(boxstyle='round,pad=0.5', 
                               facecolor='white', 
                               edgecolor='gray',
                               alpha=0.9))

                # Configuración del subplot
                ax.set_xlabel(f'Competencia {self.COMPETENCIAS_NOMBRES[comp]}', fontsize=12, fontweight='bold')
                ax.set_ylabel('Calificación Final (Resultado)', fontsize=12, fontweight='bold')
                ax.set_title(f'{self.COMPETENCIAS_NOMBRES[comp]} vs Resultado Final', 
                            fontsize=14, fontweight='bold', pad=10)

                # Leyenda
                ax.legend(fontsize=9, loc='upper left', framealpha=0.95)

                # Grid
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

                # Límites de los ejes
                ax.set_xlim([0, 100])
                ax.set_ylim([0, 100])

                # Líneas de referencia
                ax.axhline(y=60, color='red', linestyle=':', linewidth=1, alpha=0.2)
                ax.axvline(x=60, color='red', linestyle=':', linewidth=1, alpha=0.2)

            # Ocultar subplots vacíos (axes ya está flatten)
            for idx in range(n_comps, len(axes)):
                axes[idx].axis('off')

            # Título general
            fig.suptitle(f'Relación entre Competencias y Calificación Final - Sede: {sede}\n(Validación de compensación entre competencias)', 
                        fontsize=16, fontweight='bold', y=0.995)

            # Ajustar layout
            plt.tight_layout()

            # Guardar
            output_path = f'{self.results_path}/competencias_vs_resultado_scatter_{sede}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"✅ Gráfico de scatter para {sede} guardado: {output_path}")

        # Guardar análisis cuantitativo de casos especiales por sede
        casos_especiales_data = []
        for sede in sedes_list:
            sede_df = self.df[self.df['sede'] == sede]
            for comp in competencias_disponibles:
                df_comp = sede_df[[comp, 'resultado']].dropna()
                if len(df_comp) > 0:
                    casos = df_comp[(df_comp[comp] < 65) & (df_comp['resultado'] >= 70)]
                    casos_especiales_data.append({
                        'Sede': sede,
                        'Competencia': self.COMPETENCIAS_NOMBRES[comp],
                        'Casos_Baja_Comp_Alta_Nota': len(casos),
                        'Total_Casos': len(df_comp),
                        'Porcentaje': (len(casos) / len(df_comp) * 100) if len(df_comp) > 0 else 0,
                        'Correlacion': np.corrcoef(df_comp[comp].values, df_comp['resultado'].values)[0, 1]
                    })

        if casos_especiales_data:
            casos_df = pd.DataFrame(casos_especiales_data)
            #casos_df.to_csv(f'{self.results_path}/analisis_compensacion_competencias_por_sede.csv', index=False)
            self.logger.info(f"✅ Análisis de compensación por sede guardado")

        return f'{self.results_path}/competencias_vs_resultado_scatter_*.png'

    def plot_barras_niveles_grado_sede(self):
        """Genera gráfico de barras apiladas mostrando distribución de niveles por grado y sede."""
        self.logger.info("Generando gráfico de barras de distribución de niveles...")

        sedes_list = self.results['sedes_list']

        # Clasificar niveles usando método helper
        self.df['nivel'] = self.df['resultado'].apply(self.classify_grade)

        # Usar constantes de clase (excluyendo 'Sin Datos' para este gráfico)
        colors = {k: v for k, v in self.PERFORMANCE_COLORS.items() if k != 'Sin Datos'}
        niveles_orden = [n for n in self.PERFORMANCE_LEVELS if n != 'Sin Datos']

        # Crear subplots por sede
        n_sedes = len(sedes_list)
        fig, axes = plt.subplots(1, n_sedes, figsize=(8*n_sedes, 8), sharey=True)

        if n_sedes == 1:
            axes = [axes]

        for sede_idx, sede in enumerate(sedes_list):
            ax = axes[sede_idx]
            sede_data = self.df[self.df['sede'] == sede]

            # Obtener grados únicos ordenados SOLO para esta sede
            grados_sede = sorted(sede_data['grado'].unique())

            # Calcular porcentajes por grado
            data_by_grado = []
            for grado in grados_sede:
                grado_data = sede_data[sede_data['grado'] == grado]
                total = len(grado_data)

                if total > 0:
                    percentages = {}
                    for nivel in niveles_orden:
                        count = (grado_data['nivel'] == nivel).sum()
                        percentages[nivel] = (count / total * 100)
                    data_by_grado.append(percentages)
                else:
                    data_by_grado.append({nivel: 0 for nivel in niveles_orden})

            # Crear barras apiladas
            x_pos = np.arange(len(grados_sede))
            bottom = np.zeros(len(grados_sede))

            for nivel in niveles_orden:
                values = [data_by_grado[i][nivel] for i in range(len(grados_sede))]
                bars = ax.bar(x_pos, values, bottom=bottom, label=nivel,
                             color=colors[nivel], edgecolor='white', linewidth=0.5, alpha=0.9)

                # Agregar porcentajes si es > 3%
                for i, (bar, value) in enumerate(zip(bars, values)):
                    if value > 3:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., bottom[i] + height/2.,
                               f'{value:.0f}%',
                               ha='center', va='center', fontsize=8,
                               fontweight='bold', color='white')

                bottom += values

            # Configuración
            ax.set_xlabel('Grado', fontsize=12, fontweight='bold')
            if sede_idx == 0:
                ax.set_ylabel('Porcentaje de Estudiantes (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'Sede: {sede}', fontsize=14, fontweight='bold', pad=10)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(grados_sede, rotation=45, ha='right')
            ax.set_ylim([0, 100])
            ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)

            if sede_idx == 0:
                ax.legend(title='Nivel de Desempeño', fontsize=10, title_fontsize=11,
                         loc='upper left', framealpha=0.95)

        fig.suptitle('Distribución de Niveles de Desempeño por Grado y Sede', 
                    fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()

        # Guardar
        output_path = f'{self.results_path}/barras_niveles_grado_sede.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✅ Gráfico de barras guardado: {output_path}")

        return output_path

    def generate_summary_statistics(self):
        """Genera estadísticas resumen del análisis."""
        self.logger.info("Generando estadísticas resumen...")

        # Estadísticas por sede
        stats_by_sede = self.df.groupby('sede').agg({
            'identificación': 'nunique',
            'año_periodo': 'nunique',
            'resultado': 'mean',
            'grado': lambda x: x.nunique()
        }).reset_index()

        stats_by_sede.columns = ['Sede', 'Total Estudiantes', 'Períodos', 'Promedio Resultado', 'Grados']

        # Estadísticas por período
        stats_by_periodo = self.df.groupby('año_periodo').agg({
            'identificación': 'nunique',
            'resultado': 'mean',
            'sede': 'nunique'
        }).reset_index()

        stats_by_periodo.columns = ['Año-Período', 'Total Estudiantes', 'Promedio Resultado', 'Sedes Activas']

        # Escribir archivos CSV
        #stats_by_sede.to_csv(f'{self.results_path}/estadisticas_por_sede.csv', index=False)
        #stats_by_periodo.to_csv(f'{self.results_path}/estadisticas_por_periodo.csv', index=False)

        self.logger.info(f"✅ Estadísticas guardadas en: {self.results_path}/")

        self.results['stats_by_sede'] = stats_by_sede
        self.results['stats_by_periodo'] = stats_by_periodo

        return stats_by_sede, stats_by_periodo

    def run_analysis(self):
        """Ejecuta el pipeline completo de análisis de distribución de calificaciones."""
        self.logger.info("=" * 60)
        self.logger.info("INICIANDO ANÁLISIS DE DISTRIBUCIÓN DE CALIFICACIONES")
        self.logger.info("=" * 60)

        try:
            # Crear directorio de resultados
            self.create_results_directory()

            # 1. Cargar datos
            self.load_grades_data()

            # 2. Crear columna año-periodo
            self.create_periodo_column()

            # 3. Preparar datos para análisis
            total_students = self.df['identificación'].nunique()
            total_sedes = self.df['sede'].nunique()
            sedes_list = sorted(self.df['sede'].unique().tolist())

            self.logger.info(f"Total de estudiantes únicos: {total_students:,}")
            self.logger.info(f"Total de sedes: {total_sedes}")
            self.logger.info(f"Sedes: {', '.join(sedes_list)}")

            self.results['total_students'] = total_students
            self.results['total_sedes'] = total_sedes
            self.results['sedes_list'] = sedes_list

            # 4. Generar gráficos
            self.logger.info("Generando gráficos...")

            # 4a. Niveles de desempeño por período y sede
            self.plot_performance_levels_by_period()

            # 4b. Distribución de calificaciones por período y sede
            self.plot_grades_distribution_by_periodo_sede()

            # 4c. Densidad de competencias por sede
            self.plot_competencias_density_by_sede()

            # 4d. Heatmap de proporción de reprobación por grado y sede
            self.plot_heatmap_reprobacion_grado_sede()

            # 4e. Heatmap de proporción de reprobación por asignatura-grado y sede
            self.plot_heatmap_reprobacion_asignatura_grado_sede()

            # 4f. Heatmap de proporción de excelencia por grado y sede
            self.plot_heatmap_excelencia_grado_sede()

            # 4g. Barras apiladas de niveles por grado y sede
            self.plot_barras_niveles_grado_sede()

            # 4h. Evolución temporal de asignaturas 1,2,3,4 por sede
            self.plot_asignaturas_evolution_by_sede()

            # 4i. Scatter plots de competencias vs resultado (validación de compensación)
            self.plot_competencias_vs_resultado()

            # 5. Generar estadísticas resumen
            self.logger.info("Generando estadísticas resumen...")
            self.generate_summary_statistics()

            self.logger.info("=" * 60)
            self.logger.info("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
            self.logger.info("=" * 60)

            return self.results

        except Exception as e:
            self.logger.error(f"❌ Error en análisis: {e}")
            raise


def main():
    """Función principal."""
    import argparse

    parser = argparse.ArgumentParser(description='Análisis de calificaciones históricas')
    parser.add_argument('--dataset', '-d', type=str, 
                       default='data/interim/calificaciones/calificaciones_2021-2025.csv',
                       help='Ruta al archivo CSV de calificaciones')
    parser.add_argument('--results', '-r', type=str, 
                       default='grades_analysis',
                       help='Nombre del folder para guardar resultados')

    args = parser.parse_args()

    # Crear y ejecutar analizador
    analyzer = GradesAnalysis(args.dataset, args.results)

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
