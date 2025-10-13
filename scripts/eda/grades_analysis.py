"""Script para análisis de calificaciones históricas 2021-2025."""

import os
import sys
import pandas as pd
import numpy as np
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
import matplotlib
matplotlib.set_loglevel("WARNING")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.eda_analysis_base import EDAAnalysisBase


class GradesAnalysis(EDAAnalysisBase):
    """Analizador de calificaciones históricas."""

    def _initialize_analysis_attributes(self):
        """Inicializar atributos específicos del análisis de calificaciones."""
        self.df = None
        self.results = {}

    def load_grades_data(self):
        """Carga los datos de calificaciones."""
        self.logger.info(f"Cargando datos de calificaciones desde: {self.dataset_path}")
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"No se encontró el dataset en: {self.dataset_path}")
        
        df = pd.read_csv(self.dataset_path)
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

    def analyze_students_by_sede_periodo(self):
        """Analiza el número de estudiantes únicos por sede y período."""
        self.logger.info("Analizando estudiantes por sede y período...")
        
        # Agrupar por sede y año-periodo, contar estudiantes únicos
        students_by_sede_periodo = self.df.groupby(['sede', 'año_periodo'])['identificación'].nunique().reset_index()
        students_by_sede_periodo.columns = ['sede', 'año_periodo', 'num_estudiantes']
        
        # Información general
        total_students = self.df['identificación'].nunique()
        total_sedes = self.df['sede'].nunique()
        sedes_list = sorted(self.df['sede'].unique())
        
        self.logger.info(f"Total de estudiantes únicos: {total_students:,}")
        self.logger.info(f"Total de sedes: {total_sedes}")
        self.logger.info(f"Sedes: {', '.join(sedes_list)}")
        
        self.results['students_by_sede_periodo'] = students_by_sede_periodo
        self.results['total_students'] = total_students
        self.results['total_sedes'] = total_sedes
        self.results['sedes_list'] = sedes_list
        
        return students_by_sede_periodo

    def analyze_students_by_grade_sede_periodo(self):
        """Analiza el número de estudiantes únicos por grado, sede y período."""
        self.logger.info("Analizando estudiantes por grado, sede y período...")
        
        # Agrupar por grado, sede y año-periodo, contar estudiantes únicos
        students_by_grade = self.df.groupby(['grado', 'sede', 'año_periodo'])['identificación'].nunique().reset_index()
        students_by_grade.columns = ['grado', 'sede', 'año_periodo', 'num_estudiantes']
        
        # Información sobre grados
        grados_list = sorted(self.df['grado'].unique())
        total_grados = len(grados_list)
        
        self.logger.info(f"Total de grados analizados: {total_grados}")
        self.logger.info(f"Grados: {', '.join(map(str, grados_list))}")
        
        self.results['students_by_grade'] = students_by_grade
        self.results['grados_list'] = grados_list
        self.results['total_grados'] = total_grados
        
        return students_by_grade

    def plot_students_evolution(self):
        """Genera gráfico de líneas mostrando la evolución de estudiantes por sede."""
        self.logger.info("Generando gráfico de evolución de estudiantes por sede...")
        
        students_data = self.results['students_by_sede_periodo']
        sedes_list = self.results['sedes_list']
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Obtener paleta de colores
        colors = self.get_beautiful_palette(len(sedes_list), palette_name='tab10')
        
        # Obtener todos los períodos únicos ordenados
        periodos_ordenados = sorted(self.df[['año_num', 'periodo_num', 'año_periodo']].drop_duplicates().values.tolist())
        periodos_labels = [p[2] for p in periodos_ordenados]
        
        # Graficar línea para cada sede
        for idx, sede in enumerate(sedes_list):
            sede_data = students_data[students_data['sede'] == sede].copy()
            
            # Asegurar que tenemos datos para todos los períodos (rellenar con 0 si falta)
            all_periodos = pd.DataFrame({'año_periodo': periodos_labels})
            sede_data = all_periodos.merge(sede_data, on='año_periodo', how='left')
            sede_data['num_estudiantes'] = sede_data['num_estudiantes'].fillna(0)
            
            # Graficar
            ax.plot(sede_data['año_periodo'], sede_data['num_estudiantes'], 
                   marker='o', linewidth=2.5, markersize=8, 
                   label=sede, color=colors[idx], alpha=0.8)
        
        # Configuración del gráfico
        ax.set_xlabel('Año - Período', fontsize=14, fontweight='bold')
        ax.set_ylabel('Número de Estudiantes', fontsize=14, fontweight='bold')
        ax.set_title('Evolución de Estudiantes por Sede (2021-2025)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Rotar etiquetas del eje x para mejor legibilidad
        plt.xticks(rotation=45, ha='right')
        
        # Leyenda
        ax.legend(title='Sede', fontsize=11, title_fontsize=12, 
                 loc='best', framealpha=0.9)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar
        output_path = f'{self.results_path}/students_evolution_by_sede.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Gráfico guardado: {output_path}")
        
        return output_path

    def plot_students_evolution_by_grade(self):
        """Genera gráficos de líneas mostrando la evolución de estudiantes por grado y sede."""
        self.logger.info("Generando gráficos de evolución de estudiantes por grado y sede...")
        
        students_by_grade = self.results['students_by_grade']
        grados_list = self.results['grados_list']
        sedes_list = self.results['sedes_list']
        
        # Obtener todos los períodos únicos ordenados
        periodos_ordenados = sorted(self.df[['año_num', 'periodo_num', 'año_periodo']].drop_duplicates().values.tolist())
        periodos_labels = [p[2] for p in periodos_ordenados]
        
        # Obtener paleta de colores para las sedes
        colors = self.get_beautiful_palette(len(sedes_list), palette_name='tab10')
        
        # Calcular layout de subplots (máximo 3 columnas)
        n_grados = len(grados_list)
        n_cols = min(3, n_grados)
        n_rows = (n_grados + n_cols - 1) // n_cols
        
        # Crear figura con subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
        
        # Asegurar que axes sea siempre un array 2D
        if n_grados == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Graficar cada grado
        for idx, grado in enumerate(grados_list):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            # Filtrar datos para este grado
            grade_data = students_by_grade[students_by_grade['grado'] == grado]
            
            # Graficar línea para cada sede
            for sede_idx, sede in enumerate(sedes_list):
                sede_grade_data = grade_data[grade_data['sede'] == sede].copy()
                
                # Asegurar que tenemos datos para todos los períodos
                all_periodos = pd.DataFrame({'año_periodo': periodos_labels})
                sede_grade_data = all_periodos.merge(sede_grade_data, on='año_periodo', how='left')
                sede_grade_data['num_estudiantes'] = sede_grade_data['num_estudiantes'].fillna(0)
                
                # Graficar
                ax.plot(sede_grade_data['año_periodo'], sede_grade_data['num_estudiantes'], 
                       marker='o', linewidth=2, markersize=6, 
                       label=sede, color=colors[sede_idx], alpha=0.8)
            
            # Configuración del subplot
            ax.set_xlabel('Año - Período', fontsize=11, fontweight='bold')
            ax.set_ylabel('Número de Estudiantes', fontsize=11, fontweight='bold')
            ax.set_title(f'Grado {grado}', fontsize=13, fontweight='bold', pad=10)
            
            # Rotar etiquetas del eje x
            ax.tick_params(axis='x', rotation=45)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
            
            # Leyenda
            ax.legend(title='Sede', fontsize=9, title_fontsize=10, 
                     loc='best', framealpha=0.9)
            
            # Grid
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Ocultar subplots vacíos si los hay
        total_subplots = n_rows * n_cols
        for idx in range(n_grados, total_subplots):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        # Título general
        fig.suptitle('Evolución de Estudiantes por Grado y Sede (2021-2025)', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar
        output_path = f'{self.results_path}/students_evolution_by_grade_and_sede.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Gráfico por grados guardado: {output_path}")
        
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
        sede_colors = self.get_beautiful_palette(len(sedes_list), palette_name='Set2')
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
        
        # Guardar estadísticas
        stats_output = f'{self.results_path}/estadisticas_resumen.txt'
        with open(stats_output, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ANÁLISIS DE CALIFICACIONES HISTÓRICAS 2021-2025\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total de registros: {len(self.df):,}\n")
            f.write(f"Total de estudiantes únicos: {self.results['total_students']:,}\n")
            f.write(f"Total de sedes: {self.results['total_sedes']}\n")
            f.write(f"Sedes: {', '.join(self.results['sedes_list'])}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("ESTADÍSTICAS POR SEDE\n")
            f.write("-" * 80 + "\n")
            f.write(stats_by_sede.to_string(index=False))
            f.write("\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("ESTADÍSTICAS POR PERÍODO\n")
            f.write("-" * 80 + "\n")
            f.write(stats_by_periodo.to_string(index=False))
            f.write("\n\n")
        
        # Guardar también como CSV
        stats_by_sede.to_csv(f'{self.results_path}/estadisticas_por_sede.csv', index=False)
        stats_by_periodo.to_csv(f'{self.results_path}/estadisticas_por_periodo.csv', index=False)
        
        self.logger.info(f"✅ Estadísticas guardadas en: {self.results_path}/")
        
        self.results['stats_by_sede'] = stats_by_sede
        self.results['stats_by_periodo'] = stats_by_periodo
        
        return stats_by_sede, stats_by_periodo

    def run_analysis(self):
        """Ejecuta el pipeline completo de análisis."""
        self.logger.info("=" * 60)
        self.logger.info("INICIANDO ANÁLISIS DE CALIFICACIONES HISTÓRICAS")
        self.logger.info("=" * 60)
        
        try:
            # Crear directorio de resultados
            self.create_results_directory()
            
            # 1. Cargar datos
            self.load_grades_data()
            
            # 2. Crear columna año-periodo
            self.create_periodo_column()
            
            # 3. Analizar estudiantes por sede y período
            self.analyze_students_by_sede_periodo()
            
            # 4. Analizar estudiantes por grado, sede y período
            self.analyze_students_by_grade_sede_periodo()
            
            # 5. Generar gráfico de evolución por sede
            self.plot_students_evolution()
            
            # 6. Generar gráfico de evolución por grado y sede
            self.plot_students_evolution_by_grade()
            
            # 7. Generar boxplots de distribución de calificaciones por período y sede
            self.plot_grades_distribution_by_periodo_sede()
            
            # 8. Generar estadísticas resumen
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

