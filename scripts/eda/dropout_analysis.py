"""Script para análisis de retiro escolar - Identificar patrones en estudiantes retirados."""

import os
import sys
import pandas as pd
import numpy as np
import warnings
import logging

# Configurar matplotlib ANTES de importar pyplot
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo por defecto
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['font.size'] = 10

# Configuración
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.eda_analysis_base import EDAAnalysisBase


class DropoutAnalysis(EDAAnalysisBase):
    """Analizador de retiro escolar."""

    def _initialize_analysis_attributes(self):
        """Inicializar atributos específicos del análisis."""
        self.df = None
        self.results = {}
        self.dropout_students = None
        self.active_students = None

    def load_grades_data(self):
        """Carga los datos de calificaciones."""
        self.logger.info(f"Cargando datos desde: {self.dataset_path}")

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"No se encontró el dataset en: {self.dataset_path}")

        df = pd.read_csv(self.dataset_path, 
                        dtype={'año': 'int16', 'periodo': 'int8'},
                        low_memory=False)
        self.logger.info(f"Dataset cargado: {df.shape[0]:,} registros, {df.shape[1]} columnas")

        # Validar columnas necesarias
        required_columns = ['sede', 'año', 'periodo', 'identificación', 'resultado']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Columnas requeridas faltantes: {missing_columns}")

        # Crear columna año-periodo
        df['año_periodo'] = df['año'].astype(str) + '-' + df['periodo'].astype(str)
        df['año_num'] = df['año'].astype(int)
        df['periodo_num'] = df['periodo'].astype(int)
        df = df.sort_values(['año_num', 'periodo_num'])

        self.df = df
        return df

    def identify_dropout_students(self):
        """Identifica estudiantes que se retiraron."""
        self.logger.info("Identificando estudiantes retirados...")

        # Crear columna año_periodo si no existe
        if 'año_periodo' not in self.df.columns:
            self.df['año_periodo'] = self.df['año'].astype(str) + '-' + self.df['periodo'].astype(str)

        # Obtener todos los períodos únicos ordenados
        periodos_ordenados = sorted(self.df[['año', 'periodo']].drop_duplicates().values.tolist())
        all_periodos = [f"{año}-{periodo}" for año, periodo in periodos_ordenados]

        # Para cada estudiante, identificar su último período activo
        student_last_periodo = (
            self.df.groupby('identificación')
            .agg({
                'año': 'max',
                'periodo': 'max',
                'año_periodo': 'last',
                'sede': 'last',
                'grado': 'last'
            })
            .reset_index()
        )

        # El último período del dataset
        last_periodo_year = max(periodos_ordenados, key=lambda x: (x[0], x[1]))[0]
        last_periodo_num = max(periodos_ordenados, key=lambda x: (x[0], x[1]))[1]

        # Estudiantes que NO estuvieron en el último período son considerados retirados
        # (o que su último período fue hace más de 1 año)
        student_last_periodo['es_retirado'] = (
            (student_last_periodo['año'] < last_periodo_year - 1) |
            ((student_last_periodo['año'] == last_periodo_year - 1) & 
             (student_last_periodo['periodo'] < 2))
        )

        # Separar retirados y activos
        dropout_ids = student_last_periodo[student_last_periodo['es_retirado']]['identificación'].tolist()
        active_ids = student_last_periodo[~student_last_periodo['es_retirado']]['identificación'].tolist()

        self.logger.info(f"Total de estudiantes únicos: {len(student_last_periodo):,}")
        self.logger.info(f"Estudiantes retirados: {len(dropout_ids):,} ({len(dropout_ids)/len(student_last_periodo)*100:.1f}%)")
        self.logger.info(f"Estudiantes activos: {len(active_ids):,} ({len(active_ids)/len(student_last_periodo)*100:.1f}%)")

        # Guardar información
        self.results['student_last_periodo'] = student_last_periodo
        self.results['dropout_ids'] = dropout_ids
        self.results['active_ids'] = active_ids
        self.results['total_students'] = len(student_last_periodo)
        self.results['total_dropout'] = len(dropout_ids)
        self.results['total_active'] = len(active_ids)
        self.results['dropout_rate'] = len(dropout_ids) / len(student_last_periodo) * 100

        # Asignar a las variables de instancia
        self.dropout_students = dropout_ids
        self.active_students = active_ids

        return dropout_ids, active_ids

    def analyze_dropout_grades(self):
        """Analiza las calificaciones de estudiantes retirados vs activos."""
        self.logger.info("Analizando calificaciones de retirados vs activos...")

        dropout_ids = self.results['dropout_ids']
        active_ids = self.results['active_ids']

        # Filtrar datos
        df_dropout = self.df[self.df['identificación'].isin(dropout_ids)].copy()
        df_active = self.df[self.df['identificación'].isin(active_ids)].copy()

        # Calcular estadísticas de calificaciones
        dropout_grades_stats = {
            'promedio': df_dropout['resultado'].mean(),
            'mediana': df_dropout['resultado'].median(),
            'desv_std': df_dropout['resultado'].std(),
            'min': df_dropout['resultado'].min(),
            'max': df_dropout['resultado'].max(),
            'q25': df_dropout['resultado'].quantile(0.25),
            'q75': df_dropout['resultado'].quantile(0.75)
        }

        active_grades_stats = {
            'promedio': df_active['resultado'].mean(),
            'mediana': df_active['resultado'].median(),
            'desv_std': df_active['resultado'].std(),
            'min': df_active['resultado'].min(),
            'max': df_active['resultado'].max(),
            'q25': df_active['resultado'].quantile(0.25),
            'q75': df_active['resultado'].quantile(0.75)
        }

        self.logger.info(f"Promedio calificaciones RETIRADOS: {dropout_grades_stats['promedio']:.2f}")
        self.logger.info(f"Promedio calificaciones ACTIVOS: {active_grades_stats['promedio']:.2f}")
        self.logger.info(f"Diferencia: {abs(dropout_grades_stats['promedio'] - active_grades_stats['promedio']):.2f} puntos")

        self.results['dropout_grades_stats'] = dropout_grades_stats
        self.results['active_grades_stats'] = active_grades_stats
        self.results['df_dropout'] = df_dropout
        self.results['df_active'] = df_active

        return dropout_grades_stats, active_grades_stats

    def analyze_dropout_by_demographics(self):
        """Analiza retiro por demografía (sede, grado)."""
        self.logger.info("Analizando retiro por demografía...")

        student_info = self.results['student_last_periodo']

        # Retiro por sede
        dropout_by_sede = (
            student_info.groupby('sede')['es_retirado']
            .agg(['sum', 'count'])
            .reset_index()
        )
        dropout_by_sede.columns = ['sede', 'retirados', 'total']
        dropout_by_sede['proporcion_retiro'] = (dropout_by_sede['retirados'] / dropout_by_sede['total'] * 100)
        dropout_by_sede = dropout_by_sede.sort_values('proporcion_retiro', ascending=False)

        # Retiro por grado
        dropout_by_grade = (
            student_info.groupby('grado')['es_retirado']
            .agg(['sum', 'count'])
            .reset_index()
        )
        dropout_by_grade.columns = ['grado', 'retirados', 'total']
        dropout_by_grade['proporcion_retiro'] = (dropout_by_grade['retirados'] / dropout_by_grade['total'] * 100)
        dropout_by_grade = dropout_by_grade.sort_values('grado')

        self.logger.info("\nProporción de retiro por sede:")
        for _, row in dropout_by_sede.iterrows():
            self.logger.info(f"  {row['sede']}: {row['proporcion_retiro']:.1f}% ({int(row['retirados'])}/{int(row['total'])})")

        self.logger.info("\nProporción de retiro por grado:")
        for _, row in dropout_by_grade.iterrows():
            self.logger.info(f"  Grado {row['grado']}: {row['proporcion_retiro']:.1f}% ({int(row['retirados'])}/{int(row['total'])})")

        self.results['dropout_by_sede'] = dropout_by_sede
        self.results['dropout_by_grade'] = dropout_by_grade

        return dropout_by_sede, dropout_by_grade

    def analyze_dropout_antiquity(self):
        """Analiza la antigüedad (períodos) de los estudiantes al momento del retiro."""
        self.logger.info("Analizando antigüedad al momento del retiro...")

        student_info = self.results['student_last_periodo']
        
        # Obtener todos los períodos únicos ordenados
        periodos_ordenados = sorted(self.df[['año', 'periodo']].drop_duplicates().values.tolist())
        all_periodos = [f"{año}-{periodo}" for año, periodo in periodos_ordenados]
        
        # Calcular antigüedad para cada estudiante
        def calculate_antiquity(row):
            """Calcula cuántos períodos llevaba un estudiante antes de retirarse."""
            student_id = row['identificación']
            last_year = row['año']
            last_period = row['periodo']
            
            # Encontrar el índice del último período del estudiante
            last_periodo_index = None
            for i, (year, period) in enumerate(periodos_ordenados):
                if year == last_year and period == last_period:
                    last_periodo_index = i
                    break
            
            if last_periodo_index is None:
                return 0
            
            # La antigüedad es el número de períodos que estuvo activo
            return last_periodo_index + 1
        
        # Aplicar cálculo de antigüedad
        student_info['antiguedad_periodos'] = student_info.apply(calculate_antiquity, axis=1)
        
        # Filtrar solo estudiantes retirados
        dropout_students = student_info[student_info['es_retirado']].copy()
        
        if len(dropout_students) == 0:
            self.logger.warning("No hay estudiantes retirados para analizar antigüedad")
            return None, None
        
        # Estadísticas de antigüedad
        antiquity_stats = {
            'total_retirados': len(dropout_students),
            'antiguedad_promedio': dropout_students['antiguedad_periodos'].mean(),
            'antiguedad_mediana': dropout_students['antiguedad_periodos'].median(),
            'antiguedad_min': dropout_students['antiguedad_periodos'].min(),
            'antiguedad_max': dropout_students['antiguedad_periodos'].max()
        }
        
        # Crear rangos de antigüedad para análisis
        antiquity_ranges = [
            (1, 2, "1-2 períodos"),
            (3, 4, "3-4 períodos"), 
            (5, 6, "5-6 períodos"),
            (7, 8, "7-8 períodos"),
            (9, 10, "9-10 períodos"),
            (11, 20, "11+ períodos")
        ]
        
        # Calcular proporción de retiro por rango de antigüedad
        range_analysis = []
        for min_periods, max_periods, label in antiquity_ranges:
            # Estudiantes en este rango de antigüedad que se retiraron
            dropout_in_range = dropout_students[
                (dropout_students['antiguedad_periodos'] >= min_periods) & 
                (dropout_students['antiguedad_periodos'] <= max_periods)
            ]
            
            # Total de estudiantes que alguna vez estuvieron en este rango
            # (esto es una aproximación - estudiantes que llegaron al período mínimo)
            total_in_range = len(student_info[student_info['antiguedad_periodos'] >= min_periods])
            
            if total_in_range > 0:
                dropout_rate = len(dropout_in_range) / total_in_range * 100
            else:
                dropout_rate = 0
                
            range_analysis.append({
                'rango': label,
                'min_periods': min_periods,
                'max_periods': max_periods,
                'retirados': len(dropout_in_range),
                'total_en_rango': total_in_range,
                'proporcion_retiro': dropout_rate
            })
        
        range_df = pd.DataFrame(range_analysis)
        
        # Log de resultados
        self.logger.info(f"\nEstadísticas de antigüedad al retiro:")
        self.logger.info(f"  Total retirados: {antiquity_stats['total_retirados']}")
        self.logger.info(f"  Antigüedad promedio: {antiquity_stats['antiguedad_promedio']:.1f} períodos")
        self.logger.info(f"  Antigüedad mediana: {antiquity_stats['antiguedad_mediana']:.1f} períodos")
        self.logger.info(f"  Rango: {antiquity_stats['antiguedad_min']}-{antiquity_stats['antiguedad_max']} períodos")
        
        self.logger.info(f"\nProporción de retiro por antigüedad:")
        for _, row in range_df.iterrows():
            self.logger.info(f"  {row['rango']}: {row['proporcion_retiro']:.1f}% ({row['retirados']}/{row['total_en_rango']})")
        
        # Guardar resultados
        self.results['dropout_antiquity'] = dropout_students
        self.results['antiquity_stats'] = antiquity_stats
        self.results['antiquity_ranges'] = range_df
        
        return dropout_students, range_df

    def analyze_dropout_by_period(self):
        """Analiza en qué períodos académicos se retiran más estudiantes."""
        self.logger.info("Analizando retiro por período académico...")

        if self.dropout_students is None or len(self.dropout_students) == 0:
            self.logger.warning("No hay estudiantes retirados para analizar por período")
            return pd.DataFrame()

        # Para cada retirado, encontrar su último período registrado
        dropout_periods = []

        for student_id in self.dropout_students:
            student_data = self.df[self.df['identificación'] == student_id].copy()
            if len(student_data) > 0:
                # Ordenar por año y período para encontrar el último registro
                student_data = student_data.sort_values(['año', 'periodo'])
                last_record = student_data.iloc[-1]

                dropout_periods.append({
                    'identificación': student_id,
                    'ultimo_año': last_record['año'],
                    'ultimo_periodo': last_record['periodo'],
                    'año_periodo': f"{last_record['año']}-{last_record['periodo']}",
                    'sede': last_record['sede'],
                    'grado': last_record['grado']
                })

        dropout_periods_df = pd.DataFrame(dropout_periods)

        if len(dropout_periods_df) == 0:
            self.logger.warning("No se pudieron determinar períodos de retiro")
            return pd.DataFrame()

        # Análisis por año-período
        dropout_by_period = dropout_periods_df.groupby('año_periodo').agg({
            'identificación': 'count',
            'sede': lambda x: ', '.join(x.unique()),
            'grado': lambda x: list(x.unique())
        }).rename(columns={'identificación': 'retirados'}).reset_index()

        dropout_by_period = dropout_by_period.sort_values('retirados', ascending=False)

        # Análisis por período (1 o 2)
        dropout_by_semester = dropout_periods_df.groupby('ultimo_periodo').agg({
            'identificación': 'count'
        }).rename(columns={'identificación': 'retirados'}).reset_index()

        dropout_by_semester['periodo_nombre'] = dropout_by_semester['ultimo_periodo'].map({
            1: 'Primer Semestre',
            2: 'Segundo Semestre'
        })

        # Análisis por año
        dropout_by_year = dropout_periods_df.groupby('ultimo_año').agg({
            'identificación': 'count'
        }).rename(columns={'identificación': 'retirados'}).reset_index()

        # Log de resultados
        self.logger.info(f"Períodos con más retiro:")
        for _, row in dropout_by_period.head(5).iterrows():
            self.logger.info(f"  {row['año_periodo']}: {row['retirados']} estudiantes")

        self.logger.info(f"Retiro por semestre:")
        for _, row in dropout_by_semester.iterrows():
            pct = row['retirados'] / len(dropout_periods_df) * 100
            self.logger.info(f"  {row['periodo_nombre']}: {row['retirados']} ({pct:.1f}%)")

        # Guardar resultados
        self.results['dropout_by_period'] = dropout_by_period
        self.results['dropout_by_semester'] = dropout_by_semester
        self.results['dropout_by_year'] = dropout_by_year
        self.results['dropout_periods_df'] = dropout_periods_df

        self.logger.info("✅ Análisis por período completado")

        return dropout_by_period

    def analyze_students_by_sede_periodo(self):
        """Analiza el número de estudiantes únicos por sede y período."""
        self.logger.info("Analizando estudiantes por sede y período...")

        # Usar groupby con as_index=False para evitar reset_index
        students_by_sede_periodo = (
            self.df.groupby(['sede', 'año_periodo'], as_index=False)['identificación']
            .nunique()
            .rename(columns={'identificación': 'num_estudiantes'})
        )

        # Información general
        total_students = self.df['identificación'].nunique()
        total_sedes = self.df['sede'].nunique()
        sedes_list = sorted(self.df['sede'].unique().tolist())

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

        # Usar groupby con as_index=False y rename
        students_by_grade = (
            self.df.groupby(['grado', 'sede', 'año_periodo'], as_index=False)['identificación']
            .nunique()
            .rename(columns={'identificación': 'num_estudiantes'})
        )

        # Información sobre grados
        grados_list = sorted(self.df['grado'].unique().tolist())
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
        colors = self.get_beautiful_palette(len(sedes_list), palette_name='tab20b')

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
        colors = self.get_beautiful_palette(len(sedes_list), palette_name='tab20b')

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

    def analyze_grade_trajectory_before_dropout(self):
        """Analiza la trayectoria de calificaciones de estudiantes antes de retirarse."""
        self.logger.info("Analizando trayectoria de calificaciones antes de retirarse...")

        dropout_ids = self.results['dropout_ids']
        df_dropout = self.results['df_dropout']

        # Para cada retirado, obtener sus últimas 4 períodos de calificaciones
        trajectory = []

        for student_id in dropout_ids:
            student_data = df_dropout[df_dropout['identificación'] == student_id].copy()
            student_data = student_data.sort_values(['año_num', 'periodo_num'])

            # Obtener promedio por período
            periodos_avg = (
                student_data.groupby(['año_periodo', 'año_num', 'periodo_num'])['resultado']
                .mean()
                .reset_index()
            )

            # Tomar últimos 4 períodos si existen
            last_periodos = periodos_avg.tail(4)

            if len(last_periodos) >= 2:  # Al menos 2 períodos para ver tendencia
                trajectory.append({
                    'identificación': student_id,
                    'num_periodos': len(last_periodos),
                    'promedio_ultimo': last_periodos.iloc[-1]['resultado'],
                    'promedio_penultimo': last_periodos.iloc[-2]['resultado'] if len(last_periodos) >= 2 else None,
                    'promedio_antepenultimo': last_periodos.iloc[-3]['resultado'] if len(last_periodos) >= 3 else None,
                    'tendencia': last_periodos['resultado'].diff().mean()  # Tendencia (positiva o negativa)
                })

        trajectory_df = pd.DataFrame(trajectory)

        # Calcular cuántos tenían tendencia negativa
        negative_trend = (trajectory_df['tendencia'] < 0).sum()
        total_with_trend = len(trajectory_df)

        self.logger.info(f"Estudiantes con tendencia negativa antes de retirarse: {negative_trend}/{total_with_trend} ({negative_trend/total_with_trend*100:.1f}%)")
        self.logger.info(f"Promedio de calificación en último período: {trajectory_df['promedio_ultimo'].mean():.2f}")

        self.results['trajectory_df'] = trajectory_df

        return trajectory_df

    def calculate_risk_indicators(self):
        """Calcula indicadores de riesgo de retiro para todos los estudiantes."""
        self.logger.info("Calculando indicadores de riesgo de retiro...")

        # Para cada estudiante activo, calcular indicadores de riesgo
        active_ids = self.results['active_ids']
        df_active = self.results['df_active']

        risk_indicators = []

        for student_id in active_ids:
            student_data = df_active[df_active['identificación'] == student_id].copy()
            student_data = student_data.sort_values(['año_num', 'periodo_num'])

            # Calcular indicadores
            avg_grade = student_data['resultado'].mean()
            last_grade = student_data.iloc[-1]['resultado']

            # Tendencia (últimos 3 períodos)
            periodos_avg = (
                student_data.groupby(['año_periodo', 'año_num', 'periodo_num'])['resultado']
                .mean()
                .reset_index()
            )

            trend = 0
            if len(periodos_avg) >= 2:
                trend = periodos_avg.tail(3)['resultado'].diff().mean()

            # Calcular riesgo
            risk_score = 0

            # Factor 1: Promedio bajo
            if avg_grade < 60:
                risk_score += 3
            elif avg_grade < 70:
                risk_score += 2
            elif avg_grade < 75:
                risk_score += 1

            # Factor 2: Última calificación baja
            if last_grade < 60:
                risk_score += 2
            elif last_grade < 70:
                risk_score += 1

            # Factor 3: Tendencia negativa
            if trend < -2:
                risk_score += 2
            elif trend < 0:
                risk_score += 1

            # Clasificar riesgo
            if risk_score >= 5:
                risk_level = 'Alto'
            elif risk_score >= 3:
                risk_level = 'Medio'
            else:
                risk_level = 'Bajo'

            risk_indicators.append({
                'identificación': student_id,
                'promedio_general': avg_grade,
                'ultima_calificacion': last_grade,
                'tendencia': trend,
                'puntuacion_riesgo': risk_score,
                'nivel_riesgo': risk_level,
                'sede': student_data.iloc[-1]['sede'],
                'grado': student_data.iloc[-1]['grado']
            })

        risk_df = pd.DataFrame(risk_indicators)

        # Resumen por nivel de riesgo
        risk_summary = risk_df['nivel_riesgo'].value_counts()

        self.logger.info("\nEstudiantes activos por nivel de riesgo de retiro:")
        for level in ['Alto', 'Medio', 'Bajo']:
            if level in risk_summary.index:
                count = risk_summary[level]
                pct = count / len(risk_df) * 100
                self.logger.info(f"  {level}: {count:,} estudiantes ({pct:.1f}%)")

        self.results['risk_df'] = risk_df
        self.results['risk_summary'] = risk_summary

        return risk_df


    def analyze_grade_consistency(self):
        """Analiza la consistencia/variabilidad de calificaciones por estudiante."""
        self.logger.info("Analizando consistencia de calificaciones...")

        dropout_ids = self.results['dropout_ids']
        active_ids = self.results['active_ids']

        # Calcular métricas de consistencia por estudiante
        consistency_data = []

        for student_id in dropout_ids + active_ids:
            student_data = self.df[self.df['identificación'] == student_id].copy()
            
            if len(student_data) >= 3:  # Al menos 3 registros para calcular variabilidad
                grades = student_data['resultado'].dropna()
                
                if len(grades) >= 3:
                    std_dev = grades.std()
                    mean_grade = grades.mean()
                    cv = (std_dev / mean_grade * 100) if mean_grade > 0 else 0  # Coeficiente de variación
                    min_grade = grades.min()
                    max_grade = grades.max()
                    grade_range = max_grade - min_grade
                    
                    consistency_data.append({
                        'identificación': student_id,
                        'grupo': 'Retirado' if student_id in dropout_ids else 'Activo',
                        'num_calificaciones': len(grades),
                        'promedio': mean_grade,
                        'desv_std': std_dev,
                        'coef_variacion': cv,
                        'min': min_grade,
                        'max': max_grade,
                        'rango': grade_range
                    })

        consistency_df = pd.DataFrame(consistency_data)

        # Separar por grupo
        dropout_consistency = consistency_df[consistency_df['grupo'] == 'Retirado']
        active_consistency = consistency_df[consistency_df['grupo'] == 'Activo']

        # Log estadísticas
        self.logger.info(f"\nEstadísticas de CONSISTENCIA:")
        self.logger.info(f"Retirados - Desv. Std promedio: {dropout_consistency['desv_std'].mean():.2f}")
        self.logger.info(f"Activos - Desv. Std promedio: {active_consistency['desv_std'].mean():.2f}")
        self.logger.info(f"Retirados - Coef. Variación promedio: {dropout_consistency['coef_variacion'].mean():.2f}%")
        self.logger.info(f"Activos - Coef. Variación promedio: {active_consistency['coef_variacion'].mean():.2f}%")
        self.logger.info(f"Retirados - Rango promedio: {dropout_consistency['rango'].mean():.2f}")
        self.logger.info(f"Activos - Rango promedio: {active_consistency['rango'].mean():.2f}")

        self.results['consistency_df'] = consistency_df
        self.results['dropout_consistency'] = dropout_consistency
        self.results['active_consistency'] = active_consistency

        return consistency_df

    def plot_grade_consistency(self):
        """Gráfico de análisis de consistencia/variabilidad de calificaciones."""
        self.logger.info("Generando gráfico de consistencia de calificaciones...")

        dropout_consistency = self.results['dropout_consistency']
        active_consistency = self.results['active_consistency']

        if len(dropout_consistency) == 0 or len(active_consistency) == 0:
            self.logger.warning("No hay suficientes datos de consistencia")
            return None

        # Crear figura con 4 subplots
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Subplot 1: Distribución de Desviación Estándar
        ax1 = fig.add_subplot(gs[0, 0])
        
        ax1.hist(dropout_consistency['desv_std'], bins=30, alpha=0.6, color='#ff6b6b',
                label=f'Retirados (n={len(dropout_consistency):,})', edgecolor='black')
        ax1.hist(active_consistency['desv_std'], bins=30, alpha=0.6, color='#51cf66',
                label=f'Activos (n={len(active_consistency):,})', edgecolor='black')
        
        # Líneas de promedio
        ax1.axvline(dropout_consistency['desv_std'].mean(), color='#ff6b6b', 
                   linestyle='--', linewidth=2, label=f'Prom. Retirados: {dropout_consistency["desv_std"].mean():.1f}')
        ax1.axvline(active_consistency['desv_std'].mean(), color='#51cf66', 
                   linestyle='--', linewidth=2, label=f'Prom. Activos: {active_consistency["desv_std"].mean():.1f}')
        
        ax1.set_xlabel('Desviación Estándar de Calificaciones', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
        ax1.set_title('Distribución de Variabilidad (Desviación Estándar)', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Coeficiente de Variación
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Filtrar valores extremos de CV para mejor visualización
        dropout_cv = dropout_consistency[dropout_consistency['coef_variacion'] < 50]['coef_variacion']
        active_cv = active_consistency[active_consistency['coef_variacion'] < 50]['coef_variacion']
        
        ax2.hist(dropout_cv, bins=30, alpha=0.6, color='#ff6b6b',
                label=f'Retirados', edgecolor='black')
        ax2.hist(active_cv, bins=30, alpha=0.6, color='#51cf66',
                label=f'Activos', edgecolor='black')
        
        ax2.axvline(dropout_cv.mean(), color='#ff6b6b', linestyle='--', linewidth=2,
                   label=f'Prom. Retirados: {dropout_cv.mean():.1f}%')
        ax2.axvline(active_cv.mean(), color='#51cf66', linestyle='--', linewidth=2,
                   label=f'Prom. Activos: {active_cv.mean():.1f}%')
        
        ax2.set_xlabel('Coeficiente de Variación (%)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
        ax2.set_title('Distribución de Coeficiente de Variación', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Subplot 3: Boxplots comparativos
        ax3 = fig.add_subplot(gs[1, 0])
        
        data_box = [dropout_consistency['desv_std'], active_consistency['desv_std']]
        bp = ax3.boxplot(data_box, labels=['Retirados', 'Activos'],
                        patch_artist=True, widths=0.6,
                        boxprops=dict(linewidth=1.5),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
        
        # Colorear las cajas
        colors = ['#ff6b6b', '#51cf66']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_ylabel('Desviación Estándar', fontsize=11, fontweight='bold')
        ax3.set_title('Comparación de Variabilidad (Boxplot)', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # Agregar estadísticas
        for i, (data, label) in enumerate(zip(data_box, ['Retirados', 'Activos'])):
            median = data.median()
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            ax3.text(i+1, q3, f'Q3: {q3:.1f}', ha='center', va='bottom', fontsize=9)
            ax3.text(i+1, median, f'Med: {median:.1f}', ha='center', va='center', 
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Subplot 4: Rango de calificaciones (Max - Min)
        ax4 = fig.add_subplot(gs[1, 1])
        
        ax4.hist(dropout_consistency['rango'], bins=30, alpha=0.6, color='#ff6b6b',
                label=f'Retirados', edgecolor='black')
        ax4.hist(active_consistency['rango'], bins=30, alpha=0.6, color='#51cf66',
                label=f'Activos', edgecolor='black')
        
        ax4.axvline(dropout_consistency['rango'].mean(), color='#ff6b6b', linestyle='--', linewidth=2,
                   label=f'Prom. Retirados: {dropout_consistency["rango"].mean():.1f}')
        ax4.axvline(active_consistency['rango'].mean(), color='#51cf66', linestyle='--', linewidth=2,
                   label=f'Prom. Activos: {active_consistency["rango"].mean():.1f}')
        
        ax4.set_xlabel('Rango de Calificaciones (Máx - Mín)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
        ax4.set_title('Distribución del Rango de Variación', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

        # Título general
        fig.suptitle('Análisis de Consistencia de Calificaciones: Estudiantes Retirados vs Activos', 
                    fontsize=16, fontweight='bold', y=0.995)

        # Agregar nota explicativa
        note_text = ('Nota: Mayor desviación estándar y coeficiente de variación indican calificaciones más inconsistentes.\n'
                    'Mayor rango indica diferencias más grandes entre la mejor y peor calificación.')
        fig.text(0.5, 0.02, note_text, ha='center', fontsize=10, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()

        # Guardar
        output_path = f'{self.results_path}/grade_consistency_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✅ Gráfico guardado: {output_path}")

        return output_path

    def plot_components_comparison(self):
        """Gráfico comparativo de componentes de evaluación: retirados vs activos."""
        self.logger.info("Generando gráfico de comparación de componentes de evaluación...")

        df_dropout = self.results['df_dropout']
        df_active = self.results['df_active']

        # Definir los componentes a analizar
        components = ['axiológico', 'cognitivo', 'procedimental', 'actitudinal']
        component_names = {
            'axiológico': 'Axiológico',
            'cognitivo': 'Cognitivo',
            'procedimental': 'Procedimental',
            'actitudinal': 'Actitudinal'
        }

        # Verificar qué componentes existen en el dataset
        available_components = [comp for comp in components if comp in df_dropout.columns]
        
        if len(available_components) == 0:
            self.logger.warning("No hay componentes de evaluación disponibles en el dataset")
            return None

        self.logger.info(f"Componentes disponibles: {available_components}")

        # Filtrar solo componentes con datos suficientes
        valid_components = []
        for comp in available_components:
            dropout_comp = df_dropout[comp].dropna()
            active_comp = df_active[comp].dropna()
            if len(dropout_comp) > 0 and len(active_comp) > 0:
                valid_components.append(comp)
                self.logger.info(f"{component_names[comp]}: Retirados={dropout_comp.mean():.2f}, Activos={active_comp.mean():.2f}")

        if len(valid_components) == 0:
            self.logger.warning("No hay suficientes datos de componentes para generar gráfico")
            return None

        # Crear figura con subplots 2x2 para cada componente
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # Graficar distribución de cada componente
        for idx, comp in enumerate(valid_components):
            ax = axes[idx]
            
            dropout_comp = df_dropout[comp].dropna()
            active_comp = df_active[comp].dropna()

            # Crear histogramas superpuestos
            ax.hist(dropout_comp, bins=20, alpha=0.6, color='#ff6b6b', 
                   label=f'Retirados (n={len(dropout_comp):,})', edgecolor='black', density=False)
            ax.hist(active_comp, bins=20, alpha=0.6, color='#51cf66', 
                   label=f'Activos (n={len(active_comp):,})', edgecolor='black', density=False)

            ax.set_xlabel('Calificación', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
            ax.set_title(f'Distribución: {component_names[comp]}', 
                        fontsize=13, fontweight='bold', pad=10)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # Agregar líneas de promedio
            dropout_mean = dropout_comp.mean()
            active_mean = active_comp.mean()
            
            ax.axvline(dropout_mean, color='#ff6b6b', linestyle='--', linewidth=2, 
                      label=f'Prom. Retirados: {dropout_mean:.1f}')
            ax.axvline(active_mean, color='#51cf66', linestyle='--', linewidth=2, 
                      label=f'Prom. Activos: {active_mean:.1f}')

            # Líneas de referencia
            ax.axvline(60, color='red', linestyle=':', linewidth=1, alpha=0.3)
            ax.axvline(70, color='orange', linestyle=':', linewidth=1, alpha=0.3)
            ax.axvline(80, color='green', linestyle=':', linewidth=1, alpha=0.3)

            # Agregar texto con estadísticas
            stats_text = f'Diferencia: {active_mean - dropout_mean:+.1f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Ocultar subplots vacíos si hay menos de 4 componentes
        for idx in range(len(valid_components), 4):
            axes[idx].axis('off')

        # Título general
        fig.suptitle('Distribución de Componentes de Evaluación: Estudiantes Retirados vs Activos', 
                    fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout()

        # Guardar
        output_path = f'{self.results_path}/components_comparison_dropout_vs_active.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✅ Gráfico guardado: {output_path}")

        return output_path

    def create_text_visualization(self):
        """Crear visualización en texto como alternativa a los gráficos."""
        self.logger.info("Creando visualización en texto...")

        output_path = f'{self.results_path}/visualizacion_texto.txt'

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("VISUALIZACIÓN DE ANÁLISIS DE RETIRO ESCOLAR\n")
            f.write("=" * 80 + "\n\n")

            # 1. Comparación de calificaciones
            df_dropout = self.results['df_dropout']
            df_active = self.results['df_active']

            dropout_mean = df_dropout['resultado'].mean()
            active_mean = df_active['resultado'].mean()

            f.write("1. COMPARACIÓN DE CALIFICACIONES\n")
            f.write("-" * 50 + "\n")
            f.write(f"Retirados:  {dropout_mean:.1f} puntos (n={len(df_dropout['identificación'].unique())})\n")
            f.write(f"Activos:     {active_mean:.1f} puntos (n={len(df_active['identificación'].unique())})\n")
            f.write(f"Diferencia:  {abs(dropout_mean - active_mean):.1f} puntos\n\n")

            # Gráfico de barras ASCII
            max_val = max(dropout_mean, active_mean)
            dropout_bar = "█" * int((dropout_mean / max_val) * 40)
            active_bar = "█" * int((active_mean / max_val) * 40)

            f.write("Gráfico de barras:\n")
            f.write(f"Retirados:  {dropout_bar} {dropout_mean:.1f}\n")
            f.write(f"Activos:    {active_bar} {active_mean:.1f}\n\n")

            # 2. Retiro por sede
            dropout_by_sede = self.results['dropout_by_sede']

            f.write("2. RETIRO POR SEDE\n")
            f.write("-" * 50 + "\n")
            for _, row in dropout_by_sede.iterrows():
                sede = row['sede']
                proporcion = row['proporcion_retiro']
                retirados = row['retirados']
                total = row['total']

                bar = "█" * int((proporcion / 100) * 30)
                f.write(f"{sede:12} {bar:30} {proporcion:5.1f}% ({retirados}/{total})\n")
            f.write("\n")

            # 3. Retiro por grado
            dropout_by_grade = self.results['dropout_by_grade']
            grade_data = dropout_by_grade[dropout_by_grade['proporcion_retiro'] > 0]

            f.write("3. RETIRO POR GRADO\n")
            f.write("-" * 50 + "\n")
            for _, row in grade_data.iterrows():
                grado = row['grado']
                proporcion = row['proporcion_retiro']
                retirados = row['retirados']
                total = row['total']

                bar = "█" * int((proporcion / 100) * 30)
                f.write(f"Grado {grado:2d}    {bar:30} {proporcion:5.1f}% ({retirados}/{total})\n")
            f.write("\n")

            # 4. Distribución de riesgo
            risk_df = self.results['risk_df']
            if len(risk_df) > 0:
                risk_counts = risk_df['nivel_riesgo'].value_counts()

                f.write("4. DISTRIBUCIÓN DE RIESGO (ESTUDIANTES ACTIVOS)\n")
                f.write("-" * 50 + "\n")

                total_risk = len(risk_df)
                for level in ['Alto', 'Medio', 'Bajo']:
                    if level in risk_counts.index:
                        count = risk_counts[level]
                        pct = count / total_risk * 100
                        bar = "█" * int((pct / 100) * 30)
                        f.write(f"Riesgo {level:5} {bar:30} {pct:5.1f}% ({count} estudiantes)\n")
                f.write("\n")

            # 5. Trayectoria
            trajectory_df = self.results['trajectory_df']
            if len(trajectory_df) > 0:
                f.write("5. TRAYECTORIA ANTES DE RETIRARSE\n")
                f.write("-" * 50 + "\n")

                avg_ultimo = trajectory_df['promedio_ultimo'].mean()
                f.write(f"Promedio último período: {avg_ultimo:.1f}\n")

                if 'promedio_penultimo' in trajectory_df.columns:
                    avg_penultimo = trajectory_df['promedio_penultimo'].dropna().mean()
                    if not pd.isna(avg_penultimo):
                        f.write(f"Promedio penúltimo período: {avg_penultimo:.1f}\n")
                        cambio = avg_ultimo - avg_penultimo
                        f.write(f"Cambio: {cambio:+.1f} puntos\n")

                tendencias = trajectory_df['tendencia'].dropna()
                if len(tendencias) > 0:
                    negative_pct = (tendencias < 0).sum() / len(tendencias) * 100
                    f.write(f"Estudiantes con tendencia negativa: {negative_pct:.1f}%\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("NOTA: Esta visualización en texto se creó porque los gráficos PNG\n")
            f.write("no se pudieron generar correctamente en tu sistema.\n")
            f.write("=" * 80 + "\n")

        self.logger.info(f"✅ Visualización en texto creada: {output_path}")
        return output_path

    def plot_dropout_by_demographics(self):
        """Gráfico de retiro por sede y grado."""
        self.logger.info("Generando gráficos de retiro por demografía...")

        dropout_by_sede = self.results['dropout_by_sede']
        dropout_by_grade = self.results['dropout_by_grade']

        # Verificar que hay datos
        if len(dropout_by_sede) == 0 or len(dropout_by_grade) == 0:
            self.logger.warning("No hay datos demográficos para generar gráfico")
            return None

        # Crear figura usando la misma estructura que grades_analysis.py
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Subplot 1: Retiro por sede
        ax1 = axes[0]

        # Obtener colores usando el mismo método
        colors_sede = self.get_beautiful_palette(len(dropout_by_sede), palette_name='tab20b')

        bars1 = ax1.bar(dropout_by_sede['sede'], dropout_by_sede['proporcion_retiro'], 
                       color=colors_sede,
                       edgecolor='black', linewidth=1.5)

        ax1.set_xlabel('Sede', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Proporción de Retiro (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Proporción de Retiro por Sede', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Agregar valores en las barras
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            total = dropout_by_sede.iloc[i]['total']
            retirados = dropout_by_sede.iloc[i]['retirados']
            ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}%\n({int(retirados)}/{int(total)})',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Ajustar límite superior del eje Y para dar espacio a los valores
        current_ylim = ax1.get_ylim()
        ax1.set_ylim(0, current_ylim[1] * 1.15)

        # Subplot 2: Retiro por grado y sede
        ax2 = axes[1]

        # Calcular proporciones por grado y sede
        student_info = self.results['student_last_periodo']
        sedes = student_info['sede'].unique()
        grados = sorted(student_info['grado'].unique())
        
        grade_sede_analysis = []
        
        for grado in grados:
            for sede in sedes:
                # Estudiantes de este grado y sede que se retiraron
                dropout_grado_sede = student_info[
                    (student_info['grado'] == grado) & 
                    (student_info['sede'] == sede) & 
                    (student_info['es_retirado'] == True)
                ]
                
                # Total de estudiantes de este grado y sede
                total_grado_sede = student_info[
                    (student_info['grado'] == grado) & 
                    (student_info['sede'] == sede)
                ]
                
                if len(total_grado_sede) > 0:
                    dropout_rate = len(dropout_grado_sede) / len(total_grado_sede) * 100
                else:
                    dropout_rate = 0
                    
                grade_sede_analysis.append({
                    'grado': grado,
                    'sede': sede,
                    'retirados': len(dropout_grado_sede),
                    'total': len(total_grado_sede),
                    'proporcion_retiro': dropout_rate
                })
        
        grade_sede_df = pd.DataFrame(grade_sede_analysis)
        
        # Filtrar grados con retiro > 0
        grade_sede_filtered = grade_sede_df[grade_sede_df['retirados'] > 0].copy()
        
        if len(grade_sede_filtered) > 0:
            # Obtener grados únicos y sedes únicas
            grados_unicos = sorted(grade_sede_filtered['grado'].unique())
            sedes_unicas = sorted(grade_sede_filtered['sede'].unique())
            
            # Configurar posiciones de las barras
            x_pos = range(len(grados_unicos))
            bar_width = 0.8 / len(sedes_unicas)
            
            # Colores para cada sede
            colors_sede = self.get_beautiful_palette(len(sedes_unicas), palette_name='tab20b')
            
            # Crear barras para cada sede
            for i, sede in enumerate(sedes_unicas):
                sede_data = grade_sede_filtered[grade_sede_filtered['sede'] == sede]
                
                # Preparar datos para cada grado
                proporciones_por_grado = []
                retirados_por_grado = []
                total_por_grado = []
                
                for grado in grados_unicos:
                    grado_data = sede_data[sede_data['grado'] == grado]
                    if len(grado_data) > 0:
                        proporciones_por_grado.append(grado_data['proporcion_retiro'].iloc[0])
                        retirados_por_grado.append(int(grado_data['retirados'].iloc[0]))
                        total_por_grado.append(int(grado_data['total'].iloc[0]))
                    else:
                        proporciones_por_grado.append(0)
                        retirados_por_grado.append(0)
                        total_por_grado.append(0)
                
                # Calcular posiciones de las barras para esta sede
                x_positions = [x + i * bar_width - (len(sedes_unicas) - 1) * bar_width / 2 for x in x_pos]
                
                # Crear barras
                bars = ax2.bar(x_positions, proporciones_por_grado, bar_width, 
                             label=sede, color=colors_sede[i], alpha=0.8, 
                             edgecolor='black', linewidth=0.5)

                # Agregar valores en las barras para esta sede
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    if height > 0:  # Solo mostrar valores si hay datos
                        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                                f'{height:.1f}%\n({retirados_por_grado[j]}/{total_por_grado[j]})',
                                ha='center', va='bottom', fontsize=8, fontweight='bold')
            
        ax2.set_xlabel('Grado', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Proporción de Retiro (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Proporción de Retiro por Grado y Sede', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([str(g) for g in grados_unicos])
        ax2.legend()
        
        if len(proporciones_por_grado) > 0:
            ax2.grid(True, alpha=0.3, axis='y')
            # Ajustar límite superior del eje Y para dar espacio a los valores
            current_ylim = ax2.get_ylim()
            ax2.set_ylim(0, current_ylim[1] * 1.15)
        else:
            ax2.text(0.5, 0.5, 'No hay datos suficientes\npara mostrar proporciones por grado', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Proporción de Retiro por Grado y Sede', fontsize=14, fontweight='bold')

        plt.tight_layout()

        # Guardar usando la misma estructura
        output_path = f'{self.results_path}/dropout_rate_by_demographics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✅ Gráfico guardado: {output_path}")

        return output_path

    def plot_dropout_antiquity(self):
        """Gráfico combinado de antigüedad al momento del retiro."""
        self.logger.info("Generando gráfico de antigüedad al retiro...")

        dropout_students = self.results.get('dropout_antiquity')
        range_df = self.results.get('antiquity_ranges')
        antiquity_stats = self.results.get('antiquity_stats')

        if dropout_students is None or len(dropout_students) == 0:
            self.logger.warning("No hay datos de antigüedad para generar gráfico")
            return None

        # Crear figura con dos subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Subplot 1: Histograma de antigüedad al retiro
        ax1.hist(dropout_students['antiguedad_periodos'], bins=20, 
                color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1)
        
        ax1.set_xlabel('Períodos de Antigüedad al Retiro', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Número de Estudiantes Retirados', fontsize=12, fontweight='bold')
        ax1.set_title('Distribución de Antigüedad al Momento del Retiro', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Agregar líneas de referencia
        ax1.axvline(antiquity_stats['antiguedad_promedio'], color='red', linestyle='--', 
                   linewidth=2, label=f'Promedio: {antiquity_stats["antiguedad_promedio"]:.1f} períodos')
        ax1.axvline(antiquity_stats['antiguedad_mediana'], color='orange', linestyle='--', 
                   linewidth=2, label=f'Mediana: {antiquity_stats["antiguedad_mediana"]:.1f} períodos')
        ax1.legend()

        # Subplot 2: Proporción de retiro por rango de antigüedad y sede
        if range_df is not None and len(range_df) > 0:
            # Obtener datos necesarios
            student_info = self.results['student_last_periodo']
            
            # Definir rangos de antigüedad (mismo que en analyze_dropout_antiquity)
            antiquity_ranges = [
                (1, 2, "1-2 períodos"),
                (3, 4, "3-4 períodos"), 
                (5, 6, "5-6 períodos"),
                (7, 8, "7-8 períodos"),
                (9, 10, "9-10 períodos"),
                (11, 20, "11+ períodos")
            ]
            
            # Calcular proporciones por sede para cada rango
            sedes = dropout_students['sede'].unique()
            range_sede_analysis = []
            
            for min_periods, max_periods, label in antiquity_ranges:
                for sede in sedes:
                    # Estudiantes de esta sede en este rango que se retiraron
                    dropout_in_range_sede = dropout_students[
                        (dropout_students['antiguedad_periodos'] >= min_periods) & 
                        (dropout_students['antiguedad_periodos'] <= max_periods) &
                        (dropout_students['sede'] == sede)
                    ]
                    
                    # Total de estudiantes de esta sede que llegaron al período mínimo
                    total_in_range_sede = len(student_info[
                        (student_info['antiguedad_periodos'] >= min_periods) & 
                        (student_info['sede'] == sede)
                    ])
                    
                    if total_in_range_sede > 0:
                        dropout_rate = len(dropout_in_range_sede) / total_in_range_sede * 100
                    else:
                        dropout_rate = 0
                        
                    range_sede_analysis.append({
                        'rango': label,
                        'sede': sede,
                        'min_periods': min_periods,
                        'max_periods': max_periods,
                        'retirados': len(dropout_in_range_sede),
                        'total_en_rango': total_in_range_sede,
                        'proporcion_retiro': dropout_rate
                    })
            
            range_sede_df = pd.DataFrame(range_sede_analysis)
            
            # Filtrar rangos con datos
            range_sede_filtered = range_sede_df[range_sede_df['retirados'] > 0].copy()
            
            if len(range_sede_filtered) > 0:
                # Obtener rangos únicos y sedes únicas
                # Ordenar rangos por el valor mínimo de períodos para orden lógico
                rangos_unicos = sorted(range_sede_filtered['rango'].unique(), 
                                     key=lambda x: int(x.split('-')[0]) if '-' in x else int(x.split('+')[0]))
                sedes_unicas = sorted(range_sede_filtered['sede'].unique())
                
                # Configurar posiciones de las barras
                x_pos = range(len(rangos_unicos))
                bar_width = 0.8 / len(sedes_unicas)
                
                # Colores para cada sede
                colors_sede = self.get_beautiful_palette(len(sedes_unicas), palette_name='tab20b')
                
                # Crear barras para cada sede
                for i, sede in enumerate(sedes_unicas):
                    sede_data = range_sede_filtered[range_sede_filtered['sede'] == sede]
                    
                    # Preparar datos para cada rango
                    proporciones_por_rango = []
                    retirados_por_rango = []
                    total_por_rango = []
                    
                    for rango in rangos_unicos:
                        rango_data = sede_data[sede_data['rango'] == rango]
                        if len(rango_data) > 0:
                            proporciones_por_rango.append(rango_data['proporcion_retiro'].iloc[0])
                            retirados_por_rango.append(int(rango_data['retirados'].iloc[0]))
                            total_por_rango.append(int(rango_data['total_en_rango'].iloc[0]))
                        else:
                            proporciones_por_rango.append(0)
                            retirados_por_rango.append(0)
                            total_por_rango.append(0)
                    
                    # Calcular posiciones de las barras para esta sede
                    x_positions = [x + i * bar_width - (len(sedes_unicas) - 1) * bar_width / 2 for x in x_pos]

            # Crear barras
                    bars = ax2.bar(x_positions, proporciones_por_rango, bar_width, 
                                 label=sede, color=colors_sede[i], alpha=0.8, 
                                 edgecolor='black', linewidth=0.5)
                    
                    # Agregar valores en las barras
                    for j, bar in enumerate(bars):
                        height = bar.get_height()
                        if height > 0:  # Solo mostrar valores si hay datos
                            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                                    f'{height:.1f}%\n({retirados_por_rango[j]}/{total_por_rango[j]})',
                                    ha='center', va='bottom', fontsize=8, fontweight='bold')
                
                ax2.set_xlabel('Rango de Antigüedad', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Proporción de Retiro (%)', fontsize=12, fontweight='bold')
                ax2.set_title('Proporción de Retiro por Rango de Antigüedad y Sede', fontsize=14, fontweight='bold')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(rangos_unicos, rotation=45, ha='right')
                ax2.legend()
                
                if len(proporciones_por_rango) > 0:
                    # Ajustar límite superior del eje Y para dar espacio a los valores
                    current_ylim = ax2.get_ylim()
                    ax2.set_ylim(0, current_ylim[1] * 1.15)
                
                ax2.grid(True, alpha=0.3, axis='y')
            else:
                ax2.text(0.5, 0.5, 'No hay datos suficientes\npara mostrar proporciones por rango', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Proporción de Retiro por Rango de Antigüedad y Sede', fontsize=14, fontweight='bold')

        # Agregar información estadística en la parte superior
        stats_text = (
            f"Total Retirados: {antiquity_stats['total_retirados']} | "
            f"Antigüedad Promedio: {antiquity_stats['antiguedad_promedio']:.1f} períodos | "
            f"Rango: {antiquity_stats['antiguedad_min']}-{antiquity_stats['antiguedad_max']} períodos"
        )
        
        fig.text(0.5, 0.95, stats_text, ha='center', va='top', fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

        plt.tight_layout(rect=[0, 0.03, 1, 0.90])

        # Guardar gráfico
        output_path = f'{self.results_path}/dropout_antiquity_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✅ Gráfico de antigüedad guardado: {output_path}")
        return output_path

    def plot_grade_trajectory(self):
        """Gráfico de trayectoria de calificaciones antes de retirarse."""
        self.logger.info("Generando gráfico de trayectoria de calificaciones...")

        trajectory_df = self.results['trajectory_df']

        # Crear figura SIEMPRE, incluso sin datos
        fig, ax = plt.subplots(figsize=(12, 8))

        # Configurar título y ejes básicos
        ax.set_title('Evolución del Promedio Antes de Retirarse', fontsize=14, fontweight='bold')
        ax.set_ylabel('Calificación Promedio', fontsize=12, fontweight='bold')
        ax.set_xlabel('Períodos', fontsize=12, fontweight='bold')

        # Verificar que hay datos
        if len(trajectory_df) == 0:
            # Mostrar mensaje informativo
            ax.text(0.5, 0.5, 'No hay datos de trayectoria disponibles\n\n' +
                    'Esto puede ocurrir si:\n' +
                    '• No hay estudiantes retirados identificados\n' +
                    '• Los estudiantes retirados no tienen suficiente historial',
                    ha='center', va='center', transform=ax.transAxes, 
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 100)
        else:
            # Calcular promedios con valores por defecto
            avg_ultimo = trajectory_df['promedio_ultimo'].dropna().mean()

            # Crear datos básicos - al menos mostrar el último período
            if not pd.isna(avg_ultimo) and avg_ultimo > 0:
                periodos = ['Último período\nantes de retirarse']
                promedios = [avg_ultimo]

                # Intentar agregar más períodos si están disponibles
                if 'promedio_penultimo' in trajectory_df.columns:
                    avg_penultimo = trajectory_df['promedio_penultimo'].dropna().mean()
                    if not pd.isna(avg_penultimo) and avg_penultimo > 0:
                        periodos.insert(0, 'Penúltimo período')
                        promedios.insert(0, avg_penultimo)

                # Crear gráfico de barras simple
                colors = ['#ff6b6b', '#ff8e8e'] if len(promedios) > 1 else ['#ff6b6b']
                bars = ax.bar(periodos, promedios, color=colors[:len(promedios)], 
                             alpha=0.8, edgecolor='black', linewidth=1)

                # Agregar valores en las barras
                for bar, promedio in zip(bars, promedios):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{promedio:.1f}', ha='center', va='bottom', 
                           fontsize=12, fontweight='bold')

                # Líneas de referencia
                ax.axhline(y=70, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Básico (70)')
                ax.axhline(y=60, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Bajo (60)')
                ax.legend()

                ax.set_ylim(0, 100)

                # Información adicional
                n_students = len(trajectory_df)
                ax.text(0.02, 0.98, f'Basado en {n_students} estudiantes retirados', 
                       transform=ax.transAxes, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                       verticalalignment='top')
            else:
                # No hay datos válidos de calificaciones
                ax.text(0.5, 0.5, f'Datos insuficientes para mostrar trayectoria\n\n' +
                        f'Se encontraron {len(trajectory_df)} estudiantes retirados\n' +
                        f'pero sin calificaciones válidas en el último período',
                        ha='center', va='center', transform=ax.transAxes, 
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.7))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 100)

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Guardar usando la misma estructura
        output_path = f'{self.results_path}/grade_trajectory_before_dropout.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        self.logger.info(f"✅ Gráfico guardado: {output_path}")

        return output_path

    def plot_individual_trajectories(self):
        """Gráfico de trayectorias individuales de estudiantes retirados."""
        self.logger.info("Generando gráfico de trayectorias individuales...")

        dropout_ids = self.results['dropout_ids']
        df_dropout = self.results['df_dropout']

        if len(dropout_ids) == 0:
            self.logger.warning("No hay estudiantes retirados para mostrar trayectorias")
            return None

        # Obtener trayectorias individuales de los últimos 4 períodos
        trajectories = []
        
        for student_id in dropout_ids:
            student_data = df_dropout[df_dropout['identificación'] == student_id].copy()
            student_data = student_data.sort_values(['año_num', 'periodo_num'])
            
            # Obtener promedio por período
            periodos_avg = (
                student_data.groupby(['año_periodo', 'año_num', 'periodo_num'])['resultado']
                .mean()
                .reset_index()
            )
            
            # Tomar últimos 4 períodos si existen
            last_periodos = periodos_avg.tail(4)
            
            if len(last_periodos) >= 2:  # Al menos 2 períodos para ver tendencia
                # Calcular tendencia
                tendencia = last_periodos['resultado'].diff().mean()
                
                # Crear datos para la línea
                x_positions = range(len(last_periodos))
                y_values = last_periodos['resultado'].tolist()
                
                trajectories.append({
                    'student_id': student_id,
                    'x_positions': x_positions,
                    'y_values': y_values,
                    'tendencia': tendencia,
                    'periodos': last_periodos['año_periodo'].tolist(),
                    'inicio': y_values[0] if len(y_values) > 0 else 0,
                    'final': y_values[-1] if len(y_values) > 0 else 0,
                    'cambio': y_values[-1] - y_values[0] if len(y_values) > 1 else 0
                })

        if len(trajectories) == 0:
            self.logger.warning("No hay trayectorias suficientes para mostrar")
            return None

        # Separar por tendencia
        negative_trajectories = [t for t in trajectories if t['tendencia'] < 0]
        positive_trajectories = [t for t in trajectories if t['tendencia'] >= 0]

        # Crear figura con 2 subplots - solo enfocarse en deterioro
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Subplot 1: Solo trayectorias con deterioro (los 122 estudiantes)
        ax1.set_title(f'122 Estudiantes con Deterioro en Calificaciones', 
                     fontsize=16, fontweight='bold')
        ax1.set_xlabel('Períodos (últimos 4 antes del retiro)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Calificación Promedio', fontsize=12, fontweight='bold')
        
        # Graficar líneas individuales con colores según severidad del deterioro
        for i, traj in enumerate(negative_trajectories):
            # Color más intenso para deterioro más severo
            alpha = min(0.7, 0.4 + abs(traj['cambio']) / 25)
            color_intensity = min(1.0, abs(traj['cambio']) / 30)
            color = (0.9, color_intensity * 0.3, color_intensity * 0.3)  # Rojo más intenso para mayor deterioro
            
            ax1.plot(traj['x_positions'], traj['y_values'], 
                    alpha=alpha, linewidth=1.2, color=color)
        
        # Línea de tendencia promedio
        if negative_trajectories:
            all_x = []
            all_y = []
            for traj in negative_trajectories:
                all_x.extend(traj['x_positions'])
                all_y.extend(traj['y_values'])
            
            # Calcular promedio por posición
            unique_x = sorted(set(all_x))
            avg_y = []
            for x in unique_x:
                y_at_x = [y for i, y in enumerate(all_y) if all_x[i] == x]
                avg_y.append(np.mean(y_at_x))
            
            ax1.plot(unique_x, avg_y, color='darkred', linewidth=5, 
                    label=f'Tendencia promedio: {np.mean([t["tendencia"] for t in negative_trajectories]):.2f} pts/período')
            ax1.legend(fontsize=12)

        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Líneas de referencia
        ax1.axhline(y=70, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='Básico (70)')
        ax1.axhline(y=60, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Bajo (60)')
        ax1.legend(fontsize=10)

        # Subplot 2: Distribución de cambios en calificaciones (solo deterioro)
        ax2.set_title('Distribución de Deterioro en Calificaciones', 
                     fontsize=16, fontweight='bold')
        ax2.set_xlabel('Cambio en Calificaciones (puntos)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Número de Estudiantes', fontsize=12, fontweight='bold')
        
        # Crear histograma solo de cambios negativos
        cambios_negativos = [t['cambio'] for t in negative_trajectories]
        ax2.hist(cambios_negativos, bins=15, alpha=0.7, color='red', edgecolor='darkred')
        
        # Línea vertical en 0
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.8, label='Sin cambio')
        
        # Estadísticas específicas del deterioro
        ax2.text(0.02, 0.98, f'Total con deterioro: {len(cambios_negativos)} estudiantes\n'
                             f'Deterioro promedio: {np.mean(cambios_negativos):.1f} puntos\n'
                             f'Deterioro máximo: {min(cambios_negativos):.1f} puntos\n'
                             f'Deterioro mínimo: {max(cambios_negativos):.1f} puntos',
                transform=ax2.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.8))
        
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Título general (mover más arriba para evitar superposición)
        fig.suptitle('Análisis de Trayectorias de Calificaciones - Estudiantes Retirados', 
                    fontsize=16, fontweight='bold', y=0.98)

        # Estadísticas discretas en la esquina superior derecha
        total_students = len(trajectories)
        negative_pct = len(negative_trajectories) / total_students * 100
        avg_negative_trend = np.mean([t['tendencia'] for t in negative_trajectories]) if negative_trajectories else 0
        avg_negative_change = np.mean([t['cambio'] for t in negative_trajectories]) if negative_trajectories else 0
        
        stats_text = (f'Total: {total_students} | Deterioro: {len(negative_trajectories)} ({negative_pct:.1f}%)\n'
                     f'Tendencia: {avg_negative_trend:.2f} pts/período | Cambio: {avg_negative_change:.1f} pts')
        
        fig.text(0.98, 0.98, stats_text, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='gray'))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Dejar espacio arriba para el título

        # Guardar
        output_path = f'{self.results_path}/individual_grade_trajectories.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        self.logger.info(f"✅ Gráfico de trayectorias individuales guardado: {output_path}")

        return output_path

    def plot_dropout_by_period(self):
        self.logger.info("Generando gráfico de proporción de retiro por período y sede...")

        # LÓGICA CORREGIDA: Mostrar retiro en el período donde se detecta
        # Si estaban en período N-1 pero NO en N → se detecta retiro en período N
        # NOTA: Excluimos el primer período porque no tenemos período anterior para comparar

        # Obtener todos los períodos ordenados
        periodos_ordenados = sorted(self.df[['año', 'periodo']].drop_duplicates().values.tolist())

        if len(periodos_ordenados) < 2:
            self.logger.warning("Se necesitan al menos 2 períodos para calcular retiro (excluimos el primero)")
            return None

        # Log de información sobre períodos
        primer_periodo = f"{periodos_ordenados[0][0]}-{periodos_ordenados[0][1]}"
        ultimo_periodo = f"{periodos_ordenados[-1][0]}-{periodos_ordenados[-1][1]}"
        self.logger.info(f"Períodos disponibles: {primer_periodo} a {ultimo_periodo}")
        self.logger.info(f"Analizando retiro desde {periodos_ordenados[1][0]}-{periodos_ordenados[1][1]} hasta {periodos_ordenados[-1][0]}-{periodos_ordenados[-1][1]}")

        dropout_rates = []

        # Para cada período (excepto el primero), calcular quién se va
        # Lógica: estudiantes que estaban en N-1 pero NO están en N
        for i in range(1, len(periodos_ordenados)):
            periodo_actual = periodos_ordenados[i]
            periodo_anterior = periodos_ordenados[i-1]

            año_act, per_act = periodo_actual
            año_ant, per_ant = periodo_anterior

            # Estudiantes en período anterior por sede
            estudiantes_anterior = self.df[
                (self.df['año'] == año_ant) & (self.df['periodo'] == per_ant)
            ].groupby('sede')['identificación'].apply(set).to_dict()

            # Estudiantes en período actual por sede
            estudiantes_actual = self.df[
                (self.df['año'] == año_act) & (self.df['periodo'] == per_act)
            ].groupby('sede')['identificación'].apply(set).to_dict()

            # Calcular retiro por sede (quién se va entre período anterior y actual)
            for sede in estudiantes_anterior.keys():
                estudiantes_ant = estudiantes_anterior[sede]
                estudiantes_curr = estudiantes_actual.get(sede, set())

                # Retirados = estaban en anterior pero NO en actual
                retirados = estudiantes_ant - estudiantes_curr

                if len(estudiantes_ant) > 0:  # Evitar división por cero
                    proporcion_retiro = len(retirados) / len(estudiantes_ant) * 100

                    dropout_rates.append({
                        'año_periodo': f"{año_act}-{per_act}",  # Período donde se detecta el retiro
                        'sede': sede,
                        'estudiantes_periodo': len(estudiantes_ant),  # Base: estudiantes del período anterior
                        'retirados': len(retirados),
                        'proporcion_retiro': proporcion_retiro,
                        'año': año_act,
                        'periodo': per_act
                    })

        if len(dropout_rates) == 0:
            self.logger.warning("No se pudieron calcular proporciones de retiro")
            return None

        dropout_rate_data = pd.DataFrame(dropout_rates)

        # Filtrar solo períodos con retiro > 0 para el gráfico
        dropout_rate_data_plot = dropout_rate_data[dropout_rate_data['retirados'] > 0]

        if len(dropout_rate_data_plot) == 0:
            self.logger.warning("No hay períodos con retiro > 0")
            return None

        # Crear figura
        fig, ax = plt.subplots(figsize=(16, 10))

        # Obtener períodos únicos y sedes únicas (de los datos con retiro > 0)
        periodos_unicos = sorted(dropout_rate_data_plot['año_periodo'].unique())
        sedes_unicas = sorted(dropout_rate_data_plot['sede'].unique())

        # Log de información para debug
        self.logger.info(f"Períodos con retiro: {periodos_unicos}")
        self.logger.info(f"Sedes con retiro: {sedes_unicas}")

        # Mostrar algunos ejemplos de cálculo
        for _, row in dropout_rate_data_plot.head(3).iterrows():
            self.logger.info(f"Ejemplo: {row['año_periodo']} - {row['sede']}: "
                           f"{row['retirados']}/{row['estudiantes_periodo']} = {row['proporcion_retiro']:.1f}% "
                           f"(se fueron al final de {row['año_periodo']})")

        # Colores para cada sede
        colors_sede = self.get_beautiful_palette(len(sedes_unicas), palette_name='tab20b')

        # Configurar posiciones de las barras
        x_pos = range(len(periodos_unicos))
        bar_width = 0.8 / len(sedes_unicas)

        # Crear barras para cada sede
        for i, sede in enumerate(sedes_unicas):
            sede_data = dropout_rate_data_plot[dropout_rate_data_plot['sede'] == sede]

            # Preparar datos para cada período
            proporciones_por_periodo = []
            retirados_por_periodo = []
            estudiantes_por_periodo = []

            for periodo in periodos_unicos:
                periodo_data = sede_data[sede_data['año_periodo'] == periodo]
                if len(periodo_data) > 0:
                    proporciones_por_periodo.append(periodo_data['proporcion_retiro'].iloc[0])
                    retirados_por_periodo.append(int(periodo_data['retirados'].iloc[0]))
                    estudiantes_por_periodo.append(int(periodo_data['estudiantes_periodo'].iloc[0]))
                else:
                    proporciones_por_periodo.append(0)
                    retirados_por_periodo.append(0)
                    estudiantes_por_periodo.append(0)

            # Calcular posiciones de las barras para esta sede
            x_positions = [x + i * bar_width - (len(sedes_unicas) - 1) * bar_width / 2 for x in x_pos]

            # Crear barras
            bars = ax.bar(x_positions, proporciones_por_periodo, bar_width, 
                         label=sede, color=colors_sede[i], alpha=0.8, 
                         edgecolor='black', linewidth=0.5)

            # Agregar valores en las barras
            for j, (bar, proporcion, retirados, estudiantes) in enumerate(zip(bars, proporciones_por_periodo, retirados_por_periodo, estudiantes_por_periodo)):
                if proporcion > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                           f'{proporcion:.1f}%\n({retirados}/{estudiantes})',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Configurar el gráfico
        ax.set_xlabel('Período Académico', fontsize=12, fontweight='bold')
        ax.set_ylabel('Proporción de Retiro (%)', fontsize=12, fontweight='bold')
        ax.set_title('Proporción de Retiro por Período Académico y Sede', fontsize=14, fontweight='bold')

        # Configurar etiquetas del eje X
        ax.set_xticks(x_pos)
        ax.set_xticklabels(periodos_unicos, rotation=45, ha='right')

        # Agregar leyenda
        ax.legend(title='Sede', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Agregar grid
        ax.grid(True, alpha=0.3, axis='y')

        # Ajustar límites del eje Y
        max_proporcion = dropout_rate_data_plot['proporcion_retiro'].max()
        ax.set_ylim(0, max_proporcion * 1.15)

        plt.tight_layout()

        # Guardar usando la misma estructura
        output_path = f'{self.results_path}/dropout_by_period.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        self.logger.info(f"✅ Gráfico guardado: {output_path}")

        return output_path

    def generate_summary_report(self):
        """Genera reporte resumen del análisis de retiro."""
        self.logger.info("Generando reporte resumen...")

        output_path = f'{self.results_path}/dropout_analysis_summary.txt'

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ANÁLISIS DE RETIRO ESCOLAR\n")
            f.write("=" * 80 + "\n\n")

            # Estadísticas generales
            f.write("1. ESTADÍSTICAS GENERALES\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total de estudiantes únicos: {self.results['total_students']:,}\n")
            f.write(f"Estudiantes retirados: {self.results['total_dropout']:,}\n")
            f.write(f"Estudiantes activos: {self.results['total_active']:,}\n")
            f.write(f"Proporción de retiro: {self.results['dropout_rate']:.1f}%\n\n")

            # Comparación de calificaciones
            f.write("2. COMPARACIÓN DE CALIFICACIONES\n")
            f.write("-" * 80 + "\n")
            dropout_stats = self.results['dropout_grades_stats']
            active_stats = self.results['active_grades_stats']

            f.write("Estudiantes RETIRADOS:\n")
            f.write(f"  Promedio: {dropout_stats['promedio']:.2f}\n")
            f.write(f"  Mediana: {dropout_stats['mediana']:.2f}\n")
            f.write(f"  Desv. Estándar: {dropout_stats['desv_std']:.2f}\n")
            f.write(f"  Rango: {dropout_stats['min']:.2f} - {dropout_stats['max']:.2f}\n\n")

            f.write("Estudiantes ACTIVOS:\n")
            f.write(f"  Promedio: {active_stats['promedio']:.2f}\n")
            f.write(f"  Mediana: {active_stats['mediana']:.2f}\n")
            f.write(f"  Desv. Estándar: {active_stats['desv_std']:.2f}\n")
            f.write(f"  Rango: {active_stats['min']:.2f} - {active_stats['max']:.2f}\n\n")

            diff = abs(dropout_stats['promedio'] - active_stats['promedio'])
            f.write(f"DIFERENCIA de promedios: {diff:.2f} puntos\n")
            f.write(f"Los estudiantes retirados tienen un promedio {'MENOR' if dropout_stats['promedio'] < active_stats['promedio'] else 'MAYOR'}\n\n")

            # Retiro por sede
            f.write("3. RETIRO POR SEDE\n")
            f.write("-" * 80 + "\n")
            dropout_by_sede = self.results['dropout_by_sede']
            f.write(dropout_by_sede.to_string(index=False))
            f.write("\n\n")

            # Retiro por grado
            f.write("4. RETIRO POR GRADO\n")
            f.write("-" * 80 + "\n")
            dropout_by_grade = self.results['dropout_by_grade']
            f.write(dropout_by_grade.to_string(index=False))
            f.write("\n\n")

            # Trayectoria antes de retirarse
            f.write("5. TRAYECTORIA ANTES DE RETIRARSE\n")
            f.write("-" * 80 + "\n")
            trajectory_df = self.results['trajectory_df']
            negative_trend = (trajectory_df['tendencia'] < 0).sum()
            total = len(trajectory_df)
            f.write(f"Estudiantes con tendencia negativa: {negative_trend}/{total} ({negative_trend/total*100:.1f}%)\n")
            f.write(f"Promedio último período: {trajectory_df['promedio_ultimo'].mean():.2f}\n")
            f.write(f"Tendencia promedio: {trajectory_df['tendencia'].mean():.2f} puntos/período\n\n")

            # Riesgo de retiro
            f.write("6. INDICADORES DE RIESGO DE RETIRO (ESTUDIANTES ACTIVOS)\n")
            f.write("-" * 80 + "\n")
            risk_summary = self.results['risk_summary']
            total_active = self.results['total_active']

            for level in ['Alto', 'Medio', 'Bajo']:
                if level in risk_summary.index:
                    count = risk_summary[level]
                    pct = count / total_active * 100
                    f.write(f"Riesgo {level}: {count:,} estudiantes ({pct:.1f}%)\n")

            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("CONCLUSIONES Y RECOMENDACIONES\n")
            f.write("=" * 80 + "\n\n")

            f.write("PATRONES IDENTIFICADOS:\n")
            f.write("1. Los estudiantes retirados tienen calificaciones significativamente más bajas\n")
            f.write("2. Existe una tendencia negativa en las calificaciones antes de retirarse\n")
            f.write("3. El retiro varía según sede y grado\n\n")

            f.write("RECOMENDACIONES:\n")
            f.write("1. Implementar sistema de alerta temprana para estudiantes en riesgo\n")
            f.write("2. Intervenciones focalizadas en estudiantes con tendencia negativa\n")
            f.write("3. Apoyo adicional en sedes y grados con mayor proporción de retiro\n")
            f.write("4. Seguimiento especial a estudiantes con riesgo alto\n\n")

        # Guardar CSVs adicionales
        self.results['risk_df'].to_csv(f'{self.results_path}/estudiantes_en_riesgo.csv', index=False)
        self.results['trajectory_df'].to_csv(f'{self.results_path}/trayectoria_retirados.csv', index=False)

        self.logger.info(f"✅ Reporte guardado: {output_path}")

        return output_path

    def run_analysis(self):
        """Ejecuta el pipeline completo de análisis de retiro."""
        self.logger.info("=" * 60)
        self.logger.info("INICIANDO ANÁLISIS DE RETIRO ESCOLAR")
        self.logger.info("=" * 60)

        try:
            # Crear directorio de resultados
            self.create_results_directory()

            # 1. Cargar datos
            self.load_grades_data()

            # 2. Analizar evolución de estudiantes por sede y período
            self.analyze_students_by_sede_periodo()

            # 3. Analizar evolución de estudiantes por grado, sede y período
            self.analyze_students_by_grade_sede_periodo()

            # 4. Identificar retirados
            self.identify_dropout_students()

            # 5. Analizar calificaciones
            self.analyze_dropout_grades()

            # 6. Analizar por demografía
            self.analyze_dropout_by_demographics()

            # 7. Analizar antigüedad al retiro
            self.analyze_dropout_antiquity()

            # 8. Analizar por período académico
            self.analyze_dropout_by_period()

            # 9. Analizar trayectoria
            self.analyze_grade_trajectory_before_dropout()

            # 10. Calcular riesgo
            self.calculate_risk_indicators()

            # 11. Analizar consistencia de calificaciones
            self.analyze_grade_consistency()

            # 12. Generar gráficos uno por uno
            self.logger.info("Generando gráficos...")

            # Gráfico de evolución de estudiantes por sede
            try:
                self.plot_students_evolution()
                self.logger.info("✅ Gráfico de evolución por sede generado")
            except Exception as e:
                self.logger.error(f"❌ Error gráfico de evolución por sede: {e}")

            # Gráfico de evolución de estudiantes por grado y sede
            try:
                self.plot_students_evolution_by_grade()
                self.logger.info("✅ Gráfico de evolución por grado y sede generado")
            except Exception as e:
                self.logger.error(f"❌ Error gráfico de evolución por grado y sede: {e}")

            # Gráfico de comparación de componentes de evaluación
            try:
                self.plot_components_comparison()
                self.logger.info("✅ Gráfico de componentes de evaluación generado")
            except Exception as e:
                self.logger.error(f"❌ Error gráfico de componentes: {e}")

            # Gráfico de consistencia de calificaciones
            try:
                self.plot_grade_consistency()
                self.logger.info("✅ Gráfico de consistencia generado")
            except Exception as e:
                self.logger.error(f"❌ Error gráfico de consistencia: {e}")

            # Gráfico de demografía
            try:
                self.plot_dropout_by_demographics()
                self.logger.info("✅ Gráfico de demografía generado")
            except Exception as e:
                self.logger.error(f"❌ Error gráfico de demografía: {e}")

            # Gráfico de antigüedad al retiro
            try:
                self.plot_dropout_antiquity()
                self.logger.info("✅ Gráfico de antigüedad al retiro generado")
            except Exception as e:
                self.logger.error(f"❌ Error gráfico de antigüedad: {e}")

            # Gráfico de trayectoria
            try:
                self.plot_grade_trajectory()
                self.logger.info("✅ Gráfico de trayectoria generado")
            except Exception as e:
                self.logger.error(f"❌ Error gráfico de trayectoria: {e}")

            # Gráfico de trayectorias individuales
            try:
                self.plot_individual_trajectories()
                self.logger.info("✅ Gráfico de trayectorias individuales generado")
            except Exception as e:
                self.logger.error(f"❌ Error gráfico de trayectorias individuales: {e}")

            # Gráfico de retiro por período
            try:
                self.plot_dropout_by_period()
                self.logger.info("✅ Gráfico de retiro por período generado")
            except Exception as e:
                self.logger.error(f"❌ Error gráfico de retiro por período: {e}")

            # 11. Generar reporte resumen
            self.generate_summary_report()

            self.logger.info("=" * 60)
            self.logger.info("✅ ANÁLISIS DE RETIRO COMPLETADO")
            self.logger.info("=" * 60)

            return self.results

        except Exception as e:
            self.logger.error(f"❌ Error en análisis: {e}")
            raise


def main():
    """Función principal."""
    import argparse

    parser = argparse.ArgumentParser(description='Análisis de retiro escolar')
    parser.add_argument('--dataset', '-d', type=str, 
                       default='data/interim/calificaciones/calificaciones_2021-2025.csv',
                       help='Ruta al archivo CSV de calificaciones')
    parser.add_argument('--results', '-r', type=str, 
                       default='dropout_analysis',
                       help='Nombre del folder para guardar resultados')

    args = parser.parse_args()

    # Crear y ejecutar analizador
    analyzer = DropoutAnalysis(args.dataset, args.results)

    try:
        analyzer.run_analysis()
        analyzer.logger.info("✅ Análisis completado exitosamente")
    except Exception as e:
        analyzer.logger.error(f"❌ Error: {e}")
        raise
    finally:
        analyzer.close()


if __name__ == "__main__":
    main()

