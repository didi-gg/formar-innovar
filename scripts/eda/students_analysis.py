"""
Script para análisis exploratorio de datos de estudiantes
Analiza características demográficas, socioeconómicas y académicas de los estudiantes
"""

import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
import argparse

warnings.filterwarnings('ignore')

# Agregar el directorio padre al path para importar utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.eda_analysis_base import EDAAnalysisBase

class StudentsAnalysis(EDAAnalysisBase):
    """
    Clase para análisis exploratorio de datos de estudiantes.
    Genera visualizaciones y estadísticas descriptivas de características demográficas,
    socioeconómicas y académicas de los estudiantes.
    """

    def _initialize_analysis_attributes(self):
        self.df_students = None
        self.categorical_columns = []
        self.numerical_columns = []
        self.ordinal_columns = []

    def _categorize_columns(self, df: pd.DataFrame):
        self.categorical_columns = []
        self.numerical_columns = []
        self.ordinal_columns = []

        for col in df.columns:
            if col in ['ID', 'documento_identificación', 'primer_apellido', 'segundo_apellido', 
                      'nombres', 'dirección']:
                continue  # Saltar columnas de identificación

            if df[col].dtype == 'object':
                # Verificar si es ordinal basado en valores únicos
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 20:  # Posible variable ordinal
                    self.categorical_columns.append(col)
                else:
                    self.categorical_columns.append(col)
            elif df[col].dtype in ['int64', 'float64']:
                if col in ['estrato', 'total_hermanos', 'horas_semana_estudio_casa']:
                    self.numerical_columns.append(col)
                else:
                    self.numerical_columns.append(col)

    def _get_ordinal_mapping(self, column: str) -> dict:
        ordinal_mappings = {
            'demuestra_confianza': {
                'Nunca lo demuestra': 1,
                'Rara vez lo demuestra': 2,
                'A veces lo demuestra': 3,
                'Frecuentemente lo demuestra': 4,
                'Siempre lo demuestra': 5
            },
            'interés_estudios_superiores': {
                'Bajo': 1,
                'Medio': 2,
                'Alto': 3
            },
            'apoyo_familiar': {
                'Bajo': 1,
                'Medio': 2,
                'Alto': 3
            },
            'participación_clase': {
                'Baja': 1,
                'Media': 2,
                'Alta': 3
            },
            'nivel_motivación': {
                'Bajo': 1,
                'Medio': 2,
                'Alto': 3
            }
        }
        return ordinal_mappings.get(column, {})

    def _encode_ordinal_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = df.copy()

        for col in df_encoded.columns:
            if col in ['demuestra_confianza', 'interés_estudios_superiores', 
                      'apoyo_familiar', 'participación_clase', 'nivel_motivación']:
                mapping = self._get_ordinal_mapping(col)
                if mapping:
                    df_encoded[f'{col}_encoded'] = df_encoded[col].map(mapping)
                    self.ordinal_columns.append(f'{col}_encoded')

        return df_encoded

    def load_and_prepare_data(self) -> pd.DataFrame:
        # Cargar el dataset principal
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"No se encontró el dataset en: {self.dataset_path}")

        df = pd.read_csv(self.dataset_path)
        self.logger.info(f"Dataset de estudiantes cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

        # Convertir fecha_nacimiento a datetime
        if 'fecha_nacimiento' in df.columns:
            df['fecha_nacimiento'] = pd.to_datetime(df['fecha_nacimiento'], errors='coerce')
            # Calcular edad
            current_year = pd.Timestamp.now().year
            df['edad'] = current_year - df['fecha_nacimiento'].dt.year
            self.numerical_columns.append('edad')

        # Categorizar columnas
        self._categorize_columns(df)

        # Codificar variables ordinales
        df_encoded = self._encode_ordinal_variables(df)

        # Guardar referencia
        self.df_students = df_encoded

        return df_encoded

    def _calculate_text_contrast_color(self, color_rgba: tuple) -> str:
        r, g, b = color_rgba[:3]
        luminosidad = 0.299 * r + 0.587 * g + 0.114 * b
        return 'white' if luminosidad < 0.5 else 'black'

    def _create_pie_chart_with_contrast(self, ax, values, labels, colors, title):
        """Crear gráfico de torta con colores de texto contrastantes"""
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', 
                                         startangle=90, colors=colors)

        # Ajustar colores de texto según luminosidad del fondo
        for autotext, color in zip(autotexts, colors):
            text_color = self._calculate_text_contrast_color(color)
            autotext.set_color(text_color)
            autotext.set_fontweight('bold')

        ax.set_title(title, fontsize=12, weight='bold')
        return wedges, texts, autotexts

    def create_demographic_analysis(self, output_dir: str):
        self.logger.info("Creando análisis demográfico...")

        # Crear figura con subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        # 1. Distribución por sede
        if 'sede' in self.df_students.columns:
            sede_counts = self.df_students['sede'].value_counts()
            colors = self.get_beautiful_palette(len(sede_counts), palette_name='tab20b')
            axes[0].bar(sede_counts.index, sede_counts.values, color=colors)
            axes[0].set_title('Distribución por Sede', fontsize=12, weight='bold')
            axes[0].set_ylabel('Número de Estudiantes')
            for i, v in enumerate(sede_counts.values):
                axes[0].text(i, v + 0.5, str(v), ha='center', va='bottom')

        # 2. Género por sede
        if 'género' in self.df_students.columns and 'sede' in self.df_students.columns:
            cross_table = pd.crosstab(self.df_students['sede'], self.df_students['género'])
            colors = self.get_beautiful_palette(len(cross_table.columns), palette_name='tab20b')
            cross_table.plot(kind='bar', ax=axes[1], color=colors)
            axes[1].set_title('Distribución de Género por Sede', fontsize=12, weight='bold')
            axes[1].set_ylabel('Número de Estudiantes')
            axes[1].legend(title='Género')
            axes[1].tick_params(axis='x', rotation=45)

        # 3. Año de ingreso por sede
        if 'año_ingreso' in self.df_students.columns and 'sede' in self.df_students.columns:
            cross_table = pd.crosstab(self.df_students['sede'], self.df_students['año_ingreso'])
            colors = self.get_beautiful_palette(len(cross_table.columns), palette_name='tab20b')
            cross_table.plot(kind='bar', ax=axes[2], color=colors)
            axes[2].set_title('Distribución de Año de Ingreso por Sede', fontsize=12, weight='bold')
            axes[2].set_ylabel('Número de Estudiantes')
            axes[2].legend(title='Año')
            axes[2].tick_params(axis='x', rotation=45)

        # 4. Antigüedad por sede
        if 'antigüedad' in self.df_students.columns and 'sede' in self.df_students.columns:
            cross_table = pd.crosstab(self.df_students['sede'], self.df_students['antigüedad'])
            colors = self.get_beautiful_palette(len(cross_table.columns), palette_name='tab20b')
            cross_table.plot(kind='bar', ax=axes[3], color=colors)
            axes[3].set_title('Distribución de Antigüedad por Sede', fontsize=12, weight='bold')
            axes[3].set_ylabel('Número de Estudiantes')
            axes[3].legend(title='Antigüedad')
            axes[3].tick_params(axis='x', rotation=45)

        # 5. Distribución de edad por sede
        if 'edad' in self.df_students.columns and 'sede' in self.df_students.columns:
            for sede in self.df_students['sede'].unique():
                edad_sede = self.df_students[self.df_students['sede'] == sede]['edad'].dropna()
                axes[4].hist(edad_sede, bins=10, alpha=0.6, label=sede)
            axes[4].set_title('Distribución de Edad por Sede', fontsize=12, weight='bold')
            axes[4].set_xlabel('Edad (años)')
            axes[4].set_ylabel('Frecuencia')
            axes[4].legend()

        # 6. Enfermedades por sede
        if 'enfermedades' in self.df_students.columns and 'sede' in self.df_students.columns:
            cross_table = pd.crosstab(self.df_students['sede'], self.df_students['enfermedades'])

            # Convertir a porcentajes por fila (por sede)
            cross_table_pct = cross_table.div(cross_table.sum(axis=1), axis=0) * 100

            colors = self.get_beautiful_palette(len(cross_table_pct.columns), palette_name='tab20b')
            cross_table_pct.plot(kind='bar', ax=axes[5], color=colors)
            axes[5].set_title('Distribución de Enfermedades por Sede', fontsize=12, weight='bold')
            axes[5].set_ylabel('Porcentaje de Estudiantes (%)')
            axes[5].legend(title='Estado de Salud')
            axes[5].tick_params(axis='x', rotation=45)

            # Agregar valores en las barras con mejor posicionamiento
            bars = axes[5].containers
            for i, bar_group in enumerate(bars):
                for j, bar in enumerate(bar_group):
                    height = bar.get_height()
                    if height > 5:  # Solo mostrar si el valor es significativo (>5%)
                        # Posicionar el texto en la parte superior de la barra
                        axes[5].text(bar.get_x() + bar.get_width()/2, height + 1, 
                                   f'{height:.1f}%', ha='center', va='bottom', 
                                   fontweight='bold', fontsize=8, color='black')
                    elif height > 0:  # Para valores pequeños, mostrar en el centro
                        axes[5].text(bar.get_x() + bar.get_width()/2, height/2, 
                                   f'{height:.1f}%', ha='center', va='center', 
                                   fontweight='bold', fontsize=8, color='white')

        plt.suptitle('Análisis Demográfico de Estudiantes por Sede', fontsize=16, weight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{output_dir}/analisis_demografico.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info("Análisis demográfico completado")

    def create_socioeconomic_analysis(self, output_dir: str):
        self.logger.info("Creando análisis socioeconómico...")

        # Crear figura con subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        # 1. Estrato por sede
        if 'estrato' in self.df_students.columns and 'sede' in self.df_students.columns:
            cross_table = pd.crosstab(self.df_students['estrato'], self.df_students['sede'])

            # Convertir a porcentajes por columna (por sede)
            cross_table_pct = cross_table.div(cross_table.sum(axis=0), axis=1) * 100

            colors = self.get_beautiful_palette(len(cross_table_pct.columns), palette_name='tab20b')

            # Crear gráfico de barras agrupadas manualmente
            x = np.arange(len(cross_table_pct.index))
            width = 0.35

            for i, (sede, color) in enumerate(zip(cross_table_pct.columns, colors)):
                bars = axes[0].bar(x + i * width, cross_table_pct[sede], width, label=sede, color=color)

                # Agregar valores en las barras
                for bar, value in zip(bars, cross_table_pct[sede]):
                    height = bar.get_height()
                    if height > 5:  # Solo mostrar si el valor es significativo
                        axes[0].text(bar.get_x() + bar.get_width()/2, height + 1, 
                                   f'{height:.1f}%', ha='center', va='bottom', 
                                   fontweight='bold', fontsize=8)

            axes[0].set_title('Distribución de Estrato por Sede', fontsize=12, weight='bold')
            axes[0].set_ylabel('Porcentaje de Estudiantes (%)')
            axes[0].set_xlabel('Estrato')
            axes[0].set_xticks(x + width / 2)
            # Convertir etiquetas a enteros para quitar .0
            estrato_labels = [str(int(float(label))) for label in cross_table_pct.index]
            axes[0].set_xticklabels(estrato_labels, rotation=45)
            axes[0].legend(title='Sede')

            # Ajustar límites del eje Y dinámicamente (+10% de espacio)
            max_val = cross_table_pct.values.max()
            axes[0].set_ylim(0, max_val * 1.10)

        # 2. Tipo de vivienda por sede
        if 'tipo_vivienda' in self.df_students.columns and 'sede' in self.df_students.columns:
            cross_table = pd.crosstab(self.df_students['sede'], self.df_students['tipo_vivienda'])

            # Convertir a porcentajes por fila (por sede)
            cross_table_pct = cross_table.div(cross_table.sum(axis=1), axis=0) * 100

            colors = self.get_beautiful_palette(len(cross_table_pct.columns), palette_name='tab20b')
            cross_table_pct.plot(kind='bar', ax=axes[1], color=colors)
            axes[1].set_title('Distribución de Tipo de Vivienda por Sede', fontsize=12, weight='bold')
            axes[1].set_ylabel('Porcentaje de Estudiantes (%)')
            axes[1].set_xlabel('Sede')
            axes[1].legend(title='Tipo Vivienda')
            axes[1].tick_params(axis='x', rotation=45)

            # Agregar valores en las barras
            bars = axes[1].containers
            for bar_group in bars:
                for bar in bar_group:
                    height = bar.get_height()
                    if height > 5:  # Solo mostrar si el valor es significativo
                        axes[1].text(bar.get_x() + bar.get_width()/2, height + 1, 
                                   f'{height:.1f}%', ha='center', va='bottom', 
                                   fontweight='bold', fontsize=8)

            # Ajustar límites del eje Y dinámicamente (+10% de espacio)
            max_val = cross_table_pct.values.max()
            axes[1].set_ylim(0, max_val * 1.10)

        # 3. Zona de vivienda por sede
        if 'zona_vivienda' in self.df_students.columns and 'sede' in self.df_students.columns:
            cross_table = pd.crosstab(self.df_students['sede'], self.df_students['zona_vivienda'])

            # Convertir a porcentajes por fila (por sede)
            cross_table_pct = cross_table.div(cross_table.sum(axis=1), axis=0) * 100

            colors = self.get_beautiful_palette(len(cross_table_pct.columns), palette_name='tab20b')
            cross_table_pct.plot(kind='bar', ax=axes[2], color=colors)
            axes[2].set_title('Distribución de Zona de Vivienda por Sede', fontsize=12, weight='bold')
            axes[2].set_ylabel('Porcentaje de Estudiantes (%)')
            axes[2].set_xlabel('Sede')
            axes[2].legend(title='Zona')
            axes[2].tick_params(axis='x', rotation=45)

            # Agregar valores en las barras
            bars = axes[2].containers
            for bar_group in bars:
                for bar in bar_group:
                    height = bar.get_height()
                    if height > 5:  # Solo mostrar si el valor es significativo
                        axes[2].text(bar.get_x() + bar.get_width()/2, height + 1, 
                                   f'{height:.1f}%', ha='center', va='bottom', 
                                   fontweight='bold', fontsize=8)

            # Ajustar límites del eje Y dinámicamente (+10% de espacio)
            max_val = cross_table_pct.values.max()
            axes[2].set_ylim(0, max_val * 1.10)

        # 4. Medio de transporte por sede
        if 'medio_transporte' in self.df_students.columns and 'sede' in self.df_students.columns:
            cross_table = pd.crosstab(self.df_students['medio_transporte'], self.df_students['sede'])

            # Convertir a porcentajes por columna (por sede)
            cross_table_pct = cross_table.div(cross_table.sum(axis=0), axis=1) * 100

            colors = self.get_beautiful_palette(len(cross_table_pct.columns), palette_name='tab20b')

            # Crear gráfico de barras agrupadas manualmente
            x = np.arange(len(cross_table_pct.index))
            width = 0.35

            for i, (sede, color) in enumerate(zip(cross_table_pct.columns, colors)):
                bars = axes[3].bar(x + i * width, cross_table_pct[sede], width, label=sede, color=color)

                # Agregar valores en las barras
                for bar, value in zip(bars, cross_table_pct[sede]):
                    height = bar.get_height()
                    if height > 5:  # Solo mostrar si el valor es significativo
                        axes[3].text(bar.get_x() + bar.get_width()/2, height + 1, 
                                   f'{height:.1f}%', ha='center', va='bottom', 
                                   fontweight='bold', fontsize=8)

            axes[3].set_title('Distribución de Medio de Transporte por Sede', fontsize=12, weight='bold')
            axes[3].set_ylabel('Porcentaje de Estudiantes (%)')
            axes[3].set_xlabel('Medio de Transporte')
            axes[3].set_xticks(x + width / 2)
            axes[3].set_xticklabels(cross_table_pct.index, rotation=45)
            axes[3].legend(title='Sede')

            # Ajustar límites del eje Y dinámicamente (+10% de espacio)
            max_val = cross_table_pct.values.max()
            axes[3].set_ylim(0, max_val * 1.10)

        # 5. Número de hermanos por sede
        if 'total_hermanos' in self.df_students.columns and 'sede' in self.df_students.columns:
            cross_table = pd.crosstab(self.df_students['total_hermanos'], self.df_students['sede'])

            # Convertir a porcentajes por columna (por sede)
            cross_table_pct = cross_table.div(cross_table.sum(axis=0), axis=1) * 100

            colors = self.get_beautiful_palette(len(cross_table_pct.columns), palette_name='tab20b')

            # Crear gráfico de barras agrupadas manualmente
            x = np.arange(len(cross_table_pct.index))
            width = 0.35

            for i, (sede, color) in enumerate(zip(cross_table_pct.columns, colors)):
                bars = axes[4].bar(x + i * width, cross_table_pct[sede], width, label=sede, color=color)

                # Agregar valores en las barras
                for bar, value in zip(bars, cross_table_pct[sede]):
                    height = bar.get_height()
                    if height > 5:  # Solo mostrar si el valor es significativo
                        axes[4].text(bar.get_x() + bar.get_width()/2, height + 1, 
                                   f'{height:.1f}%', ha='center', va='bottom', 
                                   fontweight='bold', fontsize=8)

            axes[4].set_title('Distribución de Número de Hermanos por Sede', fontsize=12, weight='bold')
            axes[4].set_ylabel('Porcentaje de Estudiantes (%)')
            axes[4].set_xlabel('Número de Hermanos')
            axes[4].set_xticks(x + width / 2)
            # Convertir etiquetas a enteros para quitar .0
            hermanos_labels = [str(int(float(label))) for label in cross_table_pct.index]
            axes[4].set_xticklabels(hermanos_labels, rotation=45)
            axes[4].legend(title='Sede')

            # Ajustar límites del eje Y dinámicamente (+10% de espacio)
            max_val = cross_table_pct.values.max()
            axes[4].set_ylim(0, max_val * 1.10)

        # 6. Composición familiar por sede
        if 'familia' in self.df_students.columns and 'sede' in self.df_students.columns:
            cross_table = pd.crosstab(self.df_students['sede'], self.df_students['familia'])

            # Convertir a porcentajes por fila (por sede)
            cross_table_pct = cross_table.div(cross_table.sum(axis=1), axis=0) * 100

            colors = self.get_beautiful_palette(len(cross_table_pct.columns), palette_name='tab20b')
            cross_table_pct.plot(kind='bar', ax=axes[5], color=colors)
            axes[5].set_title('Distribución de Composición Familiar por Sede', fontsize=12, weight='bold')
            axes[5].set_ylabel('Porcentaje de Estudiantes (%)')
            axes[5].set_xlabel('Sede')
            axes[5].legend(title='Familia')
            axes[5].tick_params(axis='x', rotation=45)

            # Agregar valores en las barras
            bars = axes[5].containers
            for bar_group in bars:
                for bar in bar_group:
                    height = bar.get_height()
                    if height > 5:  # Solo mostrar si el valor es significativo
                        axes[5].text(bar.get_x() + bar.get_width()/2, height + 1, 
                                   f'{height:.1f}%', ha='center', va='bottom', 
                                   fontweight='bold', fontsize=8)

            # Ajustar límites del eje Y dinámicamente (+10% de espacio)
            max_val = cross_table_pct.values.max()
            axes[5].set_ylim(0, max_val * 1.10)

        plt.suptitle('Análisis Socioeconómico de Estudiantes por Sede', fontsize=16, weight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{output_dir}/analisis_socioeconomico.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info("Análisis socioeconómico completado")

    def create_academic_analysis(self, output_dir: str):
        self.logger.info("Creando análisis académico...")

        # Crear figura con subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        # 1. Horas de estudio por sede
        if 'horas_semana_estudio_casa' in self.df_students.columns and 'sede' in self.df_students.columns:
            for sede in self.df_students['sede'].unique():
                horas_sede = self.df_students[self.df_students['sede'] == sede]['horas_semana_estudio_casa'].dropna()
                axes[0].hist(horas_sede, bins=10, alpha=0.6, label=sede)
            axes[0].set_title('Distribución de Horas de Estudio por Sede', fontsize=12, weight='bold')
            axes[0].set_xlabel('Horas por Semana')
            axes[0].set_ylabel('Frecuencia')
            axes[0].legend()

        # 2. Interés en estudios superiores por sede
        if 'interés_estudios_superiores' in self.df_students.columns and 'sede' in self.df_students.columns:
            cross_table = pd.crosstab(self.df_students['sede'], self.df_students['interés_estudios_superiores'])

            # Convertir a porcentajes por fila (por sede)
            cross_table_pct = cross_table.div(cross_table.sum(axis=1), axis=0) * 100

            colors = self.get_beautiful_palette(len(cross_table_pct.columns), palette_name='tab20b')
            cross_table_pct.plot(kind='bar', ax=axes[1], color=colors)
            axes[1].set_title('Distribución de Interés en Estudios Superiores por Sede', fontsize=12, weight='bold')
            axes[1].set_ylabel('Porcentaje de Estudiantes (%)')
            axes[1].set_xlabel('Sede')
            axes[1].legend(title='Interés')
            axes[1].tick_params(axis='x', rotation=45)

            # Agregar valores en las barras
            bars = axes[1].containers
            for bar_group in bars:
                for bar in bar_group:
                    height = bar.get_height()
                    if height > 5:  # Solo mostrar si el valor es significativo
                        axes[1].text(bar.get_x() + bar.get_width()/2, height + 1, 
                                   f'{height:.1f}%', ha='center', va='bottom', 
                                   fontweight='bold', fontsize=8)

            # Ajustar límites del eje Y dinámicamente (+10% de espacio)
            max_val = cross_table_pct.values.max()
            axes[1].set_ylim(0, max_val * 1.10)

        # 3. Actividades extracurriculares por sede
        if 'actividades_extracurriculares' in self.df_students.columns and 'sede' in self.df_students.columns:
            cross_table = pd.crosstab(self.df_students['actividades_extracurriculares'], self.df_students['sede'])

            # Convertir a porcentajes por columna (por sede)
            cross_table_pct = cross_table.div(cross_table.sum(axis=0), axis=1) * 100

            colors = self.get_beautiful_palette(len(cross_table_pct.columns), palette_name='tab20b')

            # Crear gráfico de barras agrupadas manualmente
            x = np.arange(len(cross_table_pct.index))
            width = 0.35

            for i, (sede, color) in enumerate(zip(cross_table_pct.columns, colors)):
                bars = axes[2].bar(x + i * width, cross_table_pct[sede], width, label=sede, color=color)

                # Agregar valores en las barras
                for bar, value in zip(bars, cross_table_pct[sede]):
                    height = bar.get_height()
                    if height > 5:  # Solo mostrar si el valor es significativo
                        axes[2].text(bar.get_x() + bar.get_width()/2, height + 1, 
                                   f'{height:.1f}%', ha='center', va='bottom', 
                                   fontweight='bold', fontsize=8)

            # Acortar etiquetas largas para mejor legibilidad
            labels = []
            for label in cross_table_pct.index:
                if len(label) > 15:
                    # Acortar etiquetas muy largas
                    if 'Artes, Deporte, Idiomas' in label:
                        labels.append('Artes+Deporte+Idiomas')
                    elif 'Artes, Deporte' in label:
                        labels.append('Artes+Deporte')
                    elif 'Artes, Idiomas' in label:
                        labels.append('Artes+Idiomas')
                    elif 'Deporte, Idiomas' in label:
                        labels.append('Deporte+Idiomas')
                    elif 'Tecnología / Diseño' in label:
                        labels.append('Tecnología/Diseño')
                    else:
                        labels.append(label[:15] + '...')
                else:
                    labels.append(label)

            axes[2].set_title('Distribución de Actividades Extracurriculares por Sede', fontsize=12, weight='bold')
            axes[2].set_ylabel('Porcentaje de Estudiantes (%)')
            axes[2].set_xlabel('Actividades Extracurriculares')
            axes[2].set_xticks(x + width / 2)
            axes[2].set_xticklabels(labels, rotation=45, ha='right')
            axes[2].legend(title='Sede')
            axes[2].tick_params(axis='x', labelsize=9)

            # Ajustar límites del eje Y dinámicamente (+10% de espacio)
            max_val = cross_table_pct.values.max()
            axes[2].set_ylim(0, max_val * 1.10)

        # 4. Proyección vocacional por sede
        if 'proyección_vocacional' in self.df_students.columns and 'sede' in self.df_students.columns:
            cross_table = pd.crosstab(self.df_students['proyección_vocacional'], self.df_students['sede'])

            # Convertir a porcentajes por columna (por sede)
            cross_table_pct = cross_table.div(cross_table.sum(axis=0), axis=1) * 100

            colors = self.get_beautiful_palette(len(cross_table_pct.columns), palette_name='tab20b')

            # Crear gráfico de barras agrupadas manualmente
            x = np.arange(len(cross_table_pct.index))
            width = 0.35

            for i, (sede, color) in enumerate(zip(cross_table_pct.columns, colors)):
                bars = axes[3].bar(x + i * width, cross_table_pct[sede], width, label=sede, color=color)

                # Agregar valores en las barras
                for bar, value in zip(bars, cross_table_pct[sede]):
                    height = bar.get_height()
                    if height > 5:  # Solo mostrar si el valor es significativo
                        axes[3].text(bar.get_x() + bar.get_width()/2, height + 1, 
                                   f'{height:.1f}%', ha='center', va='bottom', 
                                   fontweight='bold', fontsize=8)

            # Acortar etiquetas largas para mejor legibilidad
            labels = []
            for label in cross_table_pct.index:
                if len(label) > 20:
                    # Acortar etiquetas muy largas
                    if 'Arte y Entretenimiento' in label:
                        labels.append('Arte y Entretenimiento')
                    elif 'Ciencias de la Salud' in label:
                        labels.append('Ciencias de la Salud')
                    elif 'Diseño y Arquitectura' in label:
                        labels.append('Diseño y Arquitectura')
                    elif 'Indefinido o No Sabe' in label:
                        labels.append('Indefinido/No Sabe')
                    elif 'Negocios y Emprendimiento' in label:
                        labels.append('Negocios/Emprendimiento')
                    elif 'Seguridad y Fuerzas Armadas' in label:
                        labels.append('Seguridad/Fuerzas Armadas')
                    elif 'Sociales y Humanidades' in label:
                        labels.append('Sociales/Humanidades')
                    else:
                        labels.append(label[:20] + '...')
                else:
                    labels.append(label)

            axes[3].set_title('Distribución de Proyección Vocacional por Sede', fontsize=12, weight='bold')
            axes[3].set_ylabel('Porcentaje de Estudiantes (%)')
            axes[3].set_xlabel('Proyección Vocacional')
            axes[3].set_xticks(x + width / 2)
            axes[3].set_xticklabels(labels, rotation=45, ha='right')
            axes[3].legend(title='Sede')
            axes[3].tick_params(axis='x', labelsize=9)

            # Ajustar límites del eje Y dinámicamente (+10% de espacio)
            max_val = cross_table_pct.values.max()
            axes[3].set_ylim(0, max_val * 1.10)

        # 5. Participación en clase por sede
        if 'participación_clase' in self.df_students.columns and 'sede' in self.df_students.columns:
            cross_table = pd.crosstab(self.df_students['sede'], self.df_students['participación_clase'])

            # Convertir a porcentajes por fila (por sede)
            cross_table_pct = cross_table.div(cross_table.sum(axis=1), axis=0) * 100

            colors = self.get_beautiful_palette(len(cross_table_pct.columns), palette_name='tab20b')
            cross_table_pct.plot(kind='bar', ax=axes[4], color=colors)
            axes[4].set_title('Distribución de Participación en Clase por Sede', fontsize=12, weight='bold')
            axes[4].set_ylabel('Porcentaje de Estudiantes (%)')
            axes[4].set_xlabel('Sede')
            axes[4].legend(title='Participación')
            axes[4].tick_params(axis='x', rotation=45)

            # Agregar valores en las barras
            bars = axes[4].containers
            for bar_group in bars:
                for bar in bar_group:
                    height = bar.get_height()
                    if height > 5:  # Solo mostrar si el valor es significativo
                        axes[4].text(bar.get_x() + bar.get_width()/2, height + 1, 
                                   f'{height:.1f}%', ha='center', va='bottom', 
                                   fontweight='bold', fontsize=8)

            # Ajustar límites del eje Y dinámicamente (+10% de espacio)
            max_val = cross_table_pct.values.max()
            axes[4].set_ylim(0, max_val * 1.10)

        # 6. Nivel de motivación por sede
        if 'nivel_motivación' in self.df_students.columns and 'sede' in self.df_students.columns:
            cross_table = pd.crosstab(self.df_students['sede'], self.df_students['nivel_motivación'])

            # Convertir a porcentajes por fila (por sede)
            cross_table_pct = cross_table.div(cross_table.sum(axis=1), axis=0) * 100

            colors = self.get_beautiful_palette(len(cross_table_pct.columns), palette_name='tab20b')
            cross_table_pct.plot(kind='bar', ax=axes[5], color=colors)
            axes[5].set_title('Distribución de Nivel de Motivación por Sede', fontsize=12, weight='bold')
            axes[5].set_ylabel('Porcentaje de Estudiantes (%)')
            axes[5].set_xlabel('Sede')
            axes[5].legend(title='Motivación')
            axes[5].tick_params(axis='x', rotation=45)

            # Agregar valores en las barras
            bars = axes[5].containers
            for bar_group in bars:
                for bar in bar_group:
                    height = bar.get_height()
                    if height > 5:  # Solo mostrar si el valor es significativo
                        axes[5].text(bar.get_x() + bar.get_width()/2, height + 1, 
                                   f'{height:.1f}%', ha='center', va='bottom', 
                                   fontweight='bold', fontsize=8)

            # Ajustar límites del eje Y dinámicamente (+10% de espacio)
            max_val = cross_table_pct.values.max()
            axes[5].set_ylim(0, max_val * 1.10)

        plt.suptitle('Análisis Académico de Estudiantes por Sede', fontsize=16, weight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{output_dir}/analisis_academico.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info("Análisis académico completado")

    def create_psychological_analysis(self, output_dir: str):
        self.logger.info("Creando análisis psicológico...")

        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        # 1. Nivel de motivación por sede
        if 'nivel_motivación' in self.df_students.columns and 'sede' in self.df_students.columns:
            cross_table = pd.crosstab(self.df_students['sede'], self.df_students['nivel_motivación'])
            colors = self.get_beautiful_palette(len(cross_table.columns), palette_name='tab20b')
            cross_table.plot(kind='bar', ax=axes[0], color=colors)
            axes[0].set_title('Distribución de Nivel de Motivación por Sede', fontsize=12, weight='bold')
            axes[0].set_ylabel('Número de Estudiantes')
            axes[0].set_xlabel('Sede')
            axes[0].legend(title='Motivación')
            axes[0].tick_params(axis='x', rotation=45)

        # 2. Apoyo familiar por sede
        if 'apoyo_familiar' in self.df_students.columns and 'sede' in self.df_students.columns:
            cross_table = pd.crosstab(self.df_students['sede'], self.df_students['apoyo_familiar'])
            colors = self.get_beautiful_palette(len(cross_table.columns), palette_name='tab20b')
            cross_table.plot(kind='bar', ax=axes[1], color=colors)
            axes[1].set_title('Distribución de Apoyo Familiar por Sede', fontsize=12, weight='bold')
            axes[1].set_ylabel('Número de Estudiantes')
            axes[1].set_xlabel('Sede')
            axes[1].legend(title='Apoyo')
            axes[1].tick_params(axis='x', rotation=45)

        # 3. NEE por sede
        if 'nee' in self.df_students.columns and 'sede' in self.df_students.columns:
            cross_table = pd.crosstab(self.df_students['sede'], self.df_students['nee'])
            colors = self.get_beautiful_palette(len(cross_table.columns), palette_name='tab20b')
            cross_table.plot(kind='bar', ax=axes[2], color=colors)
            axes[2].set_title('Distribución de NEE por Sede', fontsize=12, weight='bold')
            axes[2].set_ylabel('Número de Estudiantes')
            axes[2].set_xlabel('Sede')
            axes[2].legend(title='NEE')
            axes[2].tick_params(axis='x', rotation=45)

        # 4. Demuestra confianza por sede
        if 'demuestra_confianza' in self.df_students.columns and 'sede' in self.df_students.columns:
            cross_table = pd.crosstab(self.df_students['sede'], self.df_students['demuestra_confianza'])
            colors = self.get_beautiful_palette(len(cross_table.columns), palette_name='tab20b')
            cross_table.plot(kind='bar', ax=axes[3], color=colors)
            axes[3].set_title('Distribución de Demuestra Confianza por Sede', fontsize=12, weight='bold')
            axes[3].set_ylabel('Número de Estudiantes')
            axes[3].set_xlabel('Sede')
            axes[3].legend(title='Confianza')
            axes[3].tick_params(axis='x', rotation=45)

        plt.suptitle('Análisis Psicológico de Estudiantes por Sede', fontsize=16, weight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{output_dir}/analisis_psicologico.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info("Análisis psicológico completado")

    def create_visualizations(self, output_dir: str):
        self.logger.info("Creando visualizaciones de análisis de estudiantes...")

        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)

        # Crear cada tipo de análisis
        self.create_demographic_analysis(output_dir)
        self.create_socioeconomic_analysis(output_dir)
        self.create_academic_analysis(output_dir)
        self.create_psychological_analysis(output_dir)

        self.logger.info("Todas las visualizaciones completadas")

    def run_analysis(self):
        import time

        total_start = time.time()
        self.logger.info("="*60)
        self.logger.info("🚀 Iniciando análisis de estudiantes...")
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

        return {
            'total_students': len(df),
            'total_variables': len(df.columns),
            'categorical_variables': len(self.categorical_columns),
            'numerical_variables': len(self.numerical_columns),
            'ordinal_variables': len(self.ordinal_columns),
            'status': 'completed'
        }


def main():

    parser = argparse.ArgumentParser(description='Análisis exploratorio de datos de estudiantes')
    parser.add_argument('--dataset', '-d', type=str, required=True,
                       help='Ruta al archivo CSV del dataset (requerido)')
    parser.add_argument('--results', '-r', type=str, required=True,
                       help='Nombre del folder para guardar resultados (requerido, se creará en reports/)')

    args = parser.parse_args()

    # Crear y ejecutar analizador siguiendo el patrón estándar
    analyzer = StudentsAnalysis(dataset_path=args.dataset, results_folder=args.results)

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
