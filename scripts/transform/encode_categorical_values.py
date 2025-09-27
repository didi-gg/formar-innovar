"""
Script para codificar variables categóricas

Este script aplica las siguientes transformaciones:
- Variables binarias con renombrado
- Codificación ordinal para variables con orden natural
- One-hot encoding para variables categóricas
- Variables dummy para listas separadas por comas
"""

import pandas as pd
import numpy as np
import sys
import os
import logging
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.base_script import BaseScript

class CategoricalEncoder(BaseScript):

    def encode_binary_variables(self, df):
        df_copy = df.copy()
        new_features = []
        self.logger.info("Codificando variables binarias...")

        # pais_origen -> es_colombiano
        if 'país_origen' in df_copy.columns:
            df_copy['es_colombiano'] = (df_copy['país_origen'] == 'Colombia').astype(int)
            df_copy.drop('país_origen', axis=1, inplace=True)
            new_features.append('es_colombiano')
            self.logger.info("país_origen -> es_colombiano")

        # medio_transporte -> medio_transporte_vehiculo_privado  
        if 'medio_transporte' in df_copy.columns:
            df_copy['medio_transporte_vehiculo_privado'] = (df_copy['medio_transporte'] == 'Vehículo privado').astype(int)
            df_copy.drop('medio_transporte', axis=1, inplace=True)
            new_features.append('medio_transporte_vehiculo_privado')
            self.logger.info("medio_transporte -> medio_transporte_vehiculo_privado")

        # tipo_vivienda -> es_alquiler
        if 'tipo_vivienda' in df_copy.columns:
            df_copy['es_alquiler'] = (df_copy['tipo_vivienda'] == 'Alquilada').astype(int)
            df_copy.drop('tipo_vivienda', axis=1, inplace=True)
            new_features.append('es_alquiler')
            self.logger.info("tipo_vivienda -> es_alquiler")

        # zona_vivienda -> zona_vivienda_urbana
        if 'zona_vivienda' in df_copy.columns:
            df_copy['zona_vivienda_urbana'] = (df_copy['zona_vivienda'] == 'Urbana').astype(int)
            df_copy.drop('zona_vivienda', axis=1, inplace=True)
            new_features.append('zona_vivienda_urbana')
            self.logger.info("zona_vivienda -> zona_vivienda_urbana")

        # rol_adicional -> tiene_rol_adicional
        if 'rol_adicional' in df_copy.columns:
            df_copy['tiene_rol_adicional'] = (df_copy['rol_adicional'] != 'Ninguno').astype(int)
            df_copy.drop('rol_adicional', axis=1, inplace=True)
            new_features.append('tiene_rol_adicional')
            self.logger.info("rol_adicional -> tiene_rol_adicional")

        # genero -> es_masculino
        if 'género' in df_copy.columns:
            df_copy['es_masculino'] = (df_copy['género'] == 'Masculino').astype(int)
            df_copy.drop('género', axis=1, inplace=True)
            new_features.append('es_masculino')
            self.logger.info("género -> es_masculino")

        # nee -> tiene_nee
        if 'nee' in df_copy.columns:
            df_copy['tiene_nee'] = (df_copy['nee'] == 'Sí').astype(int)
            df_copy.drop('nee', axis=1, inplace=True)
            new_features.append('tiene_nee')
            self.logger.info("nee -> tiene_nee")

        # enfermedades -> tiene_enfermedades
        if 'enfermedades' in df_copy.columns:
            df_copy['tiene_enfermedades'] = (df_copy['enfermedades'] == 'Sí').astype(int)
            df_copy.drop('enfermedades', axis=1, inplace=True)
            new_features.append('tiene_enfermedades')
            self.logger.info("enfermedades -> tiene_enfermedades")

        # antigüedad -> es_antiguo
        if 'antigüedad' in df_copy.columns:
            df_copy['es_antiguo'] = (df_copy['antigüedad'] == 'Antiguo').astype(int)
            df_copy.drop('antigüedad', axis=1, inplace=True)
            new_features.append('es_antiguo')
            self.logger.info("antigüedad -> es_antiguo")

        return df_copy, new_features

    def encode_ordinal_variables(self, df):
        df_copy = df.copy()
        modified_features = []
        self.logger.info("Codificando variables ordinales...")

        # participacion_clase: Baja=1, Media=2, Alta=3
        if 'participación_clase' in df_copy.columns:
            participacion_map = {'Baja': 1, 'Media': 2, 'Alta': 3}
            df_copy['participacion_clase'] = df_copy['participación_clase'].map(participacion_map)
            df_copy.drop('participación_clase', axis=1, inplace=True)
            modified_features.append('participacion_clase')
            self.logger.info("participación_clase codificada")

        # apoyo_familiar: Bajo=1, Medio=2, Alto=3
        if 'apoyo_familiar' in df_copy.columns:
            apoyo_map = {'Bajo': 1, 'Medio': 2, 'Alto': 3}
            df_copy['apoyo_familiar'] = df_copy['apoyo_familiar'].map(apoyo_map)
            modified_features.append('apoyo_familiar')
            self.logger.info("apoyo_familiar codificada")

        # nivel_motivación: Bajo=1, Medio=2, Alto=3
        if 'nivel_motivación' in df_copy.columns:
            motivacion_map = {'Bajo': 1, 'Medio': 2, 'Alto': 3}
            df_copy['nivel_motivación'] = df_copy['nivel_motivación'].map(motivacion_map)
            modified_features.append('nivel_motivación')
            self.logger.info("nivel_motivación codificada")

        # nivel_confianza: Nunca=0, Rara_vez=1, A_veces=2, Frecuentemente=3, Siempre=4
        if 'demuestra_confianza' in df_copy.columns:
            confianza_map = {
                'Nunca lo demuestra': 0,
                'Rara vez lo demuestra': 1, 
                'A veces lo demuestra': 2,
                'Frecuentemente lo demuestra': 3,
                'Siempre lo demuestra': 4
            }

            df_copy['nivel_confianza'] = df_copy['demuestra_confianza'].map(confianza_map)
            df_copy.drop('demuestra_confianza', axis=1, inplace=True)
            modified_features.append('nivel_confianza')
            self.logger.info("demuestra_confianza -> nivel_confianza_codificada")

        # interes_estudios_superiores: Bajo=1, Medio=2, Alto=3
        if 'interés_estudios_superiores' in df_copy.columns:
            interes_map = {'Bajo': 1, 'Medio': 2, 'Alto': 3}
            df_copy['interes_estudios_superiores'] = df_copy['interés_estudios_superiores'].map(interes_map)
            df_copy.drop('interés_estudios_superiores', axis=1, inplace=True)
            modified_features.append('interes_estudios_superiores')
            self.logger.info("interés_estudios_superiores_codificada")

        # nivel_educativo -> educación_profesional: Técnico=0, Pregrado=1, Maestría=1, Otro=0
        if 'nivel_educativo' in df_copy.columns:
            educacion_map = {'Técnico': 0, 'Pregrado': 1, 'Maestría': 1}
            df_copy['educación_profesional'] = df_copy['nivel_educativo'].map(educacion_map).fillna(0).astype(int)
            df_copy.drop('nivel_educativo', axis=1, inplace=True)
            modified_features.append('educación_profesional')
            self.logger.info("nivel_educativo -> educación_profesional_codificada")
        return df_copy, modified_features

    def encode_dummy_variables(self, df):
        df_copy = df.copy()
        new_features = []
        self.logger.info("Creando variables dummy...")

        # actividades_extracurriculares: artes, deportes, idiomas
        if 'actividades_extracurriculares' in df_copy.columns:
            df_copy['actividad_artes'] = df_copy['actividades_extracurriculares'].str.contains('Arte|arte', na=False).astype(int)
            df_copy['actividad_deportes'] = df_copy['actividades_extracurriculares'].str.contains('Deporte|deporte', na=False).astype(int)
            df_copy['actividad_idiomas'] = df_copy['actividades_extracurriculares'].str.contains('Idioma|idioma', na=False).astype(int)
            df_copy.drop('actividades_extracurriculares', axis=1, inplace=True)
            new_features.extend(['actividad_artes', 'actividad_deportes', 'actividad_idiomas'])
            self.logger.info("actividades_extracurriculares -> dummies creadas")

        # familia: madre, padre, otros
        if 'familia' in df_copy.columns:
            df_copy['familia_madre'] = df_copy['familia'].str.contains('Mamá|mamá|Madre|madre', na=False).astype(int)
            df_copy['familia_padre'] = df_copy['familia'].str.contains('Papá|papá|Padre|padre', na=False).astype(int)
            df_copy['familia_otros'] = df_copy['familia'].str.contains('otros|hermanos|abuelos|tíos', na=False).astype(int)
            df_copy.drop('familia', axis=1, inplace=True)
            new_features.extend(['familia_madre', 'familia_padre', 'familia_otros'])
            self.logger.info("familia -> dummies creadas")

        return df_copy, new_features

    def encode_onehot_variables(self, df):
        df_copy = df.copy()
        new_features = []
        self.logger.info("Aplicando one-hot encoding...")

        # proyeccion_vocacional
        if 'proyección_vocacional' in df_copy.columns:
            dummies = pd.get_dummies(df_copy['proyección_vocacional'], prefix='proyeccion', dummy_na=False)
            df_copy = pd.concat([df_copy, dummies], axis=1)
            df_copy.drop('proyección_vocacional', axis=1, inplace=True)
            new_features.extend(dummies.columns.tolist())
            self.logger.info("proyección_vocacional -> one-hot encoding")

        # jornada_preferida
        if 'jornada_preferida' in df_copy.columns:
            dummies = pd.get_dummies(df_copy['jornada_preferida'], prefix='jornada', dummy_na=False)
            df_copy = pd.concat([df_copy, dummies], axis=1)
            df_copy.drop('jornada_preferida', axis=1, inplace=True)
            new_features.extend(dummies.columns.tolist())
            self.logger.info("jornada_preferida -> one-hot encoding")

        # dia_preferido
        if 'dia_preferido' in df_copy.columns:
            dummies = pd.get_dummies(df_copy['dia_preferido'], prefix='dia', dummy_na=False)
            df_copy = pd.concat([df_copy, dummies], axis=1)
            df_copy.drop('dia_preferido', axis=1, inplace=True)
            new_features.extend(dummies.columns.tolist())
            self.logger.info("dia_preferido -> one-hot encoding")
        return df_copy, new_features

    def keep_numeric_variables(self, df):
        df_copy = df.copy()
        self.logger.info("Verificando variables numéricas...")

        # estrato: ya es numérica 1-6
        if 'estrato' in df_copy.columns:
            df_copy['estrato'] = pd.to_numeric(df_copy['estrato'], errors='coerce')
            self.logger.info("estrato mantenida como numérica")

        # period: ya es numérica 1,2,3,4
        if 'period' in df_copy.columns:
            df_copy['period'] = pd.to_numeric(df_copy['period'], errors='coerce')
            self.logger.info("period mantenida como numérica")
        return df_copy

    def encode_categorical_variables(self, df):
        self.logger.info("=== INICIANDO CODIFICACIÓN DE VARIABLES CATEGÓRICAS ===")
        self.logger.info(f"Shape inicial del dataset: {df.shape}")

        # Aplicar todas las transformaciones en secuencia y recopilar características
        all_processed_features = []

        df_encoded = df.copy()
        df_encoded, binary_features = self.encode_binary_variables(df_encoded)
        all_processed_features.extend(binary_features)

        df_encoded, ordinal_features = self.encode_ordinal_variables(df_encoded) 
        all_processed_features.extend(ordinal_features)

        df_encoded, dummy_features = self.encode_dummy_variables(df_encoded)
        all_processed_features.extend(dummy_features)

        df_encoded, onehot_features = self.encode_onehot_variables(df_encoded)
        all_processed_features.extend(onehot_features)

        df_encoded = self.keep_numeric_variables(df_encoded)

        # Mostrar resumen de transformaciones
        self.print_encoding_summary(df, df_encoded, all_processed_features)

        self.logger.info("=== CODIFICACIÓN COMPLETADA ===")
        return df_encoded, all_processed_features

    def print_encoding_summary(self, df_original, df_encoded, processed_features):
        """
        Imprime un resumen de las transformaciones aplicadas.

        Args:
            df_original: DataFrame original
            df_encoded: DataFrame codificado
            processed_features: Lista de características procesadas
        """
        self.logger.info("\n=== RESUMEN DE CODIFICACIÓN ===")
        self.logger.info(f"Shape original: {df_original.shape}")
        self.logger.info(f"Shape final: {df_encoded.shape}")
        self.logger.info(f"Columnas añadidas: {df_encoded.shape[1] - df_original.shape[1]}")

        # Información sobre características procesadas
        self.logger.info(f"Total de características procesadas: {len(processed_features)}")
        self.logger.info(f"Características procesadas: {processed_features}")

        # Contar variables por tipo de codificación
        binary_vars = [col for col in processed_features if col.startswith(('es_', 'tiene_', 'zona_vivienda_urbana', 'medio_transporte_vehiculo_privado', 'es_alquiler'))]
        ordinal_vars = [col for col in processed_features if col in ['participacion_clase', 'apoyo_familiar', 'nivel_motivación', 'nivel_confianza', 'interes_estudios_superiores', 'educación_profesional']]
        dummy_vars = [col for col in processed_features if col.startswith(('actividad_', 'familia_'))]
        onehot_vars = [col for col in processed_features if col.startswith(('proyeccion_', 'jornada_', 'dia_'))]

        self.logger.info(f"Variables binarias creadas: {len(binary_vars)}")
        self.logger.info(f"Variables ordinales: {len(ordinal_vars)}")
        self.logger.info(f"Variables dummy: {len(dummy_vars)}")
        self.logger.info(f"Variables one-hot: {len(onehot_vars)}")

        # Verificar valores perdidos
        missing_count = df_encoded.isnull().sum().sum()
        self.logger.info(f"Total de valores perdidos: {missing_count}")


def encode_categorical_variables(df, logger=None):
    encoder = CategoricalEncoder(logger=logger)
    return encoder.encode_categorical_variables(df)


def main():
    """Función principal para uso desde línea de comandos."""
    import argparse

    parser = argparse.ArgumentParser(description='Codificar variables categóricas del dataset FICC')
    parser.add_argument('--input', '-i', 
                       default='data/interim/full_short_dataset_moodle.csv',
                       help='Archivo CSV de entrada')
    parser.add_argument('--output', '-o',
                       help='Archivo CSV de salida (opcional)')

    args = parser.parse_args()

    # Cargar datos
    print(f"Cargando datos desde: {args.input}")
    df = pd.read_csv(args.input)

    # Codificar variables
    df_encoded, processed_features = encode_categorical_variables(df)
    return df_encoded, processed_features

if __name__ == "__main__":
    main()