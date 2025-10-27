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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()

        self.fitted_ = False
        self.feature_names_in_ = None

    def _setup_logging(self):
        """Configurar logging básico."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def encode_binary_variables(self, df):
        df_copy = df.copy()
        new_features = []

        # pais_origen -> es_colombiano
        if 'país_origen' in df_copy.columns:
            df_copy['es_colombiano'] = (df_copy['país_origen'] == 'Colombia').astype(int)
            df_copy.drop('país_origen', axis=1, inplace=True)
            new_features.append('es_colombiano')

        # medio_transporte -> medio_transporte_vehiculo_privado  
        if 'medio_transporte' in df_copy.columns:
            df_copy['medio_transporte_vehiculo_privado'] = (df_copy['medio_transporte'] == 'Vehículo privado').astype(int)
            df_copy.drop('medio_transporte', axis=1, inplace=True)
            new_features.append('medio_transporte_vehiculo_privado')

        # tipo_vivienda -> es_alquiler
        if 'tipo_vivienda' in df_copy.columns:
            df_copy['es_alquiler'] = (df_copy['tipo_vivienda'] == 'Alquilada').astype(int)
            df_copy.drop('tipo_vivienda', axis=1, inplace=True)
            new_features.append('es_alquiler')

        # zona_vivienda -> zona_vivienda_urbana
        if 'zona_vivienda' in df_copy.columns:
            df_copy['zona_vivienda_urbana'] = (df_copy['zona_vivienda'] == 'Urbana').astype(int)
            df_copy.drop('zona_vivienda', axis=1, inplace=True)
            new_features.append('zona_vivienda_urbana')

        # rol_adicional -> tiene_rol_adicional
        if 'rol_adicional' in df_copy.columns:
            df_copy['tiene_rol_adicional'] = (df_copy['rol_adicional'] != 'Ninguno').astype(int)
            df_copy.drop('rol_adicional', axis=1, inplace=True)
            new_features.append('tiene_rol_adicional')

        # genero -> es_masculino
        if 'género' in df_copy.columns:
            df_copy['es_masculino'] = (df_copy['género'] == 'Masculino').astype(int)
            df_copy.drop('género', axis=1, inplace=True)
            new_features.append('es_masculino')

        # nee -> tiene_nee
        if 'nee' in df_copy.columns:
            df_copy['tiene_nee'] = (df_copy['nee'] == 'Sí').astype(int)
            df_copy.drop('nee', axis=1, inplace=True)
            new_features.append('tiene_nee')

        # enfermedades -> tiene_enfermedades
        if 'enfermedades' in df_copy.columns:
            df_copy['tiene_enfermedades'] = (df_copy['enfermedades'] == 'Sí').astype(int)
            df_copy.drop('enfermedades', axis=1, inplace=True)
            new_features.append('tiene_enfermedades')

        # antigüedad -> es_antiguo
        if 'antigüedad' in df_copy.columns:
            df_copy['es_antiguo'] = (df_copy['antigüedad'] == 'Antiguo').astype(int)
            df_copy.drop('antigüedad', axis=1, inplace=True)
            new_features.append('es_antiguo')

        return df_copy, new_features

    def encode_ordinal_variables(self, df):
        df_copy = df.copy()
        modified_features = []

        # participacion_clase: Baja=1, Media=2, Alta=3
        if 'participación_clase' in df_copy.columns:
            participacion_map = {'Baja': 1, 'Media': 2, 'Alta': 3}
            df_copy['participacion_clase'] = df_copy['participación_clase'].map(participacion_map)
            df_copy.drop('participación_clase', axis=1, inplace=True)
            modified_features.append('participacion_clase')

        # apoyo_familiar: Bajo=1, Medio=2, Alto=3
        if 'apoyo_familiar' in df_copy.columns:
            apoyo_map = {'Bajo': 1, 'Medio': 2, 'Alto': 3}
            df_copy['apoyo_familiar'] = df_copy['apoyo_familiar'].map(apoyo_map)
            modified_features.append('apoyo_familiar')

        # nivel_motivación: Bajo=1, Medio=2, Alto=3
        if 'nivel_motivación' in df_copy.columns:
            motivacion_map = {'Bajo': 1, 'Medio': 2, 'Alto': 3}
            df_copy['nivel_motivación'] = df_copy['nivel_motivación'].map(motivacion_map)
            modified_features.append('nivel_motivación')

        # time_engagement_level: sin_datos=0, bajo=1, moderado=2, alto=3, muy_alto=4
        if 'time_engagement_level' in df_copy.columns:
            engagement_map = {
                'sin_datos': 0,
                'bajo': 1,
                'moderado': 2,
                'alto': 3,
                'muy_alto': 4
            }
            df_copy['time_engagement_level'] = df_copy['time_engagement_level'].map(engagement_map).fillna(0).astype(int)
            modified_features.append('time_engagement_level')

        # estrato: mantener como ordinal numérico (1-6)
        if 'estrato' in df_copy.columns:
            df_copy['estrato'] = pd.to_numeric(df_copy['estrato'], errors='coerce').fillna(0).astype(int)
            modified_features.append('estrato')

        # period: mantener como ordinal numérico (1, 2, 3, 4)
        if 'period' in df_copy.columns:
            df_copy['period'] = pd.to_numeric(df_copy['period'], errors='coerce').fillna(0).astype(int)
            modified_features.append('period')

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

        # interes_estudios_superiores: Bajo=1, Medio=2, Alto=3
        if 'interés_estudios_superiores' in df_copy.columns:
            interes_map = {'Bajo': 1, 'Medio': 2, 'Alto': 3}
            df_copy['interes_estudios_superiores'] = df_copy['interés_estudios_superiores'].map(interes_map)
            df_copy.drop('interés_estudios_superiores', axis=1, inplace=True)
            modified_features.append('interes_estudios_superiores')

        # nivel_educativo -> educación_profesional: Técnico=0, Pregrado=1, Maestría=1, Otro=0
        if 'nivel_educativo' in df_copy.columns:
            educacion_map = {'Técnico': 0, 'Pregrado': 1, 'Maestría': 1}
            df_copy['educación_profesional'] = df_copy['nivel_educativo'].map(educacion_map).fillna(0).astype(int)
            df_copy.drop('nivel_educativo', axis=1, inplace=True)
            modified_features.append('educación_profesional')
        return df_copy, modified_features

    def encode_dummy_variables(self, df):
        df_copy = df.copy()
        new_features = []

        # actividades_extracurriculares: artes, deportes, idiomas
        if 'actividades_extracurriculares' in df_copy.columns:
            df_copy['actividad_artes'] = df_copy['actividades_extracurriculares'].str.contains('Arte|arte', na=False).astype(int)
            df_copy['actividad_deportes'] = df_copy['actividades_extracurriculares'].str.contains('Deporte|deporte', na=False).astype(int)
            df_copy['actividad_idiomas'] = df_copy['actividades_extracurriculares'].str.contains('Idioma|idioma', na=False).astype(int)
            df_copy.drop('actividades_extracurriculares', axis=1, inplace=True)
            new_features.extend(['actividad_artes', 'actividad_deportes', 'actividad_idiomas'])

        # familia: madre, padre, otros
        if 'familia' in df_copy.columns:
            df_copy['familia_madre'] = df_copy['familia'].str.contains('Mamá|mamá|Madre|madre', na=False).astype(int)
            df_copy['familia_padre'] = df_copy['familia'].str.contains('Papá|papá|Padre|padre', na=False).astype(int)
            df_copy['familia_otros'] = df_copy['familia'].str.contains('otros|hermanos|abuelos|tíos', na=False).astype(int)
            df_copy.drop('familia', axis=1, inplace=True)
            new_features.extend(['familia_madre', 'familia_padre', 'familia_otros'])

        return df_copy, new_features

    def encode_onehot_variables(self, df):
        df_copy = df.copy()
        new_features = []

        # proyeccion_vocacional
        if 'proyección_vocacional' in df_copy.columns:
            dummies = pd.get_dummies(df_copy['proyección_vocacional'], prefix='proyeccion', dummy_na=False, drop_first=True)
            df_copy = pd.concat([df_copy, dummies], axis=1)
            df_copy.drop('proyección_vocacional', axis=1, inplace=True)
            new_features.extend(dummies.columns.tolist())

        # jornada_preferida
        if 'jornada_preferida' in df_copy.columns:
            dummies = pd.get_dummies(df_copy['jornada_preferida'], prefix='jornada', dummy_na=False, drop_first=True)
            df_copy = pd.concat([df_copy, dummies], axis=1)
            df_copy.drop('jornada_preferida', axis=1, inplace=True)
            new_features.extend(dummies.columns.tolist())

        # dia_preferido
        if 'dia_preferido' in df_copy.columns:
            dummies = pd.get_dummies(df_copy['dia_preferido'], prefix='dia', dummy_na=False, drop_first=True)
            df_copy = pd.concat([df_copy, dummies], axis=1)
            df_copy.drop('dia_preferido', axis=1, inplace=True)
            new_features.extend(dummies.columns.tolist())
        return df_copy, new_features

    def keep_numeric_variables(self, df):
        df_copy = df.copy()

        # estrato: ya es numérica 1-6
        if 'estrato' in df_copy.columns:
            df_copy['estrato'] = pd.to_numeric(df_copy['estrato'], errors='coerce')

        # period: ya es numérica 1,2,3,4
        if 'period' in df_copy.columns:
            df_copy['period'] = pd.to_numeric(df_copy['period'], errors='coerce')
        return df_copy

    def encode_categorical_variables(self, df):
        self.logger.info(f"Codificando variables categóricas: {df.shape}")

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

        # Verificar errores críticos
        non_numeric_final = df_encoded.select_dtypes(include=['object']).columns.tolist()
        if non_numeric_final:
            self.logger.error(f"⚠️ COLUMNAS NO NUMÉRICAS EN OUTPUT: {non_numeric_final}")

        # Reportar NaN si existen (solo en caso de error)
        total_nans = df_encoded.isna().sum().sum()
        if total_nans > 0:
            self.logger.warning(f"⚠️ {total_nans} valores NaN después del encoding")

        return df_encoded, all_processed_features

    # Métodos para compatibilidad con sklearn
    def fit(self, X, y=None):
        # Guardar nombres de columnas si es DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()

        # Marcar como ajustado primero
        self.fitted_ = True

        # Hacer una transformación para obtener los nombres de salida
        # Necesitamos los nombres ANTES de convertir a numpy
        if not isinstance(X, pd.DataFrame):
            if self.feature_names_in_ is not None and len(self.feature_names_in_) == X.shape[1]:
                X_df = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        else:
            X_df = X

        # Obtener el DataFrame transformado ANTES de convertir a numpy
        df_encoded, _ = self.encode_categorical_variables(X_df)
        self.feature_names_out_ = df_encoded.columns.tolist()

        return self

    def transform(self, X):
        # Convertir numpy array a DataFrame si es necesario
        if isinstance(X, np.ndarray):
            if hasattr(self, 'feature_names_in_') and self.feature_names_in_ is not None and len(self.feature_names_in_) == X.shape[1]:
                # Si el número de columnas coincide, usar los nombres guardados
                X = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                # Si no coincide, usar nombres genéricos (esto puede pasar en pipelines)
                X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        df_encoded, _ = self.encode_categorical_variables(X)

        # SIEMPRE retornar numpy array float64 para compatibilidad con sklearn
        # Los nombres de columnas se preservan vía get_feature_names_out()
        result = df_encoded.astype(float).values

        return result

    def fit_transform(self, X, y=None):
        """Ajusta y transforma en un solo paso."""
        return self.fit(X, y).transform(X)

    def set_output(self, *, transform=None):
        if transform is not None:
            self._output_format = transform
        return self

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'fitted_')
        if hasattr(self, 'feature_names_out_'):
            return np.array(self.feature_names_out_)
        else:
            return np.array([f'feature_{i}' for i in range(len(self.feature_names_in_))])

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