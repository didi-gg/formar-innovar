"""
Handler para manejo de valores atípicos usando método IQR

Este script implementa un transformador compatible con sklearn que:
- Detecta valores atípicos usando el método del Rango Intercuartílico (IQR)
- Reemplaza valores extremos con la mediana de cada columna
- Mantiene compatibilidad con pipelines de sklearn
"""

import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from typing import Optional, Union


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Maneja valores atípicos usando el método IQR (Rango Intercuartílico).

    Parámetros:
    -----------
    iqr_factor : float, default=1.5
        Factor multiplicador para el IQR. Valores típicos:
        - 1.5: Detección estándar de outliers
        - 3.0: Detección más conservadora (solo outliers extremos)

    Funcionamiento:
    ---------------
    1. Calcula Q1 (percentil 25) y Q3 (percentil 75) para cada columna
    2. Calcula IQR = Q3 - Q1
    3. Define límites: [Q1 - iqr_factor*IQR, Q3 + iqr_factor*IQR]
    4. Reemplaza valores fuera de estos límites con la mediana
    """

    def __init__(self, iqr_factor: float = 1.5):
        self.iqr_factor = iqr_factor
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()

        # Atributos que se calculan durante fit
        self.fitted_ = False
        self.feature_names_in_ = None
        self.lower_bounds_ = None
        self.upper_bounds_ = None
        self.medians_ = None
        self.outlier_stats_ = {}

    def _setup_logging(self):
        """Configurar logging básico."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> 'OutlierHandler':
        # Convertir a DataFrame si es necesario
        if isinstance(X, np.ndarray):
            if hasattr(self, 'feature_names_in_') and self.feature_names_in_ is not None:
                X_df = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        else:
            X_df = X.copy()

        # Guardar nombres de columnas
        self.feature_names_in_ = X_df.columns.tolist()

        # Seleccionar solo columnas numéricas
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            self.logger.warning("No se encontraron columnas numéricas para procesar")
            self.lower_bounds_ = pd.Series(dtype=float)
            self.upper_bounds_ = pd.Series(dtype=float)
            self.medians_ = pd.Series(dtype=float)
            self.fitted_ = True
            return self

        self.logger.info(f"Calculando límites IQR para {len(numeric_cols)} columnas numéricas")
        self.logger.info(f"Factor IQR: {self.iqr_factor}")

        # Calcular estadísticas para cada columna numérica
        self.lower_bounds_ = pd.Series(index=numeric_cols, dtype=float)
        self.upper_bounds_ = pd.Series(index=numeric_cols, dtype=float)
        self.medians_ = pd.Series(index=numeric_cols, dtype=float)

        for col in numeric_cols:
            # Obtener datos válidos (sin NaN)
            valid_data = X_df[col].dropna()

            if len(valid_data) == 0:
                self.logger.warning(f"Columna '{col}' no tiene datos válidos")
                continue

            # Calcular cuartiles
            Q1 = valid_data.quantile(0.25)
            Q3 = valid_data.quantile(0.75)
            IQR = Q3 - Q1

            # Calcular límites
            lower_bound = Q1 - self.iqr_factor * IQR
            upper_bound = Q3 + self.iqr_factor * IQR

            # Calcular mediana
            median_val = valid_data.median()

            # Guardar estadísticas
            self.lower_bounds_[col] = lower_bound
            self.upper_bounds_[col] = upper_bound
            self.medians_[col] = median_val

            # Contar outliers para estadísticas
            outliers_count = ((valid_data < lower_bound) | (valid_data > upper_bound)).sum()
            outliers_pct = (outliers_count / len(valid_data)) * 100

            self.outlier_stats_[col] = {
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'median': median_val,
                'outliers_count': outliers_count,
                'outliers_percentage': outliers_pct,
                'total_values': len(valid_data)
            }

            self.logger.info(f"  {col}: {outliers_count} outliers ({outliers_pct:.1f}%) "
                           f"fuera de [{lower_bound:.2f}, {upper_bound:.2f}]")

        self.fitted_ = True
        self.logger.info("✓ Cálculo de límites IQR completado")
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        check_is_fitted(self, 'fitted_')

        # Convertir a DataFrame si es necesario
        if isinstance(X, np.ndarray):
            if self.feature_names_in_ is not None and len(self.feature_names_in_) == X.shape[1]:
                X_df = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        else:
            X_df = X.copy()

        # Aplicar transformación solo a columnas que tienen límites calculados
        total_replacements = 0

        for col in self.lower_bounds_.index:
            if col not in X_df.columns:
                continue

            # Identificar outliers
            lower_bound = self.lower_bounds_[col]
            upper_bound = self.upper_bounds_[col]
            median_val = self.medians_[col]

            # Crear máscara de outliers
            outlier_mask = (X_df[col] < lower_bound) | (X_df[col] > upper_bound)

            # Contar reemplazos
            replacements = outlier_mask.sum()
            total_replacements += replacements

            # Reemplazar outliers con mediana
            if replacements > 0:
                X_df.loc[outlier_mask, col] = median_val
                self.logger.debug(f"  {col}: {replacements} valores reemplazados con mediana {median_val:.2f}")

        if total_replacements > 0:
            self.logger.info(f"✓ {total_replacements} valores atípicos reemplazados con mediana")

        # Retornar como numpy array para compatibilidad con sklearn
        return X_df.values.astype(float)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, 'fitted_')
        if self.feature_names_in_ is not None:
            return np.array(self.feature_names_in_)
        else:
            return np.array([f'feature_{i}' for i in range(len(self.feature_names_in_))])

    def get_outlier_stats(self) -> dict:
        check_is_fitted(self, 'fitted_')
        return self.outlier_stats_.copy()

    def set_output(self, *, transform=None):
        if transform is not None:
            self._output_format = transform
        return self
