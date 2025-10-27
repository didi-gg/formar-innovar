import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.preprocessing.encode_categorical_values import CategoricalEncoder
from scripts.modelling.base_pipeline import BasePipeline

class LinearRegressionPipeline(BasePipeline):

    def __init__(self, random_state: int = 42):
        self.model_name = "linear_regression"
        self.title = "Regresión Lineal"
        super().__init__(random_state)

    def _create_pipeline(self) -> Pipeline:
        # Pipeline para variables numéricas
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        # Preservar nombres de columnas dentro del pipeline
        numeric_transformer.set_output(transform="pandas")

        # Pipeline para variables categóricas
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', CategoricalEncoder())  # ← CategoricalEncoder NECESITA nombres de columnas
        ])
        # Preservar nombres de columnas para que el encoder los reciba
        categorical_transformer.set_output(transform="pandas")

        # Combinar ambos transformadores
        # verbose_feature_names_out=False hace que ColumnTransformer preserve los nombres originales
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.num_cols),
                ('cat', categorical_transformer, self.cat_cols)
            ],
            verbose_feature_names_out=False
        )

        # Pipeline completo con modelo
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

        return pipeline

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LinearRegressionPipeline':
        self.logger.info("=== Entrenando Regresión Lineal ===")
        self.logger.info(f"Datos: {X.shape[0]} muestras, {X.shape[1]} características")

        # Verificar datos de entrada (antes de preprocessing)
        self.logger.info(f"📊 Verificando datos de ENTRADA:")
        self.logger.info(f"  - X tiene NaNs: {X.isnull().sum().sum()}")
        self.logger.info(f"  - y tiene NaNs: {y.isnull().sum()}")
        self.logger.info(f"  - Tipos de X: {X.dtypes.value_counts().to_dict()}")

        # Crear pipeline
        self.pipeline = self._create_pipeline()

        # Entrenar modelo completo
        self.logger.info("Entrenando modelo...")
        self.pipeline.fit(X, y)  # ← Pipeline internamente hace fit + transform en cada paso

        # Verificar datos DESPUÉS del preprocessing (lo que realmente ve el modelo)
        self.logger.info(f"📊 Verificando datos DESPUÉS del preprocessing:")
        X_processed = self.pipeline.named_steps['preprocessor'].transform(X)

        # Obtener nombres de columnas del preprocessor
        try:
            feature_names = self.pipeline.named_steps['preprocessor'].get_feature_names_out()
            self.logger.info(f"  - Feature names disponibles: {len(feature_names)}")
            self.logger.info(f"  - Primeras 10: {list(feature_names[:10])}")
            self.logger.info(f"  - Últimas 10: {list(feature_names[-10:])}")
        except:
            self.logger.warning("  - No se pudieron obtener feature names")

        # Verificar el array procesado
        self.logger.info(f"  - X_processed tipo: {type(X_processed)}")
        self.logger.info(f"  - X_processed shape: {X_processed.shape}")
        self.logger.info(f"  - X_processed dtype: {X_processed.dtype}")
        self.logger.info(f"  - X_processed tiene NaNs: {np.isnan(X_processed).sum()}")
        self.logger.info(f"  - X_processed finitos: {np.isfinite(X_processed).all()}")

        if X_processed.dtype == np.float64 or X_processed.dtype == np.float32:
            self.logger.info(f"  ✅ Datos son numéricos (float)")
        elif X_processed.dtype == np.int64 or X_processed.dtype == np.int32:
            self.logger.info(f"  ✅ Datos son numéricos (int)")
        else:
            self.logger.warning(f"  ⚠️  Datos NO son numéricos: dtype={X_processed.dtype}")

        # Validación cruzada
        self.logger.info("Realizando validación cruzada 5×8...")
        self._cross_validate(X, y)

        # Guardar modelo
        self._save_model()

        self.logger.info("✓ Entrenamiento completado exitosamente")
        return self

    def analyze(self, X: pd.DataFrame, y: pd.Series):
        if self.pipeline is None:
            raise ValueError("El modelo no ha sido entrenado. Llama a fit() primero.")

        self.logger.info("=== Realizando Análisis Completo ===")

        # Análisis de multicolinealidad
        self.logger.info("Analizando multicolinealidad...")
        self._analyze_multicollinearity(X, y)

        # Curvas de aprendizaje
        self.logger.info("Generando curvas de aprendizaje...")
        self._analyze_learning_curves(X, y)
        self.logger.info("✓ Análisis completado exitosamente")