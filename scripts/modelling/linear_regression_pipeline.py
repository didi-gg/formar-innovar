import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional



from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.preprocessing.encode_categorical_values import CategoricalEncoder
from scripts.preprocessing.outlier_handler import OutlierHandler
from scripts.modelling.base_pipeline import BasePipeline
from scripts.preprocessing.g_smote import GSMOTERegressor

class LinearRegressionPipeline(BasePipeline):
    def __init__(self, random_state: int = 42, use_resampling: bool = True):
        self.model_name = "linear_regression"
        self.title = "Regresión Lineal"
        self.use_resampling = use_resampling

        super().__init__(random_state)

    def _create_pipeline(self) -> Pipeline:
        # --- Pipeline para variables numéricas ---
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('outlier_handler', OutlierHandler(iqr_factor=1.5)),
            ('scaler', StandardScaler())
        ])
        numeric_transformer.set_output(transform="pandas")

        # --- Pipeline para variables categóricas ---
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', CategoricalEncoder())  # Usa nombres de columnas
        ])
        categorical_transformer.set_output(transform="pandas")

        # --- Combinación de ambos ---
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.num_cols),
                ('cat', categorical_transformer, self.cat_cols)
            ],
            verbose_feature_names_out=False
        )
        preprocessor.set_output(transform="pandas")

        # --- Definir pasos del pipeline completo ---
        steps = [('preprocessor', preprocessor)]

        if self.use_resampling:
            steps.append(('resampler', GSMOTERegressor(
                n_synthetic_multiplier=3.0,
                selection_strategy="combined",
                random_state=self.random_state,
            )))

        steps.append(('regressor', LinearRegression()))

        # --- Pipeline final con imblearn (permite fit_resample) ---
        pipeline = Pipeline(steps=steps)
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

        # === 🔧 BLOQUE CORREGIDO ===
        self.logger.info("📊 Verificando datos DESPUÉS del preprocessing:")
        X_processed_df = self.pipeline.named_steps['preprocessor'].transform(X)

        try:
            feature_names = self.pipeline.named_steps['preprocessor'].get_feature_names_out()
            self.logger.info(f"  - Feature names disponibles: {len(feature_names)}")
            self.logger.info(f"  - Primeras 10: {list(feature_names[:10])}")
            self.logger.info(f"  - Últimas 10: {list(feature_names[-10:])}")
        except Exception:
            self.logger.warning("  - No se pudieron obtener feature names")

        self.logger.info(f"  - X_processed tipo: {type(X_processed_df)}")
        self.logger.info(f"  - X_processed shape: {X_processed_df.shape}")
        self.logger.info(f"  - X_processed dtypes (top 5): {X_processed_df.dtypes.astype(str).head().to_dict()}")

        X_proc = X_processed_df.to_numpy()
        self.logger.info(f"  - NaNs totales: {np.isnan(X_proc).sum()}")
        self.logger.info(f"  - ¿Todos finitos?: {np.isfinite(X_proc).all()}")

        non_numeric_cols = X_processed_df.columns[~X_processed_df.dtypes.apply(np.issubdtype, args=(np.number,))]
        if len(non_numeric_cols) == 0:
            self.logger.info("  ✅ Todas las columnas son numéricas")
        else:
            self.logger.warning(f"  ⚠️ Columnas NO numéricas detectadas: {list(non_numeric_cols[:10])}")
        # === 🔧 FIN BLOQUE CORREGIDO ===

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
