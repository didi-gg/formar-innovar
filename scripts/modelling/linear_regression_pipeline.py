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

        # Pipeline para variables categóricas
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', CategoricalEncoder())
        ])

        # Combinar ambos transformadores
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, self.num_cols),
            ('cat', categorical_transformer, self.cat_cols)
        ])

        # Pipeline completo con modelo
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

        return pipeline

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LinearRegressionPipeline':
        self.logger.info("=== Entrenando Regresión Lineal ===")
        self.logger.info(f"Datos: {X.shape[0]} muestras, {X.shape[1]} características")

        # Crear pipeline
        self.pipeline = self._create_pipeline()

        # Entrenar modelo completo
        self.logger.info("Entrenando modelo...")
        self.pipeline.fit(X, y)  # ← Pipeline internamente hace fit + transform en cada paso

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

# Ejemplo de uso
if __name__ == "__main__":
    # Crear datos de ejemplo con características numéricas y categóricas
    from sklearn.datasets import make_regression

    # Generar datos base
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'num_feature_{i}' for i in range(X.shape[1])])

    # Agregar algunas características categóricas
    np.random.seed(42)

    y_series = pd.Series(y, name='target')

    # Definir columnas numéricas y categóricas
    num_cols = [f'num_feature_{i}' for i in range(10)]
    cat_cols = []

    # Crear instancia del modelo
    model = LinearRegressionPipeline()

    # Establecer las columnas numéricas y categóricas ANTES de fit
    model.num_cols = num_cols
    model.cat_cols = cat_cols

    # Entrenar modelo
    model.fit(X_df, y_series)

    # Realizar análisis (opcional)
    model.analyze(X_df, y_series)

    # Hacer predicciones
    predictions = model.predict(X_df[:10])
    model.logger.info(f"Predicciones: {predictions[:5]}")

    # Obtener métricas
    metrics = model.get_metrics()
    model.logger.info(f"RMSE Test: {metrics['rmse_test'].mean():.4f}")
