import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.preprocessing.encode_categorical_values import CategoricalEncoder
from scripts.modelling.base_pipeline import BasePipeline

class ElasticNetPipeline(BasePipeline):

    def __init__(self, random_state: int = 42):
        self.model_name = "elasticnet"
        self.title = "Elastic Net"
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
            ('regressor', ElasticNet(random_state=self.random_state))
        ])

        return pipeline

    def _get_param_grid(self) -> Dict[str, list]:
        # Define la grilla de hiperparámetros para tuning.
        return {
            'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ElasticNetPipeline':
        self.logger.info("=== Entrenando ElasticNet con Tuning ===")
        self.logger.info(f"Datos: {X.shape[0]} muestras, {X.shape[1]} características")

        # Crear pipeline base
        base_pipeline = self._create_pipeline()

        # Configurar GridSearchCV
        param_grid = self._get_param_grid()
        cv_inner = RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state)

        self.logger.info("Realizando tuning de hiperparámetros...")
        self.logger.info(f"Parámetros a optimizar: {list(param_grid.keys())}")

        grid_search = GridSearchCV(
            base_pipeline,
            param_grid,
            cv=cv_inner,
            scoring=self.SCORING,
            refit='rmse',
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )

        # Entrenar con tuning
        grid_search.fit(X, y)

        # Guardar mejor modelo y parámetros
        self.pipeline = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        self.logger.info("\n--- Mejores Hiperparámetros ---")
        for param, value in self.best_params.items():
            self.logger.info(f"  {param}: {value}")

        # Extraer resultados del mejor modelo en formato compatible
        self._extract_cv_results(grid_search)

        # Guardar modelo
        self._save_model()

        self.logger.info("\n✓ Entrenamiento completado exitosamente")
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

        # Extraer importancia de características
        self.logger.info("Analizando importancia de características...")
        self.get_feature_importance()

        self.logger.info("✓ Análisis completado exitosamente")


    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if self.pipeline is None:
            return None

        # Obtener coeficientes del modelo ElasticNet
        regressor = self.pipeline.named_steps['regressor']
        coefficients = regressor.coef_

        # Crear DataFrame con importancia
        self.feature_importance = pd.DataFrame({
            'Feature': [f'feature_{i}' for i in range(len(coefficients))],
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values('Abs_Coefficient', ascending=False)

        self.logger.info("Importancia de características:")
        self.logger.info(self.feature_importance.head())

        return self.feature_importance



# Ejemplo de uso
if __name__ == "__main__":
    # Crear datos de ejemplo
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

    # Entrenar modelo
    model = ElasticNetPipeline()
    model.num_cols = num_cols
    model.cat_cols = cat_cols
    model.fit(X_df, y_series)

    # Realizar análisis (opcional)
    model.analyze(X_df, y_series)

    # Hacer predicciones
    predictions = model.predict(X_df[:10])
    print(f"Predicciones: {predictions[:5]}")

    # Obtener métricas y parámetros
    metrics = model.get_metrics()
    best_params = model.get_best_params()
    print(f"RMSE Test: {metrics['rmse_test'].mean():.4f}")
    print(f"Mejores parámetros: {best_params}")

    # Obtener importancia de características
    feature_importance = model.get_feature_importance()
    print("Top 5 características más importantes:")
    print(feature_importance.head())
