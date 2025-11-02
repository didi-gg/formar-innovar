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
from scripts.preprocessing.outlier_handler import OutlierHandler
from scripts.modelling.base_pipeline import BasePipeline
from scripts.modelling.weighted_mae_scorer import default_weighted_mae_scorer

class ElasticNetPipeline(BasePipeline):

    def __init__(self, random_state: int = 42):
        self.model_name = "elasticnet"
        self.title = "Elastic Net"
        super().__init__(random_state)
        
        # Sobrescribir SCORING para incluir métrica personalizada
        self.SCORING = {
            'rmse': 'neg_mean_squared_error',
            'mae': 'neg_mean_absolute_error',
            'r2': 'r2',
            'weighted_mae': default_weighted_mae_scorer
        }

    def _create_pipeline(self) -> Pipeline:
        # Pipeline para variables numéricas
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('outlier_handler', OutlierHandler(iqr_factor=1.5)),
            ('scaler', StandardScaler())
        ])
        numeric_transformer.set_output(transform="pandas")

        # Pipeline para variables categóricas
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', CategoricalEncoder())  # ← CategoricalEncoder NECESITA nombres de columnas
        ])
        categorical_transformer.set_output(transform="pandas")

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
            ('regressor', ElasticNet(
                random_state=self.random_state,
                max_iter=10000,  # Aumentar iteraciones para convergencia
                tol=1e-3  # Tolerancia para convergencia
            ))
        ])

        return pipeline

    def _get_param_grid(self) -> Dict[str, list]:
        # Define la grilla de hiperparámetros para tuning.
        return {
            'regressor__alpha': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
            # Excluir 0.0 y 1.0 para enfocarse en ElasticNet puro (mezcla de L1 y L2)
            # l1_ratio=0.0 es Ridge puro, l1_ratio=1.0 es Lasso puro
            'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
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
            refit='weighted_mae',  # Usar métrica personalizada para seleccionar mejor modelo
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

        # Obtener nombres reales de las características del preprocessor
        try:
            preprocessor = self.pipeline.named_steps['preprocessor']
            feature_names = preprocessor.get_feature_names_out()
            self.logger.info(f"✅ Obtenidos {len(feature_names)} nombres reales de características")
        except Exception as e:
            self.logger.warning(f"⚠️  No se pudieron obtener nombres reales: {e}")
            feature_names = [f'feature_{i}' for i in range(len(coefficients))]

        # Crear DataFrame con importancia
        self.feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values('Abs_Coefficient', ascending=False)

        self.logger.info("Importancia de características:")
        self.logger.info(self.feature_importance.head())

        return self.feature_importance