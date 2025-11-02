import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.preprocessing.encode_categorical_values import CategoricalEncoder
from scripts.modelling.base_pipeline import BasePipeline
from scripts.modelling.weighted_mae_scorer import default_weighted_mae_scorer

class RandomForestPipeline(BasePipeline):

    def __init__(self, random_state: int = 42):
        self.model_name = "random_forest"
        self.title = "Random Forest"
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
            ('regressor', RandomForestRegressor(
                criterion='absolute_error',  # Entrenar minimizando MAE
                random_state=self.random_state
            ))
        ])
        return pipeline

    def _get_param_grid(self) -> Dict[str, list]:
        return {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [None, 15, 25],
            'regressor__min_samples_split': [2, 10],
            'regressor__min_samples_leaf': [1, 3]
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RandomForestPipeline':
        self.logger.info("=== Entrenando Random Forest con Tuning ===")
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

        # Extraer importancia de características
        self.logger.info("Analizando importancia de características...")
        self._extract_feature_importance()

        # Curvas de aprendizaje
        self.logger.info("Generando curvas de aprendizaje...")
        self._analyze_learning_curves(X, y)

        self.logger.info("✓ Análisis completado exitosamente")