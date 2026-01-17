import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.preprocessing.encode_categorical_values import CategoricalEncoder
from scripts.modelling.base_pipeline import BasePipeline
from scripts.modelling.weighted_mae_scorer import default_weighted_mae_scorer
from scripts.preprocessing.g_smote import GSMOTERegressor


class RandomForestPipeline(BasePipeline):

    def __init__(self, random_state: int = 42, use_resampling: bool = True):
        self.model_name = "random_forest"
        self.title = "Random Forest"
        self.use_resampling = use_resampling
        super().__init__(random_state)
        
        # Sobrescribir SCORING para incluir métrica personalizada
        self.SCORING = {
            'rmse': 'neg_mean_squared_error',
            'mae': 'neg_mean_absolute_error',
            'r2': 'r2',
            'weighted_mae': default_weighted_mae_scorer
        }


    def _create_pipeline(self) -> Pipeline:
        # Numéricas
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
        ])
        numeric_transformer.set_output(transform="pandas")

        # Categóricas
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', CategoricalEncoder()) # necesita nombres de columnas
        ])
        categorical_transformer.set_output(transform="pandas")

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.num_cols),
                ('cat', categorical_transformer, self.cat_cols)
            ],
            verbose_feature_names_out=False
        )
        preprocessor.set_output(transform="pandas")

        rf = RandomForestRegressor(
            n_jobs=1,
            max_features='sqrt',
            bootstrap=True,
            max_samples=0.8,
            criterion='absolute_error',
            random_state=self.random_state,
            verbose=False,
            warm_start=True
        )

        steps = [('preprocessor', preprocessor)]

        if self.use_resampling:
            steps.append(('resampler', GSMOTERegressor(
                n_synthetic_multiplier=3.0,
                selection_strategy="combined",
                random_state=self.random_state,
            )))

        steps.append(('regressor', rf))

        pipeline = Pipeline(steps=steps)
        return pipeline

    def _get_param_grid(self) -> Dict[str, list]:
        return {
            'regressor__n_estimators': [80, 120],
            'regressor__max_depth': [12, 18, 24, 32],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4],
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

        # === Logging de configuración de CV ===
        self._log_cv_configuration(cv_inner, X, y)

        grid_search = GridSearchCV(
            base_pipeline,
            param_grid,
            cv=cv_inner,
            scoring=self.SCORING,
            refit='weighted_mae',  # Usar métrica personalizada para seleccionar mejor modelo
            n_jobs=-1,
            verbose=False,
            return_train_score=True
        )

        # Entrenar con tuning (con barra de progreso)
        n_candidates = len(list(ParameterGrid(param_grid)))
        n_splits_total = cv_inner.get_n_splits(X, y)  # RepeatedKFold ya incluye repeats
        total_fits = n_candidates * n_splits_total

        self.logger.info(f"Total de combinaciones: {n_candidates} | Splits totales: {n_splits_total} | Fits: {total_fits}")

        with tqdm_joblib(tqdm(total=total_fits, desc="GridSearchCV", unit="fit")):
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