import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.preprocessing.encode_categorical_values import CategoricalEncoder
from scripts.modelling.base_pipeline import BasePipeline

class CatBoostPipeline(BasePipeline):

    def __init__(self, random_state: int = 42):
        self.model_name = "catboost"
        self.title = "CatBoost"
        super().__init__(random_state)

    def _create_pipeline(self) -> Pipeline:
        # CatBoost maneja nativamente valores faltantes y variables categóricas
        # Mantener transformador numérico mínimo por compatibilidad
        numeric_transformer = Pipeline(steps=[
            ('passthrough', 'passthrough')  # No hacer transformaciones
        ])
        numeric_transformer.set_output(transform="pandas")

        # Transformador para convertir categóricas a string (requerido por CatBoost)
        def convert_to_string(X):
            return X.astype(str)
        
        categorical_transformer = FunctionTransformer(
            convert_to_string, 
            validate=False
        )

        # Preprocessor que mantiene numéricas y convierte categóricas a string
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.num_cols),
                ('cat', categorical_transformer, self.cat_cols)
            ],
            verbose_feature_names_out=False
        )

        # Pipeline completo con modelo
        # CatBoost necesita saber qué columnas son categóricas
        cat_features_indices = list(range(len(self.num_cols), len(self.num_cols) + len(self.cat_cols)))
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', CatBoostRegressor(
                cat_features=cat_features_indices,
                random_state=self.random_state,
                verbose=False, # Silenciar output durante entrenamiento
                allow_writing_files=False # No crear archivos temporales
            ))
        ])
        return pipeline


    def _get_param_grid(self) -> Dict[str, list]:
        # Define la grilla de hiperparámetros para tuning.
        return {
            'regressor__iterations': [100, 300, 500],
            'regressor__depth': [4, 6, 8],
            'regressor__learning_rate': [0.01, 0.1, 0.2],
            'regressor__l2_leaf_reg': [1, 3, 5]
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'CatBoostPipeline':
        self.logger.info("=== Entrenando CatBoost con Tuning ===")
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

        # Mostrar importancia de características
        self.logger.info("Analizando importancia de características...")
        self._extract_feature_importance()

        # Curvas de aprendizaje
        self.logger.info("Generando curvas de aprendizaje...")
        self._analyze_learning_curves(X, y)

        self.logger.info("✓ Análisis completado exitosamente")