import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

from catboost import CatBoostRegressor
from imblearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.preprocessing.encode_categorical_values import CategoricalEncoder
from scripts.modelling.base_pipeline import BasePipeline
from scripts.modelling.weighted_mae_scorer import default_weighted_mae_scorer
from scripts.preprocessing.g_smote import GSMOTERegressor, GSMOTEBalancedRegressor
from scripts.modelling.smote_diagnostic import SMOTEDiagnostic
from sklearn.model_selection import RepeatedKFold, GridSearchCV, StratifiedKFold


# Función global para conversión a string (necesaria para serialización)
def convert_to_string(X):
    """Convierte todas las columnas a string para CatBoost."""
    return X.astype(str)

class CatBoostPipeline(BasePipeline):

    def __init__(self, random_state: int = 42, use_resampling: bool = True):
        self.model_name = "catboost"
        self.title = "CatBoost"
        self.use_resampling = use_resampling

        super().__init__(random_state)
        
        # Sobrescribir SCORING para incluir métrica personalizada
        self.SCORING = {
            'rmse': 'neg_mean_squared_error',
            'mae': 'neg_mean_absolute_error',
            'r2': 'r2',
            'weighted_mae': default_weighted_mae_scorer
        }
     
    def _band_labels(self, y: pd.Series) -> pd.Series:
        """Bandea y en 0–60, 60–70, 70–80, 80–90, 90–100 (cerrando extremos)."""
        return pd.cut(
            y,
            bins=[-np.inf, 60, 70, 80, 90, np.inf],
            labels=['0-60', '60-70', '70-80', '80-90', '90-100'],
            right=True, include_lowest=True
        )

    def _make_sample_weight(self, y: pd.Series) -> np.ndarray:
        """Pesos por banda (ajústalos si quieres ser más/menos agresiva)."""
        bands = self._band_labels(y)
        wmap = {'0-60': 8.0, '60-70': 3.0, '70-80': 1.5, '80-90': 1.0, '90-100': 1.0}
        return bands.map(wmap).astype(float).values

    def _create_pipeline(self) -> Pipeline:
        steps = []

        # Si se usa resampling, hacerlo ANTES del preprocessor
        if self.use_resampling:
            steps.append(('resampler', GSMOTEBalancedRegressor(
                k_neighbors=5,
                #n_synthetic_multiplier=6.0,
                selection_strategy="combined",
                random_state=self.random_state,
            )))

        # 2️⃣ Luego aplicas el preprocesamiento
        numeric_transformer = 'passthrough'
        categorical_transformer = FunctionTransformer(convert_to_string, validate=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.num_cols),
                ('cat', categorical_transformer, self.cat_cols)
            ],
            verbose_feature_names_out=False
        )
        preprocessor.set_output(transform="pandas")

        steps.append(('preprocessor', preprocessor))

        # 3️⃣ CatBoost con índices de columnas categóricas
        cat_features_indices = list(range(len(self.num_cols), len(self.num_cols) + len(self.cat_cols)))

        catboost = CatBoostRegressor(
            loss_function='MAE',
            cat_features=cat_features_indices,
            random_state=self.random_state,
            verbose=False,
            allow_writing_files=False,
            thread_count=1
        )

        steps.append(('regressor', catboost))

        return Pipeline(steps=steps)


    def _get_param_grid(self) -> Dict[str, list]:
        return {
            'regressor__iterations': [300, 500],
            'regressor__depth': [4, 6, 8],
            'regressor__learning_rate': [0.03, 0.1],
            'regressor__l2_leaf_reg': [1, 3]
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'CatBoostPipeline':
        self.logger.info("=== Entrenando CatBoost con Tuning ===")
        self.logger.info(f"Datos: {X.shape[0]} muestras, {X.shape[1]} características")

        base_pipeline = self._create_pipeline()
        param_grid = self._get_param_grid()

        # === NUEVO: CV estratificada por bandas ===
        y_band = self._band_labels(y)
        cv_strat = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        cv_splits = list(cv_strat.split(X, y_band))  # iterable de splits

        self.logger.info("Realizando tuning de hiperparámetros...")
        self.logger.info(f"Parámetros a optimizar: {list(param_grid.keys())}")

        # === Logging de configuración de CV ===
        self._log_cv_configuration(cv_splits, X, y)

        grid_search = GridSearchCV(
            base_pipeline,
            param_grid,
            cv=cv_splits,                      # ← usamos los splits estratificados
            scoring=self.SCORING,
            refit='weighted_mae',
            n_jobs=-1,
            verbose=False,
            return_train_score=True
        )

        # === NUEVO: sample_weight por banda (solo si NO hay resampling) ===
        fit_params = {}
        if not self.use_resampling:
            sample_w = self._make_sample_weight(y)
            fit_params = {"regressor__sample_weight": sample_w}
            self.logger.info("Usando pesos por banda (sin resampling)")
        else:
            self.logger.info("Resampling activo: no se usan pesos adicionales (SMOTE ya balancea)")

        # Barra de progreso
        from sklearn.model_selection import ParameterGrid
        n_candidates = len(list(ParameterGrid(param_grid)))
        n_splits_total = len(cv_splits)
        total_fits = n_candidates * n_splits_total
        self.logger.info(f"Total de combinaciones: {n_candidates} | Splits totales: {n_splits_total} | Fits: {total_fits}")
        
        # Mostrar distribución de pesos solo si se usan
        if not self.use_resampling:
            bands = self._band_labels(y)
            sample_w = self._make_sample_weight(y)
            self.logger.info("Distribución de pesos por banda:")
            self.logger.info(pd.DataFrame({"band": bands, "w": sample_w}).groupby("band")["w"].agg(["count","mean","sum"]).to_string())
        else:
            self.logger.info("Distribución de datos será balanceada por SMOTE")

        from tqdm.auto import tqdm
        from tqdm_joblib import tqdm_joblib
        with tqdm_joblib(tqdm(total=total_fits, desc="GridSearchCV", unit="fit")):
            grid_search.fit(X, y, **fit_params)   # ← pesos pasan al CatBoost dentro del pipeline

        self.pipeline = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        self.logger.info("\n--- Mejores Hiperparámetros ---")
        for param, value in self.best_params.items():
            self.logger.info(f"  {param}: {value}")

        self._extract_cv_results(grid_search)
        
        # Generar diagnóstico de SMOTE si se usó resampling
        if self.use_resampling:
            self._generate_smote_diagnostic(X, y)
        
        self._save_model()
        self.logger.info("\n✓ Entrenamiento completado exitosamente")
        return self
    
    def _generate_smote_diagnostic(self, X: pd.DataFrame, y: pd.Series):
        """Genera un reporte de diagnóstico sobre SMOTE."""
        try:
            self.logger.info("\n=== Generando diagnóstico de SMOTE ===")
            
            # Obtener métricas del resampler si están disponibles
            resampler = self.pipeline.named_steps.get('resampler')
            resampler_metrics = None
            
            if resampler and hasattr(resampler, '_last_diversity_metrics'):
                resampler_metrics = {
                    'diversity': resampler._last_diversity_metrics,
                    'distribution': getattr(resampler, '_last_distribution_metrics', None)
                }
            
            # Obtener datos resampleados simulando el pipeline (solo el resampler)
            try:
                X_resampled, y_resampled = resampler.fit_resample(X, y)
            except Exception as e:
                self.logger.warning(f"No se pudieron obtener datos resampleados para diagnóstico: {e}")
                X_resampled, y_resampled = None, None
            
            # Generar reporte
            diagnostic = SMOTEDiagnostic()
            diagnostic.generate_diagnostic_report(
                model_name=self.model_name,
                X_original=X,
                y_original=y,
                X_resampled=X_resampled,
                y_resampled=y_resampled,
                resampler_metrics=resampler_metrics,
                cv_results_with_resampling=self.cv_results,
                cv_results_without_resampling=None,  # No tenemos comparación sin resampling aquí
                model_ts_dir=self.model_ts_dir
            )
            
        except Exception as e:
            self.logger.warning(f"Error generando diagnóstico de SMOTE: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())

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