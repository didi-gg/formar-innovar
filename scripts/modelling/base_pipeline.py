from sklearn.base import clone as sk_clone
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import logging
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
from sklearn.model_selection import learning_curve
from typing import Dict, Any, Optional
from sklearn.model_selection import RepeatedKFold, cross_validate
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class BasePipeline(ABC):

    MODELS_DIR = "models"

    SCORING = {
        'rmse': 'neg_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'r2': 'r2'
    }

    model_name = None
    title = None
    pipeline = None
    cv_results = None
    learning_curve_results = None
    vif_analysis = None
    num_cols = None
    cat_cols = None
    n_splits = 4
    n_repeats = 5
    feature_importance = None
    best_params = None


    def __init__(self, random_state: int = 42):

        self.random_state = random_state
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._create_directories()

        self._setup_logger()

    def _create_directories(self):
        self.models_dir = Path("models")
        self.model_specific_dir = os.path.join(self.models_dir, self.model_name)
        self.model_ts_dir = os.path.join(self.model_specific_dir, self.timestamp)

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.model_specific_dir, exist_ok=True)
        os.makedirs(self.model_ts_dir, exist_ok=True)

    def _setup_logger(self):
        self.logger = logging.getLogger(f"{self.model_name}_{self.timestamp}")
        self.logger.setLevel(logging.INFO)

        # Evitar duplicar handlers si ya existen
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Formato de logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Handler para archivo
        log_file = os.path.join(self.model_ts_dir, f"log.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # Handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # Agregar handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    @abstractmethod
    def _create_pipeline(self):
        pass

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass

    @abstractmethod
    def analyze(self, X: pd.DataFrame, y: pd.Series):
        pass

    def _log_cv_configuration(self, cv, X: pd.DataFrame, y: pd.Series):
        """
        Registra la configuración de validación cruzada y detalles de cada fold.
        Maneja tanto objetos CV con método .split() como listas de tuplas pre-computadas.
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"📊 CONFIGURACIÓN VALIDACIÓN CRUZADA")
        self.logger.info(f"{'='*80}")
        
        # Detectar si cv es una lista de tuplas (splits pre-computados) o un objeto CV
        is_list_of_tuples = isinstance(cv, list) and len(cv) > 0 and isinstance(cv[0], tuple) and len(cv[0]) == 2
        
        # Obtener los splits según el tipo de cv
        if is_list_of_tuples:
            # cv es una lista de tuplas pre-computadas: [(train_idx, test_idx), ...]
            splits_list = cv
            n_total_splits = len(cv)
        elif hasattr(cv, 'split'):
            # cv es un objeto con método split() (RepeatedKFold, StratifiedKFold, etc.)
            splits_list = list(cv.split(X, y))
            n_total_splits = len(splits_list)
        elif hasattr(cv, 'get_n_splits'):
            # cv tiene método get_n_splits pero no split (caso especial)
            n_total_splits = cv.get_n_splits(X, y)
            splits_list = list(cv) if hasattr(cv, '__iter__') else []
        else:
            # Intentar convertir a lista
            splits_list = list(cv) if hasattr(cv, '__iter__') else []
            n_total_splits = len(splits_list)
        
        self.logger.info(f"Esquema: {self.n_splits}-Fold × {self.n_repeats} Repeticiones = {n_total_splits} folds totales")
        self.logger.info(f"Total de datos: {len(X)} registros\n")
        
        # Iterar sobre los splits
        for i, (train_idx, test_idx) in enumerate(splits_list):
            
            n_train = len(train_idx)
            n_test = len(test_idx)
            total = n_train + n_test
            pct_train = 100 * n_train / total
            pct_test = 100 * n_test / total
            
            y_train_fold = y.iloc[train_idx]
            y_test_fold = y.iloc[test_idx]
            
            self.logger.info(f"Fold {i+1:2d}/{n_total_splits}:")
            self.logger.info(f"  📦 Train: {n_train:5d} registros ({pct_train:5.1f}%) | "
                           f"y_mean={y_train_fold.mean():.2f}, y_std={y_train_fold.std():.2f}")
            self.logger.info(f"  🧪 Test:  {n_test:5d} registros ({pct_test:5.1f}%) | "
                           f"y_mean={y_test_fold.mean():.2f}, y_std={y_test_fold.std():.2f}")

        self.logger.info(f"\n{'='*80}\n")

    @abstractmethod
    def _cross_validate(self, X: pd.DataFrame, y: pd.Series):
        pass

    def _cross_validate(self, X: pd.DataFrame, y: pd.Series):
        pipeline_for_cv = self._create_pipeline()

        # Configurar validación cruzada 5×8
        cv = RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state)
        
        # Logging detallado de splits
        self._log_cv_configuration(cv, X, y)

        # Realizar validación cruzada con pipeline limpio
        cv_results = cross_validate(
            pipeline_for_cv, X, y,
            cv=cv,
            scoring=self.SCORING,
            n_jobs=-1,
            return_train_score=True,
            error_score=np.nan  # Continuar aunque fallen algunos folds
        )

        self.logger.info(f"Claves disponibles en cv_results: {list(cv_results.keys())}")
        # Identificar qué folds fallaron
        nan_folds = np.where(np.isnan(cv_results['test_rmse']))[0]
        if len(nan_folds) > 0:
            self.logger.warning(f"⚠️  Folds que fallaron (índices): {nan_folds.tolist()}")
            self.logger.warning(f"   Causa probable: Matriz singular debido a multicolinealidad extrema")

        # Procesar resultados - las claves vienen del diccionario SCORING
        self.cv_results = {
            'rmse_test': np.sqrt(-cv_results['test_rmse']),
            'mae_test': -cv_results['test_mae'],
            'r2_test': cv_results['test_r2'],
            'rmse_train': np.sqrt(-cv_results['train_rmse']),
            'mae_train': -cv_results['train_mae'],
            'r2_train': cv_results['train_r2']
        }
        
        # Agregar Weighted MAE si está disponible
        if 'test_weighted_mae' in cv_results:
            self.cv_results['weighted_mae_test'] = -cv_results['test_weighted_mae']
            self.cv_results['weighted_mae_train'] = -cv_results['train_weighted_mae']

        # Mostrar resultados
        self.logger.info("--- Resultados Validación Cruzada (5×8) ---")

        # Contar y reportar NaN si existen
        metrics_to_report = ['rmse', 'mae', 'r2']
        if 'weighted_mae_test' in self.cv_results:
            metrics_to_report.append('weighted_mae')
            
        for metric in metrics_to_report:
            test_scores = self.cv_results[f'{metric}_test']
            train_scores = self.cv_results[f'{metric}_train']

            # Contar NaN
            n_nan_test = np.isnan(test_scores).sum()
            n_nan_train = np.isnan(train_scores).sum()

            if n_nan_test > 0 or n_nan_train > 0:
                self.logger.warning(f"{metric.upper()} - NaN encontrados: Test={n_nan_test}/{len(test_scores)}, Train={n_nan_train}/{len(train_scores)}")

            # Formatear nombre de métrica para mostrar
            metric_display = "WEIGHTED MAE" if metric == 'weighted_mae' else metric.upper()
            self.logger.info(f"{metric_display}:")
            self.logger.info(f"  Test:  {np.nanmean(test_scores):.4f} ± {np.nanstd(test_scores):.4f}")
            self.logger.info(f"  Train: {np.nanmean(train_scores):.4f} ± {np.nanstd(train_scores):.4f}")

    def _extract_cv_results(self, grid_search: GridSearchCV):
        # Extraer los resultados de validación cruzada del mejor modelo en formato compatible
        best_idx = grid_search.best_index_

        # Extraer los splits individuales para el mejor modelo
        cv_results = grid_search.cv_results_

        split_keys = [k for k in cv_results.keys() if k.startswith('split')]

        # Identificar cuántos splits hay
        n_splits = grid_search.n_splits_

        # Extraer scores de test para cada métrica
        rmse_test = []
        mae_test = []
        r2_test = []

        rmse_train = []
        mae_train = []
        r2_train = []

        for i in range(n_splits):
            # Test scores - las claves vienen del diccionario SCORING
            try:
                rmse_test.append(np.sqrt(-cv_results[f'split{i}_test_rmse'][best_idx]))
            except KeyError:
                self.logger.warning(f"No se encontró split{i}_test_rmse")
                rmse_test.append(np.nan)

            try:
                mae_test.append(-cv_results[f'split{i}_test_mae'][best_idx])
            except KeyError:
                self.logger.warning(f"No se encontró split{i}_test_mae")
                mae_test.append(np.nan)

            try:
                r2_test.append(cv_results[f'split{i}_test_r2'][best_idx])
            except KeyError:
                self.logger.warning(f"No se encontró split{i}_test_r2")
                r2_test.append(np.nan)

            # Train scores
            try:
                rmse_train.append(np.sqrt(-cv_results[f'split{i}_train_rmse'][best_idx]))
            except KeyError:
                self.logger.warning(f"No se encontró split{i}_train_rmse")
                rmse_train.append(np.nan)

            try:
                mae_train.append(-cv_results[f'split{i}_train_mae'][best_idx])
            except KeyError:
                self.logger.warning(f"No se encontró split{i}_train_mae")
                mae_train.append(np.nan)

            try:
                r2_train.append(cv_results[f'split{i}_train_r2'][best_idx])
            except KeyError:
                self.logger.warning(f"No se encontró split{i}_train_r2")
                r2_train.append(np.nan)

        # Guardar en formato compatible con base_pipeline
        self.cv_results = {
            'rmse_test': np.array(rmse_test),
            'mae_test': np.array(mae_test),
            'r2_test': np.array(r2_test),
            'rmse_train': np.array(rmse_train),
            'mae_train': np.array(mae_train),
            'r2_train': np.array(r2_train)
        }

        # Mostrar resultados
        self.logger.info("--- Resultados Validación Cruzada (5×8) ---")

        # Contar y reportar NaN si existen
        for metric in ['rmse', 'mae', 'r2']:
            test_scores = self.cv_results[f'{metric}_test']
            train_scores = self.cv_results[f'{metric}_train']

            # Contar NaN
            n_nan_test = np.isnan(test_scores).sum()
            n_nan_train = np.isnan(train_scores).sum()

            if n_nan_test > 0 or n_nan_train > 0:
                self.logger.warning(f"{metric.upper()} - NaN encontrados: Test={n_nan_test}/{len(test_scores)}, Train={n_nan_train}/{len(train_scores)}")

            self.logger.info(f"{metric.upper()}:")
            # Usar nanmean y nanstd para ignorar NaN
            self.logger.info(f"  Test:  {np.nanmean(test_scores):.4f} ± {np.nanstd(test_scores):.4f}")
            self.logger.info(f"  Train: {np.nanmean(train_scores):.4f} ± {np.nanstd(train_scores):.4f}")

    def _analyze_learning_curves(self, X: pd.DataFrame, y: pd.Series):
        try:
            # Calcular curva de aprendizaje usando MAE
            # Nota: Empezar desde 15% para evitar conjuntos demasiado pequeños
            estimator = sk_clone(self.pipeline) if self.pipeline is not None else sk_clone(self._create_pipeline())
            # (opcional) etiqueta de contexto para ver logs del SMOTE en las curvas
            try:
                estimator.set_params(resampler__context="learning_curve")
            except Exception:
                pass

            train_sizes = np.linspace(0.15, 1.0, 10)
            train_sizes_abs, train_scores, val_scores = learning_curve(
                estimator,
                X, y,
                train_sizes=train_sizes,
                cv=self.n_splits,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                shuffle=True,
                random_state=self.random_state,
                error_score=np.nan
            )

            # Convertir a MAE positivo
            train_mae = -train_scores
            val_mae = -val_scores

            # Calcular estadísticas (ignorando NaN de folds fallidos)
            train_mae_mean = np.nanmean(train_mae, axis=1)
            train_mae_std = np.nanstd(train_mae, axis=1)
            val_mae_mean = np.nanmean(val_mae, axis=1)
            val_mae_std = np.nanstd(val_mae, axis=1)

            # Guardar resultados base
            self.learning_curve_results = {
                'train_sizes': train_sizes_abs,
                'train_mae_mean': train_mae_mean,
                'train_mae_std': train_mae_std,
                'val_mae_mean': val_mae_mean,
                'val_mae_std': val_mae_std
            }

            # Calcular curvas de aprendizaje para Weighted MAE si está disponible
            if hasattr(self, 'SCORING') and 'weighted_mae' in self.SCORING:
                try:
                    self.logger.info("Calculando curvas de aprendizaje para Weighted MAE...")
                    train_sizes_abs_w, train_scores_w, val_scores_w = learning_curve(
                        self.pipeline, X, y,
                        train_sizes=train_sizes,
                        cv=self.n_splits,
                        scoring=self.SCORING['weighted_mae'],
                        n_jobs=-1,
                        random_state=self.random_state,
                        error_score=np.nan
                    )

                    # Convertir a Weighted MAE positivo (scorer devuelve valores negativos)
                    train_wmae = -train_scores_w
                    val_wmae = -val_scores_w

                    # Calcular estadísticas
                    train_wmae_mean = np.nanmean(train_wmae, axis=1)
                    train_wmae_std = np.nanstd(train_wmae, axis=1)
                    val_wmae_mean = np.nanmean(val_wmae, axis=1)
                    val_wmae_std = np.nanstd(val_wmae, axis=1)

                    # Agregar a resultados
                    self.learning_curve_results.update({
                        'train_wmae_mean': train_wmae_mean,
                        'train_wmae_std': train_wmae_std,
                        'val_wmae_mean': val_wmae_mean,
                        'val_wmae_std': val_wmae_std
                    })

                except Exception as e:
                    self.logger.warning(f"No se pudo calcular curva de aprendizaje para Weighted MAE: {e}")

            # Crear y guardar gráfico(s)
            self._plot_learning_curves()

        except Exception as e:
            self.logger.error(f"Error analizando curvas de aprendizaje: {e}")
            self.learning_curve_results = None

    def _plot_learning_curves(self):
        if self.learning_curve_results is None:
            return

        lc = self.learning_curve_results
        
        # Verificar si tenemos datos de Weighted MAE
        has_weighted_mae = all(key in lc for key in ['train_wmae_mean', 'val_wmae_mean'])
        
        if has_weighted_mae:
            # Crear figura con subplots para ambas métricas
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Gráfico 1: MAE estándar
            ax1.plot(lc['train_sizes'], lc['train_mae_mean'], 'o-', color='blue', 
                    label='Entrenamiento', linewidth=2, markersize=6)
            ax1.fill_between(lc['train_sizes'], 
                           lc['train_mae_mean'] - lc['train_mae_std'],
                           lc['train_mae_mean'] + lc['train_mae_std'],
                           alpha=0.2, color='blue')

            ax1.plot(lc['train_sizes'], lc['val_mae_mean'], 'o-', color='red', 
                    label='Validación', linewidth=2, markersize=6)
            ax1.fill_between(lc['train_sizes'],
                           lc['val_mae_mean'] - lc['val_mae_std'],
                           lc['val_mae_mean'] + lc['val_mae_std'],
                           alpha=0.2, color='red')

            ax1.set_xlabel('Tamaño del Conjunto de Entrenamiento', fontsize=12)
            ax1.set_ylabel('MAE (Mean Absolute Error)', fontsize=12)
            ax1.set_title(f'Curvas de Aprendizaje - MAE Estándar', fontsize=13, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)
            
            # Gráfico 2: Weighted MAE
            ax2.plot(lc['train_sizes'], lc['train_wmae_mean'], 'o-', color='green', 
                    label='Entrenamiento', linewidth=2, markersize=6)
            ax2.fill_between(lc['train_sizes'], 
                           lc['train_wmae_mean'] - lc['train_wmae_std'],
                           lc['train_wmae_mean'] + lc['train_wmae_std'],
                           alpha=0.2, color='green')

            ax2.plot(lc['train_sizes'], lc['val_wmae_mean'], 'o-', color='orange', 
                    label='Validación', linewidth=2, markersize=6)
            ax2.fill_between(lc['train_sizes'],
                           lc['val_wmae_mean'] - lc['val_wmae_std'],
                           lc['val_wmae_mean'] + lc['val_wmae_std'],
                           alpha=0.2, color='orange')

            ax2.set_xlabel('Tamaño del Conjunto de Entrenamiento', fontsize=12)
            ax2.set_ylabel('Weighted MAE', fontsize=12)
            ax2.set_title(f'Curvas de Aprendizaje - Weighted MAE', fontsize=13, fontweight='bold')
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)
            
            # Ajustar layout y guardar
            plt.suptitle(f'Curvas de Aprendizaje - {self.title}', fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            save_path = os.path.join(self.model_ts_dir, f"learning_curves_comparison.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Curvas de aprendizaje comparativas guardadas en: {save_path}")
            
        else:
            # Gráfico simple solo con MAE estándar
            plt.figure(figsize=(10, 6))

            plt.plot(lc['train_sizes'], lc['train_mae_mean'], 'o-', color='blue', 
                    label='Entrenamiento', linewidth=2, markersize=6)
            plt.fill_between(lc['train_sizes'], 
                           lc['train_mae_mean'] - lc['train_mae_std'],
                           lc['train_mae_mean'] + lc['train_mae_std'],
                           alpha=0.2, color='blue')

            plt.plot(lc['train_sizes'], lc['val_mae_mean'], 'o-', color='red', 
                    label='Validación', linewidth=2, markersize=6)
            plt.fill_between(lc['train_sizes'],
                           lc['val_mae_mean'] - lc['val_mae_std'],
                           lc['val_mae_mean'] + lc['val_mae_std'],
                           alpha=0.2, color='red')

            plt.xlabel('Tamaño del Conjunto de Entrenamiento', fontsize=12)
            plt.ylabel('MAE (Mean Absolute Error)', fontsize=12)
            plt.title(f'Curvas de Aprendizaje - {self.title}', fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)

            save_path = os.path.join(self.model_ts_dir, f"learning_curve.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Curvas de aprendizaje guardadas en: {save_path}")

    def _analyze_multicollinearity(self, X: pd.DataFrame, y: pd.Series):
        try:
            X_processed = self.pipeline.named_steps['preprocessor'].transform(X)

            self.logger.info(f"DEBUG VIF: tipo de X_processed = {type(X_processed)}")

            # Obtener nombres de características
            feature_names = None
            if isinstance(X_processed, pd.DataFrame):
                # Si es DataFrame, usar sus columnas
                feature_names = X_processed.columns.tolist()
                X_processed_array = X_processed.values
                self.logger.info(f"DEBUG VIF: X_processed es DataFrame con {len(feature_names)} columnas")
                self.logger.info(f"DEBUG VIF: Primeros 10 nombres de columnas: {feature_names[:10]}")
                self.logger.info(f"DEBUG VIF: Últimos 10 nombres de columnas: {feature_names[-10:]}")

                # Contar cuántos son genéricos
                generic_count = sum(1 for name in feature_names if str(name).startswith('feature_') or str(name).replace('.', '').isdigit())
                self.logger.info(f"DEBUG VIF: Features con nombres genéricos: {generic_count}/{len(feature_names)}")

                # Intentar obtener nombres del preprocessor como respaldo
                try:
                    preprocessor = self.pipeline.named_steps['preprocessor']
                    alt_names = preprocessor.get_feature_names_out()
                    self.logger.info(f"DEBUG VIF: Nombres del preprocessor.get_feature_names_out(): {alt_names[:10]}")
                    self.logger.info(f"DEBUG VIF: Últimos nombres del preprocessor: {alt_names[-10:]}")
                except Exception as e:
                    self.logger.warning(f"DEBUG VIF: No se pudieron obtener nombres del preprocessor: {e}")
            else:
                # Si es array, intentar obtener nombres del preprocessor
                self.logger.info("DEBUG VIF: X_processed NO es DataFrame, intentando get_feature_names_out()")
                try:
                    # Obtener del preprocessor (excluyendo el modelo final)
                    # Use pipeline[:-1] si el preprocessor tiene pasos múltiples
                    preprocessor = self.pipeline.named_steps['preprocessor']

                    # Intentar primero con el transformador completo (sin el modelo)
                    if hasattr(preprocessor, 'get_feature_names_out'):
                        try:
                            feature_names = preprocessor.get_feature_names_out().tolist()
                            self.logger.info(f"DEBUG VIF: Obtenidos {len(feature_names)} nombres de preprocessor.get_feature_names_out()")
                        except:
                            # Si falla, intentar con transformers individuales
                            feature_names = preprocessor.get_feature_names_out(input_features=None)
                            feature_names = list(feature_names)
                            self.logger.info(f"DEBUG VIF: Obtenidos {len(feature_names)} nombres usando input_features=None")
                    else:
                        raise AttributeError("preprocessor no tiene get_feature_names_out()")

                    self.logger.info(f"DEBUG VIF: Primeros 10 nombres: {feature_names[:10]}")
                    self.logger.info(f"DEBUG VIF: Últimos 10 nombres: {feature_names[-10:]}")
                except Exception as e:
                    # Si falla, usar nombres genéricos
                    self.logger.warning(f"DEBUG VIF: get_feature_names_out() falló: {e}")
                    feature_names = [f'feature_{i}' for i in range(X_processed.shape[1])]

                X_processed_array = X_processed

            # Convertir a array denso si es necesario
            if hasattr(X_processed_array, 'toarray'):
                X_processed_array = X_processed_array.toarray()

            # Calcular VIF para cada característica
            vif_data = []
            for i in range(X_processed_array.shape[1]):
                try:
                    # Verificar si la columna tiene varianza cero
                    col_variance = np.var(X_processed_array[:, i])
                    if col_variance < 1e-10:  # Varianza prácticamente cero
                        vif_data.append({
                            'Feature': feature_names[i],
                            'VIF': np.nan,
                            'Reason': 'Varianza ≈ 0'
                        })
                    else:
                        vif_value = variance_inflation_factor(X_processed_array, i)
                        vif_data.append({
                            'Feature': feature_names[i],
                            'VIF': vif_value,
                            'Reason': None
                        })
                except Exception as e:
                    vif_data.append({
                        'Feature': feature_names[i],
                        'VIF': np.inf,
                        'Reason': f'Error: {str(e)[:50]}'
                    })

            self.vif_analysis = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)

            # Mostrar resultados
            self.logger.info("--- Análisis de Multicolinealidad (VIF) ---")
            self.logger.info(f"Total de features analizadas: {len(self.vif_analysis)}")

            # Mostrar TODAS las features en orden de índice (no de VIF) para debugging
            self.logger.info("\n📊 TODAS LAS FEATURES (en orden original):")
            vif_sorted_by_index = self.vif_analysis.copy()
            vif_sorted_by_index['index'] = range(len(vif_sorted_by_index))

            # Dividir en grupos de 20 para mejor legibilidad
            for i in range(0, len(feature_names), 20):
                group_end = min(i + 20, len(feature_names))
                self.logger.info(f"\n  Features {i}-{group_end-1}:")
                for j in range(i, group_end):
                    vif_value = self.vif_analysis[self.vif_analysis['Feature'] == feature_names[j]]['VIF'].values[0]
                    if np.isnan(vif_value):
                        vif_str = "N/A (const)"
                        status = "⚠️"
                    elif np.isinf(vif_value):
                        vif_str = "∞"
                        status = "⚠️"
                    else:
                        vif_str = f"{vif_value:.2f}"
                        status = "⚠️" if vif_value > 10 else "✓"
                    self.logger.info(f"    [{j:2d}] {status} {feature_names[j]:40s} VIF = {vif_str}")

            # Luego resaltar las problemáticas
            self.logger.info("\n" + "="*70)
            high_vif = self.vif_analysis[self.vif_analysis['VIF'] > 10.0]
            if len(high_vif) > 0:
                self.logger.info(f"⚠️  RESUMEN: {len(high_vif)}/{len(self.vif_analysis)} variables con VIF > 10.0")
                self.logger.info(f"\n📊 Top 10 variables con mayor colinealidad:")
                for _, row in high_vif.head(10).iterrows():
                    vif_val = row['VIF']
                    if np.isinf(vif_val):
                        self.logger.info(f"  {row['Feature']}: ∞ (colinealidad perfecta)")
                    elif np.isnan(vif_val):
                        self.logger.info(f"  {row['Feature']}: N/A (constante)")
                    else:
                        self.logger.info(f"  {row['Feature']}: {vif_val:.2f}")

                # Advertencia sobre impacto en validación cruzada
                extreme_vif = high_vif[high_vif['VIF'] > 100]
                if len(extreme_vif) > 0:
                    self.logger.warning(f"\n⚠️  {len(extreme_vif)} variables con VIF > 100 pueden causar fallas en algunos folds de CV")
                    self.logger.warning(f"   (matriz singular → LinearRegression no puede invertir X^T X)")
                    self.logger.warning(f"   Recomendación: Eliminar variables redundantes o usar regularización")
            else:
                self.logger.info("✓ No se detectó multicolinealidad significativa")

        except Exception as e:
            self.logger.error(f"Error calculando VIF: {e}")
            self.vif_analysis = None

    def _plot_feature_importance(self):
        """Crea y guarda el gráfico de importancia de características."""
        if self.feature_importance is None:
            return

        # Tomar top 15 características
        top_features = self.feature_importance.head(15)

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['Importance'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importancia')
        plt.title(f'Importancia de Características - {self.title}')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Guardar gráfico
        save_path = os.path.join(self.model_ts_dir, f"feature_importance.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Gráfico de importancia guardado en: {save_path}")

    def _extract_feature_importance(self):
        if self.pipeline is None:
            return

        # Obtener importancia del modelo
        regressor = self.pipeline.named_steps['regressor']
        importances = regressor.feature_importances_

        # 1) Mejor opción: si CatBoost conoce los nombres
        feature_names = None
        try:
            if hasattr(regressor, "feature_names_"):
                fn = list(regressor.feature_names_)
                if len(fn) == len(importances):
                    feature_names = fn
        except Exception:
            pass

        # 2) Segunda opción: del preprocessor
        if feature_names is None:
            try:
                pre = self.pipeline.named_steps['preprocessor']
                fn = pre.get_feature_names_out()
                # ColumnTransformer con verbose_feature_names_out=False suele devolver nombres originales
                if len(fn) == len(importances):
                    feature_names = list(fn)
            except Exception:
                feature_names = None

        # 3) Fallback seguro: orden ColumnTransformer = [num, cat]
        if feature_names is None:
            feature_names = list(self.num_cols) + list(self.cat_cols)

        # Crear DataFrame ordenado
        self.feature_importance = (
            pd.DataFrame({"Feature": feature_names, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )

        self._show_feature_importance()
        self._plot_feature_importance()

    def _show_feature_importance(self):
        """Muestra la importancia de características."""
        if self.feature_importance is None:
            self.logger.info("No hay información de importancia de características disponible.")
            return

        self.logger.info("--- Importancia de Características (Top 10) ---")
        for _, row in self.feature_importance.head(10).iterrows():
            self.logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")


    def _save_model(self):
        """Guarda el modelo entrenado con timestamp."""
        filename = f"model.pkl"
        filepath = os.path.join(self.model_ts_dir, filename)

        # Guardar modelo completo
        model_data = {
            'pipeline': self.pipeline,
            'cv_results': self.cv_results,
            'vif_analysis': self.vif_analysis,
            'learning_curve_results': self.learning_curve_results,
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'feature_importance': self.feature_importance,
        }

        joblib.dump(model_data, filepath)
        self.logger.info(f"Modelo guardado en: {filepath}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise ValueError("El modelo no ha sido entrenado. Llama a fit() primero.")

        return self.pipeline.predict(X)

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series, ts: str = None) -> Dict[str, float]:
        if ts is not None:
            # Cargar modelo de timestamp específico
            model_path = os.path.join(self.model_specific_dir, ts, "model.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

            model_data = joblib.load(model_path)
            pipeline = model_data['pipeline']
            self.logger.info(f"Modelo cargado desde: {model_path}")
        else:
            # Usar pipeline actual
            if self.pipeline is None:
                raise ValueError("No hay modelo disponible. Entrena o carga un modelo primero.")
            pipeline = self.pipeline

        # Realizar predicciones
        y_pred = pipeline.predict(X_test)

        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        result = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

        self.logger.info(f"Métricas de evaluación: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        return result

    def get_metrics(self) -> Dict[str, Any]:
        return self.cv_results

    def get_vif_analysis(self) -> Optional[pd.DataFrame]:
        return self.vif_analysis

    def get_best_params(self) -> Dict[str, Any]:
        return self.best_params

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        return self.feature_importance

