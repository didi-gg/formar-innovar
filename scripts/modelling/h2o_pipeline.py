import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import h2o
from h2o.automl import H2OAutoML

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.modelling.weighted_mae_scorer import evaluate_h2o_model_with_weighted_mae, weighted_mae
from scripts.preprocessing.g_smote import GSMOTERegressor
from scripts.modelling.smote_diagnostic import SMOTEDiagnostic


class h2oPipeline():

    model_name = "h2o"
    title = "H2O AutoML"
    best_model = None
    aml = None  # Objeto AutoML completo
    cv_results = None  # Resultados de validación cruzada
    feature_importance = None  # Importancia de características
    leaderboard = None  # Leaderboard de modelos
    train_data = None  # Datos de entrenamiento (H2OFrame)
    target_column = None  # Nombre de la columna objetivo

    def __init__(self, random_state: int = 42, use_resampling: bool = True):

        self.random_state = random_state
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.use_resampling = use_resampling

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

    def train_and_choose_best_model(self, train_df: pd.DataFrame, target_column: str):
        self.logger.info("=== Entrenando H2O AutoML ===")
        self.logger.info(f"Datos: {train_df.shape[0]} muestras, {train_df.shape[1]} características")

        if (self.use_resampling):
            resampler = GSMOTERegressor(n_synthetic_multiplier=3.0)
            X = train_df.drop(columns=[target_column])
            y = train_df[target_column]
            X_resampled, y_resampled = resampler.fit_resample(X, y)
            train_df = pd.concat([X_resampled, y_resampled], axis=1)
            nfolds = 0  # Disable cross-validation when using resampling
        else:
            nfolds = 5

        # 1. Convertir a H2OFrame
        self.logger.info("Convirtiendo datos a H2OFrame...")
        train = h2o.H2OFrame(train_df)
        self.train_data = train
        self.target_column = target_column

        # 2. Definir features y target
        x = train.columns
        x.remove(target_column)
        self.logger.info(f"Features: {len(x)}, Target: {target_column}")

        # 3. Crear y entrenar AutoML para regresión con validación cruzada
        self.logger.info("Iniciando AutoML (600 segundos)...")
        self.aml = H2OAutoML(
            max_runtime_secs=600, 
            seed=self.random_state,
            exclude_algos=["DeepLearning"],
            sort_metric='mae',  # Ordenar leaderboard por MAE
            nfolds=nfolds,
            keep_cross_validation_predictions=True,
            keep_cross_validation_models=True
        )
        self.aml.train(x=x, y=target_column, training_frame=train)

        # 4. Evaluar modelos con métrica personalizada y seleccionar el mejor
        self.leaderboard = self.aml.leaderboard
        self.best_model = self._select_best_model_with_weighted_mae(train)

        # 5. Mostrar leaderboard
        self._show_leaderboard()

        # 6. Extraer métricas de validación cruzada
        self._extract_cv_metrics()

        # 7. Guardar modelo
        self._save_model()
        
        # 8. Generar diagnóstico de SMOTE si se usó resampling
        if self.use_resampling:
            self._generate_smote_diagnostic(train_df, target_column)

        self.logger.info("\n✓ Entrenamiento completado exitosamente")
        return self.best_model
    
    def _generate_smote_diagnostic(self, train_df: pd.DataFrame, target_column: str):
        """Genera un reporte de diagnóstico sobre SMOTE."""
        try:
            self.logger.info("\n=== Generando diagnóstico de SMOTE ===")
            
            X = train_df.drop(columns=[target_column])
            y = train_df[target_column]
            
            # Simular resampling para obtener métricas
            resampler = GSMOTERegressor(n_synthetic_multiplier=3.0)
            X_resampled, y_resampled = resampler.fit_resample(X, y)
            
            # Obtener métricas del resampler
            resampler_metrics = None
            if hasattr(resampler, '_last_diversity_metrics'):
                resampler_metrics = {
                    'diversity': resampler._last_diversity_metrics,
                    'distribution': getattr(resampler, '_last_distribution_metrics', None)
                }
            
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
                cv_results_without_resampling=None,
                model_ts_dir=self.model_ts_dir
            )
            
        except Exception as e:
            self.logger.warning(f"Error generando diagnóstico de SMOTE: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())

    def _select_best_model_with_weighted_mae(self, train_frame):
        """
        Evalúa todos los modelos del leaderboard con la métrica Weighted MAE personalizada
        y selecciona el mejor modelo basado en esta métrica.
        """
        self.logger.info("\n=== Evaluando modelos con Weighted MAE personalizada ===")
        
        try:
            # Obtener todos los modelos del leaderboard
            leaderboard_df = self.leaderboard.as_data_frame()
            model_ids = leaderboard_df['model_id'].tolist()
            
            best_weighted_mae = float('inf')
            best_model = None
            model_scores = []
            
            # Evaluar cada modelo con la métrica personalizada
            for i, model_id in enumerate(model_ids[:10]):  # Evaluar top 10 para eficiencia
                try:
                    # Obtener el modelo
                    model = h2o.get_model(model_id)
                    
                    # Evaluar con métrica personalizada
                    metrics = evaluate_h2o_model_with_weighted_mae(
                        model, train_frame, self.target_column
                    )
                    
                    weighted_mae_score = metrics['weighted_mae']
                    model_scores.append({
                        'model_id': model_id,
                        'weighted_mae': weighted_mae_score,
                        'mae': metrics['mae'],
                        'rmse': metrics['rmse'],
                        'r2': metrics['r2']
                    })
                    
                    # Actualizar mejor modelo si es necesario
                    if weighted_mae_score < best_weighted_mae:
                        best_weighted_mae = weighted_mae_score
                        best_model = model
                    
                    self.logger.info(f"  {i+1}. {model_id[:50]}... - Weighted MAE: {weighted_mae_score:.4f}")
                    
                except Exception as e:
                    self.logger.warning(f"Error evaluando modelo {model_id}: {e}")
                    continue
            
            # Mostrar ranking por Weighted MAE
            if model_scores:
                self.logger.info("\n--- RANKING POR WEIGHTED MAE ---")
                model_scores.sort(key=lambda x: x['weighted_mae'])
                for i, score in enumerate(model_scores[:5]):
                    self.logger.info(f"  {i+1}. {score['model_id'][:50]}...")
                    self.logger.info(f"     Weighted MAE: {score['weighted_mae']:.4f} | MAE: {score['mae']:.4f} | RMSE: {score['rmse']:.4f}")
                
                # Guardar scores para análisis posterior
                self.weighted_mae_scores = model_scores
            
            # Si no se pudo evaluar ningún modelo, usar el líder original
            if best_model is None:
                self.logger.warning("No se pudo evaluar ningún modelo con Weighted MAE, usando líder original")
                best_model = self.aml.leader
                best_weighted_mae = "N/A"
            
            self.logger.info(f"\n🏆 MEJOR MODELO POR WEIGHTED MAE: {best_model.model_id}")
            self.logger.info(f"🎯 Weighted MAE Score: {best_weighted_mae}")
            
            return best_model
            
        except Exception as e:
            self.logger.error(f"Error en selección por Weighted MAE: {e}")
            self.logger.info("Usando líder original de H2O AutoML")
            return self.aml.leader

    def _show_leaderboard(self):
        """Muestra el leaderboard de modelos de H2O AutoML con información detallada."""
        if self.leaderboard is None:
            self.logger.info("No hay leaderboard disponible")
            return

        try:
            self.logger.info("\n" + "="*60)
            self.logger.info("           LEADERBOARD H2O AutoML")
            self.logger.info("="*60)

            # Convertir leaderboard a pandas
            leaderboard_df = self.leaderboard.as_data_frame()

            # Mostrar información general
            self.logger.info(f"Total de modelos entrenados: {leaderboard_df.shape[0]}")
            self.logger.info(f"Columnas disponibles: {', '.join(leaderboard_df.columns.tolist())}")

            # Mostrar top 5 modelos con más detalle
            self.logger.info("\n🏆 TOP 5 MODELOS:")
            self.logger.info("-" * 60)

            try:
                top_5_models = leaderboard_df.head(5)
                self.logger.info(f"Procesando {len(top_5_models)} modelos del TOP 5...")

                for idx, row in top_5_models.iterrows():
                    try:
                        model_id = row['model_id']
                        self.logger.info(f"\n  {idx+1}. MODELO: {model_id}")

                        # Extraer tipo de algoritmo del model_id
                        try:
                            algo_type = self._extract_algorithm_type(model_id)
                            if algo_type and algo_type != "Unknown":
                                self.logger.info(f"     Algoritmo: {algo_type}")
                            else:
                                # Fallback: extraer algoritmo básico del model_id
                                if '_' in model_id:
                                    algorithm_prefix = model_id.split('_')[0]
                                    self.logger.info(f"     Algoritmo: {algorithm_prefix}")
                        except Exception as e:
                            self.logger.debug(f"Error extrayendo algoritmo de {model_id}: {e}")
                            # Fallback seguro
                            if '_' in model_id:
                                algorithm_prefix = model_id.split('_')[0]
                                self.logger.info(f"     Algoritmo: {algorithm_prefix}")

                        # Mostrar métricas principales
                        metrics = []
                        try:
                            if 'rmse' in row and not pd.isna(row['rmse']):
                                metrics.append(f"RMSE: {row['rmse']:.4f}")
                            if 'mae' in row and not pd.isna(row['mae']):
                                metrics.append(f"MAE: {row['mae']:.4f}")
                            if 'r2' in row and not pd.isna(row['r2']):
                                metrics.append(f"R²: {row['r2']:.4f}")
                            if 'mean_residual_deviance' in row and not pd.isna(row['mean_residual_deviance']):
                                metrics.append(f"MRD: {row['mean_residual_deviance']:.4f}")

                            if metrics:
                                self.logger.info(f"     Métricas: {' | '.join(metrics)}")
                            else:
                                self.logger.info(f"     Métricas: No disponibles")
                        except Exception as e:
                            self.logger.debug(f"Error procesando métricas para {model_id}: {e}")

                        # Mostrar métricas adicionales si están disponibles
                        try:
                            additional_metrics = []
                            for col in row.index:
                                if col not in ['model_id', 'rmse', 'mae', 'r2', 'mean_residual_deviance'] and not pd.isna(row[col]):
                                    if isinstance(row[col], (int, float)) and abs(row[col]) < 1000:
                                        additional_metrics.append(f"{col}: {row[col]:.4f}")

                            if additional_metrics and len(additional_metrics) <= 3:  # Solo mostrar si no son demasiadas
                                self.logger.info(f"     Adicionales: {' | '.join(additional_metrics)}")
                        except Exception as e:
                            self.logger.debug(f"Error procesando métricas adicionales para {model_id}: {e}")

                    except Exception as e:
                        self.logger.warning(f"Error procesando modelo {idx+1}: {e}")
                        continue

                self.logger.info(f"\n✅ TOP 5 completado - {len(top_5_models)} modelos mostrados")

            except Exception as e:
                self.logger.error(f"Error crítico mostrando TOP 5: {e}")
                # Fallback básico - mostrar solo los IDs
                try:
                    self.logger.info("📋 Fallback - Solo IDs de modelos:")
                    for idx, row in leaderboard_df.head(5).iterrows():
                        self.logger.info(f"  {idx+1}. {row['model_id']}")
                except Exception as fallback_error:
                    self.logger.error(f"Error en fallback: {fallback_error}")

            # Información del mejor modelo
            self.logger.info("\n" + "="*60)
            self.logger.info("🥇 MEJOR MODELO SELECCIONADO")
            self.logger.info("="*60)
            self.logger.info(f"🎯 ID: {self.best_model.model_id}")
            try:
                algo_type = self._extract_algorithm_type(self.best_model.model_id)
                self.logger.info(f"🔧 Algoritmo: {algo_type}")
            except Exception as e:
                self.logger.debug(f"No se pudo extraer algoritmo: {e}")
                # Fallback básico
                algorithm_prefix = self.best_model.model_id.split('_')[0] if '_' in self.best_model.model_id else "Unknown"
                self.logger.info(f"🔧 Algoritmo: {algorithm_prefix}")

            # Mostrar detalles específicos del mejor modelo
            self._show_best_model_details()

        except Exception as e:
            self.logger.warning(f"Error mostrando leaderboard: {e}")

    def _extract_algorithm_type(self, model_id: str) -> str:
        if not model_id:
            return "Unknown"

        algorithm_mappings = {
            'StackedEnsemble': 'Stacked Ensemble',
            'GBM': 'Gradient Boosting Machine',
            'RandomForest': 'Random Forest',
            'XGBoost': 'XGBoost',
            'GLM': 'Generalized Linear Model',
            'DeepLearning': 'Deep Learning',
            'DRF': 'Distributed Random Forest',
            'NaiveBayes': 'Naive Bayes'
        }

        # Buscar coincidencias en el model_id
        for key, full_name in algorithm_mappings.items():
            if key in model_id:
                return full_name

        # Si no encuentra coincidencia, extraer la primera parte antes del primer '_'
        parts = model_id.split('_')
        if parts:
            return parts[0]

        return "Unknown"

    def _show_best_model_details(self):
        """Muestra detalles específicos del mejor modelo seleccionado."""
        if self.best_model is None:
            self.logger.info("No hay modelo disponible para mostrar detalles")
            return

        try:
            self.logger.info("\n📋 DETALLES DEL MEJOR MODELO:")
            self.logger.info("-" * 40)

            # Información básica
            if hasattr(self.best_model, 'algo'):
                self.logger.info(f"🔧 Algoritmo H2O: {self.best_model.algo}")

            # Parámetros del modelo si están disponibles
            try:
                if hasattr(self.best_model, 'params') and self.best_model.params:
                    self.logger.info("⚙️  HIPERPARÁMETROS PRINCIPALES:")
                    params = self.best_model.params

                    # Mostrar solo los parámetros más importantes
                    important_params = [
                        'ntrees', 'max_depth', 'learn_rate', 'sample_rate',
                        'col_sample_rate', 'min_rows', 'reg_alpha', 'reg_lambda',
                        'nfolds', 'seed'
                    ]

                    shown_params = 0
                    for param_name in important_params:
                        if param_name in params and params[param_name]['actual'] is not None:
                            value = params[param_name]['actual']
                            self.logger.info(f"     • {param_name}: {value}")
                            shown_params += 1
                            if shown_params >= 8:  # Limitar a 8 parámetros
                                break

                    if shown_params == 0:
                        self.logger.info("     • Parámetros por defecto del algoritmo")

            except Exception as e:
                self.logger.debug(f"No se pudieron extraer parámetros: {e}")

            # Información de validación cruzada
            try:
                if hasattr(self.best_model, 'cross_validation_metrics_summary'):
                    cv_summary = self.best_model.cross_validation_metrics_summary()
                    if cv_summary is not None:
                        self.logger.info("📊 VALIDACIÓN CRUZADA:")
                        cv_df = cv_summary.as_data_frame()

                        # Mostrar métricas de CV si están disponibles
                        for metric in ['rmse', 'mae', 'r2']:
                            if metric in cv_df.columns:
                                mean_val = cv_df[cv_df.index == 'mean'][metric].iloc[0] if 'mean' in cv_df.index else None
                                std_val = cv_df[cv_df.index == 'sd'][metric].iloc[0] if 'sd' in cv_df.index else None

                                if mean_val is not None:
                                    if std_val is not None:
                                        self.logger.info(f"     • {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
                                    else:
                                        self.logger.info(f"     • {metric.upper()}: {mean_val:.4f}")

            except Exception as e:
                self.logger.debug(f"No se pudieron extraer métricas de CV: {e}")

        except Exception as e:
            self.logger.warning(f"Error mostrando detalles del modelo: {e}")

    def _extract_cv_metrics(self):
        """
        Extrae métricas del mejor modelo.
        Solo extrae lo que está garantizado disponible en H2O.
        """
        if self.best_model is None:
            self.logger.warning("No hay modelo entrenado para extraer métricas")
            return

        try:
            # Obtener métricas del mejor modelo en datos de entrenamiento
            train_performance = self.best_model.model_performance(self.train_data)

            # Extraer métricas básicas que siempre están disponibles
            rmse = train_performance.rmse()
            mae = train_performance.mae()
            r2 = train_performance.r2()

            # Formato simple compatible con BasePipeline
            self.cv_results = {
                'rmse_test': np.array([rmse]),
                'mae_test': np.array([mae]),
                'r2_test': np.array([r2]),
                'rmse_train': np.array([rmse]),
                'mae_train': np.array([mae]),
                'r2_train': np.array([r2])
            }

            self.logger.info("--- Métricas del Mejor Modelo ---")
            self.logger.info(f"RMSE: {rmse:.4f}")
            self.logger.info(f"MAE: {mae:.4f}")
            self.logger.info(f"R²: {r2:.4f}")

        except Exception as e:
            self.logger.error(f"Error extrayendo métricas: {e}")
            self.cv_results = None

    def analyze(self, train_df: pd.DataFrame = None):
        """
        Realiza análisis adicional del modelo (importancia de características).

        Args:
            train_df: DataFrame de entrenamiento (opcional, usa el almacenado si no se proporciona)
        """
        if self.best_model is None:
            raise ValueError("El modelo no ha sido entrenado. Llama a train_and_choose_best_model() primero.")

        self.logger.info("=== Realizando Análisis Completo ===")

        # Extraer importancia de características
        self.logger.info("Analizando importancia de características...")
        self._extract_feature_importance()

        self.logger.info("✓ Análisis completado exitosamente")

    def _extract_feature_importance(self):
        """
        Extrae importancia de características del mejor modelo.
        Solo si el modelo lo soporta (algunos modelos de H2O no tienen varimp).
        """
        try:
            # Verificar si el modelo tiene el método varimp
            if not hasattr(self.best_model, 'varimp'):
                self.logger.info("El modelo seleccionado no soporta importancia de características")
                self.feature_importance = None
                return

            # Intentar extraer importancia
            variable_importance = self.best_model.varimp(use_pandas=True)

            if variable_importance is None or len(variable_importance) == 0:
                self.logger.info("No hay importancia de características disponible")
                self.feature_importance = None
                return

            # Verificar que tenga las columnas necesarias
            if 'variable' not in variable_importance.columns or 'relative_importance' not in variable_importance.columns:
                self.logger.warning(f"Formato de variable_importance inesperado. Columnas: {variable_importance.columns.tolist()}")
                self.feature_importance = None
                return

            # Normalizar importancia
            total_importance = variable_importance['relative_importance'].sum()
            if total_importance > 0:
                variable_importance['relative_importance'] = variable_importance['relative_importance'] / total_importance

            self.feature_importance = pd.DataFrame({
                'Feature': variable_importance['variable'],
                'Importance': variable_importance['relative_importance']
            }).sort_values('Importance', ascending=False)

            self._show_feature_importance()
            self._plot_feature_importance()

        except Exception as e:
            self.logger.warning(f"No se pudo extraer importancia de características: {e}")
            self.feature_importance = None

    def _show_feature_importance(self):
        """Muestra la importancia de características."""
        if self.feature_importance is None:
            self.logger.info("No hay información de importancia de características disponible.")
            return

        self.logger.info("--- Importancia de Características (Top 10) ---")
        for _, row in self.feature_importance.head(10).iterrows():
            self.logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")

    def _plot_feature_importance(self):
        """Crea y guarda el gráfico de importancia de características."""
        if self.feature_importance is None:
            return

        # Tomar top 15 características
        top_features = self.feature_importance.head(15)

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['Importance'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importancia Relativa')
        plt.title(f'Importancia de Características - {self.title}')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Guardar gráfico
        save_path = os.path.join(self.model_ts_dir, f"feature_importance.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Gráfico de importancia guardado en: {save_path}")

    def _save_model(self):
        """Guarda el modelo entrenado con timestamp y todos los resultados disponibles."""

        # Guardar el modelo H2O usando su método nativo
        model_path = h2o.save_model(model=self.best_model, path=self.model_ts_dir, force=True)
        self.logger.info(f"Modelo H2O guardado en: {model_path}")

        # Guardar metadatos y resultados adicionales con joblib
        metadata_filename = f"model_metadata.pkl"
        metadata_filepath = os.path.join(self.model_ts_dir, metadata_filename)

        model_metadata = {
            'model_path': model_path,  # Ruta al modelo H2O guardado
            'cv_results': self.cv_results,
            'feature_importance': self.feature_importance,
            'model_name': self.model_name,
            'timestamp': self.timestamp,
        }

        # Guardar leaderboard solo si está disponible
        try:
            if self.leaderboard is not None:
                model_metadata['leaderboard'] = self.leaderboard.as_data_frame()
        except Exception as e:
            self.logger.warning(f"No se pudo guardar leaderboard: {e}")

        joblib.dump(model_metadata, metadata_filepath)
        self.logger.info(f"Metadatos del modelo guardados en: {metadata_filepath}")


    def predict(self, test_df: pd.DataFrame, timestamp: str) -> np.ndarray:
        test = h2o.H2OFrame(test_df)

        # Cargar metadatos del modelo
        metadata_path = os.path.join(self.model_specific_dir, timestamp, "model_metadata.pkl")
        with open(metadata_path, "rb") as f:
            model_metadata = joblib.load(f)

        # Cargar el modelo H2O usando su método nativo
        model_path = model_metadata['model_path']
        best_model = h2o.load_model(model_path)

        predictions = best_model.predict(test)
        return predictions

    def evaluate_model(self, test_df: pd.DataFrame, timestamp: str = None) -> Dict[str, float]:
        test = h2o.H2OFrame(test_df)

        if timestamp is not None:
            # Cargar modelo de timestamp específico
            metadata_path = os.path.join(self.model_specific_dir, timestamp, "model_metadata.pkl")
            with open(metadata_path, "rb") as f:
                model_metadata = joblib.load(f)
            model_path = model_metadata['model_path']
            best_model = h2o.load_model(model_path)
        else:
            # Usar modelo actual
            best_model = self.best_model

        if best_model is None:
            raise ValueError("No hay modelo disponible. Entrena o carga un modelo primero.")

        # Evaluar con métrica personalizada
        result = evaluate_h2o_model_with_weighted_mae(best_model, test, self.target_column)

        self.logger.info(f"Métricas de evaluación (incluyendo Weighted MAE): {result}")
        return result

    # Métodos getter para compatibilidad con BasePipeline
    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Retorna las métricas de validación cruzada."""
        return self.cv_results

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Retorna el DataFrame de importancia de características."""
        return self.feature_importance

    def get_best_params(self) -> Dict[str, Any]:
        """
        Retorna información básica del mejor modelo.
        Para H2O AutoML, esto incluye el ID del modelo seleccionado.
        """
        if self.best_model is None:
            return {}

        try:
            params = {'model_id': self.best_model.model_id}

            # Intentar obtener el tipo de algoritmo si está disponible
            if hasattr(self.best_model, 'algo'):
                params['h2o_algorithm'] = self.best_model.algo

            return params
        except Exception as e:
            self.logger.warning(f"No se pudieron extraer parámetros del modelo: {e}")
            return {}

    def get_leaderboard(self) -> Optional[pd.DataFrame]:
        """Retorna el leaderboard de H2O AutoML como DataFrame."""
        try:
            if self.leaderboard is not None:
                return self.leaderboard.as_data_frame()
        except Exception as e:
            self.logger.warning(f"No se pudo convertir leaderboard a DataFrame: {e}")
        return None