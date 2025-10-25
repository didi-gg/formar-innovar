"""
Pipeline independiente para CatBoost con tuning de hiperparámetros y análisis completo.
"""

import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, cross_validate, learning_curve, GridSearchCV

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.preprocessing.encode_categorical_values import CategoricalEncoder


class CatBoostPipeline:
    """
    Pipeline independiente para CatBoost con análisis completo y tuning de hiperparámetros.
    
    Características:
    - Encoding categórico automático
    - Sin imputación (CatBoost maneja nulos automáticamente)
    - Sin escalado (CatBoost no lo necesita)
    - Tuning de hiperparámetros (iterations, depth, learning_rate)
    - Validación cruzada 5×8
    - Análisis de importancia de características
    - Curvas de aprendizaje
    - Guardado automático del modelo
    """
    
    def __init__(self, random_state: int = 42):
        """
        Inicializa el pipeline de CatBoost.
        
        Args:
            random_state: Semilla para reproducibilidad
        """
        self.random_state = random_state
        self.pipeline = None
        self.best_params = None
        self.cv_results = None
        self.feature_importance = None
        self.learning_curve_results = None
        self.model_name = "catboost"
        
        # Crear directorios necesarios
        self._create_directories()
    
    def _create_directories(self):
        """Crea los directorios necesarios para guardar modelos y resultados."""
        self.models_dir = Path("models")
        self.results_dir = Path("models/results")
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _create_pipeline(self) -> Pipeline:
        """
        Crea el pipeline completo de CatBoost.
        
        Returns:
            Pipeline con encoding categórico y modelo CatBoost
        """
        steps = []
        
        # 1. Encoding categórico (CatBoost maneja nulos y no necesita escalado)
        steps.append(('categorical_encoder', CategoricalEncoder()))
        
        # 2. Modelo CatBoost
        steps.append(('regressor', CatBoostRegressor(
            random_state=self.random_state,
            verbose=False,  # Silenciar output durante entrenamiento
            allow_writing_files=False  # No crear archivos temporales
        )))
        
        return Pipeline(steps)
    
    def _get_param_grid(self) -> Dict[str, list]:
        """
        Define la grilla de hiperparámetros para tuning.
        
        Returns:
            Diccionario con parámetros a optimizar
        """
        return {
            'regressor__iterations': [100, 300, 500],
            'regressor__depth': [4, 6, 8],
            'regressor__learning_rate': [0.01, 0.1, 0.2],
            'regressor__l2_leaf_reg': [1, 3, 5]
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'CatBoostPipeline':
        """
        Entrena el modelo CatBoost con tuning de hiperparámetros.
        
        Args:
            X: Características de entrenamiento
            y: Variable objetivo
            
        Returns:
            self
        """
        print(f"=== Entrenando CatBoost con Tuning ===")
        print(f"Datos: {X.shape[0]} muestras, {X.shape[1]} características")
        
        # Crear pipeline base
        base_pipeline = self._create_pipeline()
        
        # Configurar GridSearchCV
        param_grid = self._get_param_grid()
        cv_inner = RepeatedKFold(n_splits=3, n_repeats=2, random_state=self.random_state)
        
        print("Realizando tuning de hiperparámetros...")
        print(f"Parámetros a optimizar: {list(param_grid.keys())}")
        
        grid_search = GridSearchCV(
            base_pipeline,
            param_grid,
            cv=cv_inner,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Entrenar con tuning
        grid_search.fit(X, y)
        
        # Guardar mejor modelo y parámetros
        self.pipeline = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        print("--- Mejores Hiperparámetros ---")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        best_score = -grid_search.best_score_
        print(f"  Mejor RMSE (CV tuning): {best_score:.4f}")
        
        # Validación cruzada final con mejores parámetros
        print("Realizando validación cruzada final 5×8...")
        self._cross_validate_with_best_params(X, y)

        # Extraer importancia de características
        self._extract_feature_importance()

        # Guardar modelo
        self._save_model()

        print("✓ Entrenamiento completado exitosamente")
        return self

    def analyze(self, X: pd.DataFrame, y: pd.Series):
        """
        Realiza análisis completo del modelo entrenado.
        
        Args:
            X: Características de entrenamiento
            y: Variable objetivo
        """
        if self.pipeline is None:
            raise ValueError("El modelo no ha sido entrenado. Llama a fit() primero.")
        
        print("=== Realizando Análisis Completo ===")
        
        # Mostrar importancia de características
        print("Analizando importancia de características...")
        self._show_feature_importance()
        
        # Curvas de aprendizaje
        print("Generando curvas de aprendizaje...")
        self._analyze_learning_curves(X, y)
        
        
        print("✓ Análisis completado exitosamente")
    
    def _cross_validate_with_best_params(self, X: pd.DataFrame, y: pd.Series):
        """
        Realiza validación cruzada 5×8 con los mejores parámetros encontrados.
        
        Args:
            X: Características de entrenamiento
            y: Variable objetivo
        """
        # Usar el pipeline con mejores parámetros (self.pipeline ya los tiene)
        # pero crear uno nuevo para evitar usar el modelo ya entrenado
        pipeline_for_cv = self._create_pipeline()
        
        # Aplicar los mejores parámetros encontrados
        for param, value in self.best_params.items():
            param_parts = param.split('__')
            if len(param_parts) == 2:
                step_name, param_name = param_parts
                pipeline_for_cv.named_steps[step_name].set_params(**{param_name: value})
        
        # Configurar validación cruzada 5×8
        cv = RepeatedKFold(n_splits=5, n_repeats=8, random_state=self.random_state)
        
        # Métricas a evaluar
        scoring = {
            'rmse': 'neg_mean_squared_error',
            'mae': 'neg_mean_absolute_error',
            'r2': 'r2'
        }
        
        # Realizar validación cruzada con pipeline limpio pero con mejores parámetros
        cv_results = cross_validate(
            pipeline_for_cv, X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=True
        )
        
        # Procesar resultados
        self.cv_results = {
            'rmse_test': np.sqrt(-cv_results['test_rmse']),
            'mae_test': -cv_results['test_mae'],
            'r2_test': cv_results['test_r2'],
            'rmse_train': np.sqrt(-cv_results['train_rmse']),
            'mae_train': -cv_results['train_mae'],
            'r2_train': cv_results['train_r2']
        }
        
        # Mostrar resultados
        print("--- Resultados Validación Cruzada (5×8) con Mejores Parámetros ---")
        for metric in ['rmse', 'mae', 'r2']:
            test_scores = self.cv_results[f'{metric}_test']
            train_scores = self.cv_results[f'{metric}_train']
            print(f"{metric.upper()}:")
            print(f"  Test:  {test_scores.mean():.4f} ± {test_scores.std():.4f}")
            print(f"  Train: {train_scores.mean():.4f} ± {train_scores.std():.4f}")
    
    def _extract_feature_importance(self):
        """
        Extrae la importancia de características del modelo CatBoost.
        """
        if self.pipeline is None:
            return
        
        # Obtener importancia del modelo CatBoost
        regressor = self.pipeline.named_steps['regressor']
        importances = regressor.feature_importances_
        
        # Crear DataFrame con importancia
        self.feature_importance = pd.DataFrame({
            'Feature': [f'feature_{i}' for i in range(len(importances))],
            'Importance': importances
        }).sort_values('Importance', ascending=False)
    
    def _show_feature_importance(self):
        """Muestra la importancia de características."""
        if self.feature_importance is None:
            print("No hay información de importancia de características disponible.")
            return
        
        print("--- Importancia de Características (Top 10) ---")
        for _, row in self.feature_importance.head(10).iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    def _analyze_learning_curves(self, X: pd.DataFrame, y: pd.Series):
        """
        Analiza y guarda las curvas de aprendizaje.
        
        Args:
            X: Características de entrenamiento
            y: Variable objetivo
        """
        try:
            # Calcular curvas de aprendizaje
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_sizes_abs, train_scores, val_scores = learning_curve(
                self.pipeline, X, y,
                train_sizes=train_sizes,
                cv=5,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,
                random_state=self.random_state
            )
            
            # Convertir a RMSE positivo
            train_rmse = -train_scores
            val_rmse = -val_scores
            
            # Calcular estadísticas
            train_rmse_mean = np.mean(train_rmse, axis=1)
            train_rmse_std = np.std(train_rmse, axis=1)
            val_rmse_mean = np.mean(val_rmse, axis=1)
            val_rmse_std = np.std(val_rmse, axis=1)
            
            # Guardar resultados
            self.learning_curve_results = {
                'train_sizes': train_sizes_abs,
                'train_rmse_mean': train_rmse_mean,
                'train_rmse_std': train_rmse_std,
                'val_rmse_mean': val_rmse_mean,
                'val_rmse_std': val_rmse_std
            }
            
            # Crear y guardar gráfico
            self._plot_learning_curves()
            
        except Exception as e:
            print(f"Error analizando curvas de aprendizaje: {e}")
            self.learning_curve_results = None
    
    def _plot_learning_curves(self):
        """Crea y guarda el gráfico de curvas de aprendizaje."""
        if self.learning_curve_results is None:
            return
        
        lc = self.learning_curve_results
        
        plt.figure(figsize=(10, 6))
        
        # Curvas de entrenamiento y validación
        plt.plot(lc['train_sizes'], lc['train_rmse_mean'], 'o-', color='blue', 
                    label='Entrenamiento', linewidth=2, markersize=6)
        plt.fill_between(lc['train_sizes'], 
                           lc['train_rmse_mean'] - lc['train_rmse_std'],
                           lc['train_rmse_mean'] + lc['train_rmse_std'],
                           alpha=0.2, color='blue')
            
        plt.plot(lc['train_sizes'], lc['val_rmse_mean'], 'o-', color='red', 
                label='Validación', linewidth=2, markersize=6)
        plt.fill_between(lc['train_sizes'],
                       lc['val_rmse_mean'] - lc['val_rmse_std'],
                       lc['val_rmse_mean'] + lc['val_rmse_std'],
                       alpha=0.2, color='red')
        
        # Configurar gráfico
        plt.xlabel('Tamaño del Conjunto de Entrenamiento')
        plt.ylabel('RMSE')
        plt.title('Curvas de Aprendizaje - CatBoost')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Guardar gráfico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_model_dir = self.results_dir / self.model_name
        os.makedirs(results_model_dir, exist_ok=True)
        
        save_path = results_model_dir / f"learning_curves_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Curvas de aprendizaje guardadas en: {save_path}")
    
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
        plt.title('Importancia de Características - CatBoost')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Guardar gráfico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_model_dir = self.results_dir / self.model_name
        os.makedirs(results_model_dir, exist_ok=True)
        
        save_path = results_model_dir / f"feature_importance_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico de importancia guardado en: {save_path}")
    
    
    def _save_model(self):
        """Guarda el modelo entrenado con timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{self.model_name}.pkl"
        filepath = self.models_dir / filename
        
        # Guardar modelo completo
        model_data = {
            'pipeline': self.pipeline,
            'best_params': self.best_params,
            'cv_results': self.cv_results,
            'feature_importance': self.feature_importance,
            'learning_curve_results': self.learning_curve_results,
            'model_name': self.model_name,
            'timestamp': timestamp
        }
        
        joblib.dump(model_data, filepath)
        print(f"Modelo guardado en: {filepath}")
        
        # También guardar solo el pipeline para uso directo
        pipeline_path = self.models_dir / f"{timestamp}_{self.model_name}_pipeline.pkl"
        joblib.dump(self.pipeline, pipeline_path)
        print(f"Pipeline guardado en: {pipeline_path}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones sobre nuevos datos.
        
        Args:
            X: Características para predecir
        
        Returns:
            Predicciones
        """
        if self.pipeline is None:
            raise ValueError("El modelo no ha sido entrenado. Llama a fit() primero.")
        
        return self.pipeline.predict(X)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtiene las métricas de validación cruzada.
        
        Returns:
            Diccionario con métricas
        """
        return self.cv_results
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        Obtiene los mejores hiperparámetros encontrados.
        
        Returns:
            Diccionario con mejores parámetros
        """
        return self.best_params
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Obtiene la importancia de las características.
        
        Returns:
            DataFrame con importancia de características
        """
        return self.feature_importance


# Ejemplo de uso
if __name__ == "__main__":
    # Crear datos de ejemplo con algunos valores nulos
    from sklearn.datasets import make_regression
    
    X, y = make_regression(n_samples=1000, n_features=15, noise=0.1, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y, name='target')
    
    # Agregar algunos valores nulos para probar que CatBoost los maneja
    np.random.seed(42)
    X_df.loc[:20, 'feature_0'] = np.nan
    X_df.loc[30:50, 'feature_2'] = np.nan
    
    # Entrenar modelo
    model = CatBoostPipeline()
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
