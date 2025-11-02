import numpy as np
from sklearn.metrics import make_scorer
import h2o


def weighted_mae(y_true, y_pred, weight_low=2.0, weight_high=1.0, threshold=60):
    """
    Calcula el MAE ponderado donde se asigna mayor peso a errores en el rango 0-60.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        weight_low: Peso para valores en rango [0, threshold] (default: 2.0)
        weight_high: Peso para valores > threshold (default: 1.0)
        threshold: Umbral que separa los rangos (default: 60)
    
    Returns:
        float: MAE ponderado
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calcular errores absolutos
    absolute_errors = np.abs(y_true - y_pred)
    
    # Asignar pesos basados en los valores reales
    weights = np.where(y_true <= threshold, weight_low, weight_high)
    
    # Calcular MAE ponderado
    weighted_errors = weights * absolute_errors
    weighted_mae_score = np.mean(weighted_errors)
    
    return weighted_mae_score


def weighted_mae_scorer(weight_low=2.0, weight_high=1.0, threshold=60):
    """
    Crea un scorer de sklearn para el MAE ponderado.
    
    Args:
        weight_low: Peso para valores en rango [0, threshold]
        weight_high: Peso para valores > threshold
        threshold: Umbral que separa los rangos
    
    Returns:
        sklearn scorer: Scorer que puede usarse en GridSearchCV
    """
    def scorer_func(y_true, y_pred):
        return weighted_mae(y_true, y_pred, weight_low, weight_high, threshold)
    
    # make_scorer con greater_is_better=False porque queremos minimizar el error
    return make_scorer(scorer_func, greater_is_better=False)


# Crear scorer predeterminado
default_weighted_mae_scorer = weighted_mae_scorer(weight_low=2.0, weight_high=1.0, threshold=60)


def h2o_weighted_mae_metric(y_true_h2o, y_pred_h2o, weight_low=2.0, weight_high=1.0, threshold=60):
    """
    Función de métrica personalizada para H2O AutoML.
    
    Args:
        y_true_h2o: H2OFrame con valores reales
        y_pred_h2o: H2OFrame con valores predichos
        weight_low: Peso para valores en rango [0, threshold]
        weight_high: Peso para valores > threshold
        threshold: Umbral que separa los rangos
    
    Returns:
        float: MAE ponderado
    """
    try:
        # Convertir H2OFrames a numpy arrays
        y_true = y_true_h2o.as_data_frame().values.flatten()
        y_pred = y_pred_h2o.as_data_frame().values.flatten()
        
        # Usar la función weighted_mae existente
        return weighted_mae(y_true, y_pred, weight_low, weight_high, threshold)
    except Exception as e:
        print(f"Error en h2o_weighted_mae_metric: {e}")
        # Fallback a MAE estándar
        return float(h2o.mean(h2o.abs(y_true_h2o - y_pred_h2o)))


def evaluate_h2o_model_with_weighted_mae(model, test_frame, target_column, weight_low=2.0, weight_high=1.0, threshold=60):
    """
    Evalúa un modelo H2O usando la métrica Weighted MAE personalizada.
    
    Args:
        model: Modelo H2O entrenado
        test_frame: H2OFrame con datos de prueba
        target_column: Nombre de la columna objetivo
        weight_low: Peso para valores en rango [0, threshold]
        weight_high: Peso para valores > threshold
        threshold: Umbral que separa los rangos
    
    Returns:
        dict: Diccionario con métricas incluyendo weighted_mae
    """
    try:
        # Obtener predicciones
        predictions = model.predict(test_frame)
        y_true = test_frame[target_column]
        y_pred = predictions
        
        # Calcular weighted MAE
        weighted_mae_score = h2o_weighted_mae_metric(y_true, y_pred, weight_low, weight_high, threshold)
        
        # Obtener métricas estándar
        performance = model.model_performance(test_frame)
        
        return {
            'weighted_mae': weighted_mae_score,
            'mae': performance.mae(),
            'rmse': performance.rmse(),
            'r2': performance.r2()
        }
    except Exception as e:
        print(f"Error evaluando modelo H2O con weighted MAE: {e}")
        # Fallback a métricas estándar
        performance = model.model_performance(test_frame)
        return {
            'weighted_mae': performance.mae(),  # Fallback a MAE estándar
            'mae': performance.mae(),
            'rmse': performance.rmse(),
            'r2': performance.r2()
        }
