import numpy as np
import pandas as pd
import os
from pathlib import Path
import logging
from datetime import datetime

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.modelling.linear_regression_pipeline import LinearRegressionPipeline
from scripts.modelling.elasticnet_pipeline import ElasticNetPipeline
from scripts.modelling.random_forest_pipeline import RandomForestPipeline
from scripts.modelling.catboost_pipeline import CatBoostPipeline
from scripts.modelling.h2o_pipeline import h2oPipeline
import h2o


def setup_logger():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    log_dir = project_root / 'models'
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f'run_all_experiments-{timestamp}.log'

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Remover handlers existentes para evitar duplicados
    logger.handlers.clear()

    # Handler para archivo
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formato del log
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Agregar handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logger configurado. Archivo de log: {log_file}")

    return logger


CATEGORICAL_FEATURES = [
    'actividades_extracurriculares',
    'apoyo_familiar',
    'demuestra_confianza',
    'dia_preferido',
    'estrato',
    'familia',
    'género',
    'interés_estudios_superiores',
    'medio_transporte',
    'nivel_motivación',
    'participación_clase',
    'proyección_vocacional',
    'rol_adicional',
    'time_engagement_level',
    'tipo_vivienda',
    'zona_vivienda',
    'jornada_preferida',
]

NUMERIC_FEATURES = [
    'horas_semana_estudio_casa',
    'count_login_mon',
    'count_login_fri',
    'intensidad',
    'total_hermanos',
    'student_total_views',
    'substring_similarity',
    'max_inactividad',
    'avg_days_from_planned_start',
    'common_bigrams',
    'count_collaboration',
    'age',
    'total_hours',
    'teacher_experiencia_nivel_ficc',
    'num_students_interacted',
    'update_events_count',
    'avg_days_since_creation',
    'total_course_time_hours',
    'num_students_viewed',
    'percent_students_viewed',
    'max_days_since_creation',
    'edad_estudiante',
    'avg_days_since_last_update',
    'sequence_match_ratio',
    'num_modules',
    'total_views',
    'teacher_total_updates',
    'teacher_experiencia_nivel',
    'count_in_english'
]

TARGET_FEATURE = 'nota_final'

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

def load_data():
    df = pd.read_csv('data/processed/train_moodle.csv')

    for feature in CATEGORICAL_FEATURES:
        df[feature] = df[feature].astype('category')

    for feature in NUMERIC_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')

    df[TARGET_FEATURE] = pd.to_numeric(df[TARGET_FEATURE], errors='coerce')
    return df

def run_model(model_class, model_name, X, y, logger, use_resampling=True):
    logger.info(f"\n=== Ejecutando {model_name} ===")

    try:
        # Entrenar modelo
        logger.info(f"🔧 Entrenando {model_name}...")
        model = model_class(random_state=42, use_resampling=use_resampling)
        model.num_cols = NUMERIC_FEATURES
        model.cat_cols = CATEGORICAL_FEATURES
        model.fit(X, y)
        model.analyze(X, y)
        logger.info(f"✅ {model_name} entrenado exitosamente")
        
        logger.info(f"model_type {model_name}")
        logger.info(f"n_samples {X.shape[0]}")
        logger.info(f"n_features {X.shape[1]}")
        logger.info(f"random_state {42}")

        # Mejores parámetros (si tiene tuning)
        if hasattr(model, 'get_best_params') and model.get_best_params():
            best_params = model.get_best_params()
            logger.info(f"Registrando {len(best_params)} hiperparámetros")
            for param, value in best_params.items():
                clean_param = param.replace('regressor__', '')
                logger.info(f"param_{clean_param}: {value}")

        # Métricas de validación cruzada
        if hasattr(model, 'get_metrics') and model.get_metrics():
            cv_metrics = model.get_metrics()
            logger.info(f"Registrando métricas de validación cruzada")
            logger.info(f"rmse_test_mean {float(cv_metrics['rmse_test'].mean())}")
            logger.info(f"rmse_test_std {float(cv_metrics['rmse_test'].std())}")
            logger.info(f"mae_test_mean {float(cv_metrics['mae_test'].mean())}")
            logger.info(f"mae_test_std {float(cv_metrics['mae_test'].std())}")
            logger.info(f"r2_test_mean {float(cv_metrics['r2_test'].mean())}")
            logger.info(f"r2_test_std {float(cv_metrics['r2_test'].std())}")
            
            # Agregar Weighted MAE si está disponible
            if 'weighted_mae_test' in cv_metrics:
                logger.info(f"weighted_mae_test_mean {float(cv_metrics['weighted_mae_test'].mean())}")
                logger.info(f"weighted_mae_test_std {float(cv_metrics['weighted_mae_test'].std())}")

        # Análisis específicos por modelo
        if hasattr(model, 'get_vif_analysis') and model.get_vif_analysis() is not None:
            vif_df = model.get_vif_analysis()
            high_vif_count = len(vif_df[vif_df['VIF'] > 10.0])
            logger.info(f"VIF: {high_vif_count} características con VIF > 10")

        if hasattr(model, 'get_feature_importance') and model.get_feature_importance() is not None:
            importance_df = model.get_feature_importance()
            if 'Importance' in importance_df.columns:
                logger.info(f"Importancia: max={importance_df['Importance'].max():.4f}")
                logger.info(f"max_feature_importance {float(importance_df['Importance'].max())}")
                logger.info(f"top_5_importance_sum {float(importance_df.head(5)['Importance'].sum())}")

    except Exception as e:
        logger.error(f"✗ Error en {model_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise e

def run_h2o_model(train_df: pd.DataFrame, logger, use_resampling=False):
    model_name = "H2O_AutoML"
    logger.info(f"\n=== Ejecutando {model_name} ===")

    try:
        logger.info("🔧 Inicializando H2O...")
        h2o.init()
        logger.info("✅ H2O inicializado")

        # Crear y entrenar modelo
        logger.info(f"🔧 Entrenando {model_name}...")
        model = h2oPipeline(random_state=42, use_resampling=use_resampling)
        # Only select ALL_FEATURES and TARGET_FEATURE
        train_df = train_df[ALL_FEATURES + [TARGET_FEATURE]].copy()
        best_model = model.train_and_choose_best_model(train_df, TARGET_FEATURE)
        logger.info(f"✅ {model_name} entrenado exitosamente")

        # Realizar análisis (importancia de características)
        logger.info("Analizando modelo...")
        model.analyze()

        # Parámetros básicos
        logger.info(f"model_type {model_name}")
        logger.info(f"n_samples {train_df.shape[0]}")
        logger.info(f"n_features {train_df.shape[1] - 1}")
        logger.info(f"random_state 42")
        logger.info(f"max_runtime_secs 300")
        logger.info(f"nfolds 5")

        # Parámetros del mejor modelo
        if hasattr(model, 'get_best_params'):
            try:
                best_params = model.get_best_params()
                if best_params:
                    logger.info(f"Registrando {len(best_params)} parámetros del mejor modelo")
                    for param, value in best_params.items():
                        logger.info(f"{param}: {value}")
            except Exception as e:
                logger.warning(f"No se pudieron registrar parámetros del modelo: {e}")

        # Métricas
            try:
                cv_metrics = model.get_metrics()
                if cv_metrics and 'rmse_test' in cv_metrics:
                    logger.info(f"Registrando métricas del modelo")
                    logger.info(f"rmse_test_mean {float(cv_metrics['rmse_test'].mean())}")
                    logger.info(f"mae_test_mean {float(cv_metrics['mae_test'].mean())}")
                    logger.info(f"r2_test_mean {float(cv_metrics['r2_test'].mean())}")

                    # Calcular desviación estándar (será 0 si solo hay un valor)
                    logger.info(f"rmse_test_std {float(cv_metrics['rmse_test'].std())}")
                    logger.info(f"mae_test_std {float(cv_metrics['mae_test'].std())}")
                    logger.info(f"r2_test_std {float(cv_metrics['r2_test'].std())}")
            except Exception as e:
                logger.warning(f"No se pudieron registrar métricas: {e}")

            # Importancia de características
            try:
                importance_df = model.get_feature_importance()
                if importance_df is not None and len(importance_df) > 0:
                    logger.info(f"max_feature_importance {float(importance_df['Importance'].max())}")
                    logger.info(f"top_5_importance_sum {float(importance_df.head(5)['Importance'].sum())}")
                    logger.info(f"Importancia: max={importance_df['Importance'].max():.4f}")
            except Exception as e:
                logger.warning(f"No se pudo registrar importancia de características: {e}")


        logger.info(f"{model_name} completado y registrado en MLflow")
    except Exception as e:
        logger.error(f"✗ Error en {model_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise e
    finally:
        # Apagar H2O
        try:
            h2o.cluster().shutdown()
            logger.info("✅ H2O apagado correctamente")
        except:
            pass

def main():
    """Ejecuta todos los modelos."""
    # Configurar logger
    logger = setup_logger()
    logger.info("🚀 Ejecutando todos los modelos")

    # Cargar datos
    logger.info("Cargando dataset...")
    df = load_data()
    logger.info(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    use_resampling = True

    # Separar features y target
    X = df[ALL_FEATURES]
    y = df[TARGET_FEATURE]
    logger.info(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} características")

    # Lista de modelos
    models = [
        #(LinearRegressionPipeline, "LinearRegression"),
        #(ElasticNetPipeline, "ElasticNet"),
        #(RandomForestPipeline, "RandomForest"),
        (CatBoostPipeline, "CatBoost")
    ]

    # Ejecutar cada modelo como nested run
    for model_class, model_name in models:
        run_model(model_class, model_name, X, y, logger, use_resampling=use_resampling)

    # Ejecutar H2O AutoML
    logger.info("\nEjecutando H2O AutoML...")
    run_h2o_model(df, logger, use_resampling=use_resampling)

    logger.info("\n✅ Todos los modelos completados (incluyendo H2O AutoML)")

if __name__ == "__main__":
    main()
