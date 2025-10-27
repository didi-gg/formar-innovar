"""
Script para evaluar todos los modelos entrenados en el conjunto de test.
Carga modelos guardados por timestamp y registra métricas en MLflow.
"""

import numpy as np
import pandas as pd
import os
import mlflow
import mlflow.sklearn
from pathlib import Path
import logging
from datetime import datetime
import traceback


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

    log_file = log_dir / f'evaluate_all_models-{timestamp}.log'

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
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
]

NUMERIC_FEATURES = [
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

def load_test_data():
    df = pd.read_csv('data/processed/test_moodle.csv')

    for feature in CATEGORICAL_FEATURES:
        df[feature] = df[feature].astype('category')

    for feature in NUMERIC_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')

    df[TARGET_FEATURE] = pd.to_numeric(df[TARGET_FEATURE], errors='coerce')
    return df


def evaluate_model(model_class, model_name, ts, X_test, y_test, logger):
    logger.info(f"\n=== Evaluando {model_name} (ts: {ts}) ===")

    try:
        # Crear instancia del pipeline
        logger.info(f"🔧 Cargando modelo {model_name} desde timestamp {ts}...")
        model = model_class(random_state=42)
        model.num_cols = NUMERIC_FEATURES
        model.cat_cols = CATEGORICAL_FEATURES

        # Evaluar modelo cargando desde el timestamp
        logger.info(f"📊 Evaluando {model_name} en conjunto de test...")
        metrics = model.evaluate_model(X_test, y_test, ts=ts)

        logger.info(f"✅ {model_name} evaluado exitosamente")
        logger.info(f"   RMSE: {metrics['rmse']:.4f}")
        logger.info(f"   MAE: {metrics['mae']:.4f}")
        logger.info(f"   R²: {metrics['r2']:.4f}")

        # Registrar en MLflow como nested run
        logger.info(f"📊 Registrando resultados en MLflow para {model_name}")
        with mlflow.start_run(run_name=f"{model_name}_Test", nested=True):
            run_id = mlflow.active_run().info.run_id
            logger.info(f"✅ Run iniciado: {run_id}")

            # Parámetros básicos
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("timestamp", ts)
            mlflow.log_param("n_test_samples", X_test.shape[0])
            mlflow.log_param("n_features", X_test.shape[1])

            # Métricas de test
            mlflow.log_metric("rmse_test", float(metrics['rmse']))
            mlflow.log_metric("mae_test", float(metrics['mae']))
            mlflow.log_metric("r2_test", float(metrics['r2']))

            logger.info(f"Run {run_id} completado para {model_name}")
        logger.info(f"{model_name} evaluación completada y registrada en MLflow")
        return metrics

    except Exception as e:
        logger.error(f"✗ Error evaluando {model_name}: {e}")
        logger.error(traceback.format_exc())
        raise e


def evaluate_h2o_model(test_df, ts, logger):
    """Evalúa el modelo H2O AutoML en el conjunto de test."""
    model_name = "H2O_AutoML"
    logger.info(f"\n=== Evaluando {model_name} (ts: {ts}) ===")

    try:
        logger.info("🔧 Inicializando H2O...")
        h2o.init()
        logger.info("✅ H2O inicializado")

        logger.info(f"🔧 Cargando modelo {model_name} desde timestamp {ts}...")
        model = h2oPipeline(random_state=42)

        # Evaluar modelo
        logger.info(f"📊 Evaluando {model_name} en conjunto de test...")
        metrics = model.evaluate_model(test_df, timestamp=ts)

        logger.info(f"✅ {model_name} evaluado exitosamente")
        logger.info(f"   RMSE: {metrics['rmse']:.4f}")
        logger.info(f"   MAE: {metrics['mae']:.4f}")
        logger.info(f"   R²: {metrics['r2']:.4f}")

        # Registrar en MLflow
        logger.info(f"📊 Registrando resultados en MLflow para {model_name}")
        with mlflow.start_run(run_name=f"{model_name}_Test", nested=True):
            run_id = mlflow.active_run().info.run_id
            logger.info(f"✅ Run iniciado: {run_id}")

            # Parámetros básicos
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("timestamp", ts)
            mlflow.log_param("n_test_samples", test_df.shape[0])
            mlflow.log_param("n_features", test_df.shape[1] - 1)  # -1 por el target

            # Métricas de test
            mlflow.log_metric("rmse_test", float(metrics['rmse']))
            mlflow.log_metric("mae_test", float(metrics['mae']))
            mlflow.log_metric("r2_test", float(metrics['r2']))

            logger.info(f"Run {run_id} completado para {model_name}")

        logger.info(f"{model_name} evaluación completada y registrada en MLflow")
        return metrics

    except Exception as e:
        logger.error(f"✗ Error evaluando {model_name}: {e}")
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
    """Ejecuta la evaluación de todos los modelos."""
    logger = setup_logger()
    logger.info("🚀 Evaluando todos los modelos en conjunto de test")

    # Configurar MLflow tracking URI
    mlflow_dir = project_root / 'mlruns'
    mlflow_dir_abs = str(mlflow_dir.absolute())

    os.makedirs(mlflow_dir_abs, exist_ok=True)
    logger.info(f"Directorio MLflow: {mlflow_dir_abs}")

    mlflow_tracking_uri = f"file://{mlflow_dir_abs}"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info(f"MLflow tracking URI: {mlflow_tracking_uri}")

    # Asegurar que no hay runs activos
    try:
        mlflow.end_run()
    except:
        pass

    # Configurar experimento único en MLflow
    try:
        experiment = mlflow.set_experiment("All_Models_Test_Evaluation")
        logger.info(f"Experimento MLflow configurado: {experiment.name}")
        logger.info(f"Experiment ID: {experiment.experiment_id}")
        logger.info(f"Artifact location: {experiment.artifact_location}")
    except Exception as e:
        logger.error(f"❌ Error configurando MLflow: {e}")
        logger.error(traceback.format_exc())
        raise e

    # Cargar datos de test
    logger.info("📊 Cargando dataset de test...")
    df_test = load_test_data()
    logger.info(f"Dataset de test cargado: {df_test.shape[0]} filas, {df_test.shape[1]} columnas")

    # Separar features y target
    X_test = df_test[ALL_FEATURES]
    y_test = df_test[TARGET_FEATURE]
    logger.info(f"Test set: {X_test.shape[0]} muestras, {X_test.shape[1]} características")

    # Definir modelos con sus timestamps
    models_config = [
        (LinearRegressionPipeline, "LinearRegression", "20251027_160652"),
        (ElasticNetPipeline, "ElasticNet", "20251027_160659"),
        (RandomForestPipeline, "RandomForest", "20251026_125812"),
        (CatBoostPipeline, "CatBoost", "20251027_121254")
    ]

    # Diccionario para almacenar resultados
    results = {}

    logger.info("\nIniciando run padre en MLflow...")
    with mlflow.start_run(run_name="All_Models_Test_Evaluation") as parent_run:
        logger.info(f"✅ Run padre iniciado: {parent_run.info.run_id}")

        # Log parámetros del experimento
        mlflow.log_params({
            "experiment_type": "test_evaluation",
            "n_models": len(models_config) + 1,  # +1 por H2O
            "test_dataset_size": len(X_test),
            "n_features": X_test.shape[1]
        })
        logger.info(f"Parámetros del experimento registrados")

        for model_class, model_name, ts in models_config:
            try:
                metrics = evaluate_model(model_class, model_name, ts, X_test, y_test, logger)
                results[model_name] = metrics
            except Exception as e:
                logger.error(f"Error evaluando {model_name}, continuando con el siguiente...")
                results[model_name] = None

        # Evaluar H2O AutoML
        logger.info("\nEvaluando H2O AutoML...")
        try:
            h2o_metrics = evaluate_h2o_model(df_test, "20251026_231034", logger)
            results["H2O_AutoML"] = h2o_metrics
        except Exception as e:
            logger.error(f"Error evaluando H2O AutoML, continuando...")
            results["H2O_AutoML"] = None

    # Resumen de resultados
    logger.info("\n" + "="*70)
    logger.info("📊 RESUMEN DE RESULTADOS EN TEST")
    logger.info("="*70)

    # Crear DataFrame con resultados
    results_list = []
    for model_name, metrics in results.items():
        if metrics:
            results_list.append({
                'Modelo': model_name,
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R²': metrics['r2']
            })

    if results_list:
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values('RMSE')

        logger.info("\n" + results_df.to_string(index=False))

        # Guardar resultados en CSV
        results_path = project_root / 'models' / 'test_evaluation_results.csv'
        results_df.to_csv(results_path, index=False)
        logger.info(f"\n✅ Resultados guardados en: {results_path}")

    logger.info("\n✅ Evaluación de todos los modelos completada")
    logger.info(f"Los resultados se guardaron en el experimento: All_Models_Test_Evaluation")
    logger.info(f"Para ver los resultados: cd {os.path.dirname(mlflow_dir_abs)} && mlflow ui")


if __name__ == "__main__":
    main()

