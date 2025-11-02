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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.modelling.linear_regression_pipeline import LinearRegressionPipeline
from scripts.modelling.elasticnet_pipeline import ElasticNetPipeline
from scripts.modelling.random_forest_pipeline import RandomForestPipeline
from scripts.modelling.catboost_pipeline import CatBoostPipeline
from scripts.modelling.h2o_pipeline import h2oPipeline
from scripts.modelling.weighted_mae_scorer import weighted_mae
import h2o
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def diagnose_model_predictions(model_name, y_train, y_test, y_pred, model_dir: Path, logger):
    """
    Realiza diagnóstico completo de las predicciones de un modelo.
    Basado en diagnose_model_metrics.py pero adaptado para cualquier modelo.
    Guarda los resultados en el directorio timestamp del modelo.
    
    Args:
        model_name: Nombre del modelo
        y_train: Valores reales del conjunto de entrenamiento
        y_test: Valores reales del conjunto de test
        y_pred: Predicciones del modelo
        model_dir: Directorio timestamp del modelo (ej: models/elasticnet/20251030_111313)
        logger: Logger
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"🔍 DIAGNÓSTICO DETALLADO - {model_name}")
    logger.info(f"{'='*80}")
    
    # Crear directorio de diagnósticos dentro del directorio del modelo
    model_diag_dir = model_dir / 'diagnostics'
    model_diag_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Análisis de distribución
    _analyze_distribution_detailed(y_train, y_test, model_diag_dir, logger)
    
    # 2. Análisis de predicciones detallado
    rmse, mae, r2 = _analyze_predictions_detailed(y_test, y_pred, model_diag_dir, logger)
    
    # 3. Análisis detallado de residuos
    _analyze_residuals_detailed(y_test, y_pred, model_diag_dir, logger)
    
    # 4. Comparación con baseline
    _compare_with_baseline(y_train, y_test, mae, r2, logger)
    
    # 5. Análisis de errores por rango
    _analyze_errors_by_range_detailed(y_test, y_pred, model_diag_dir, logger)
    
    logger.info(f"\n✅ Diagnóstico de {model_name} guardado en: {model_diag_dir}")
    
    return model_diag_dir


def _analyze_distribution_detailed(y_train, y_test, output_dir: Path, logger):
    """Analiza la distribución de calificaciones."""
    logger.info("\n" + "="*70)
    logger.info("=== ANÁLISIS DE DISTRIBUCIÓN DE CALIFICACIONES ===")
    logger.info("="*70)
    
    # Estadísticas de train
    logger.info("\n📊 CONJUNTO DE ENTRENAMIENTO:")
    logger.info(f"  Media:          {y_train.mean():.2f}")
    logger.info(f"  Std Dev:        {y_train.std():.2f}")
    logger.info(f"  Mediana:        {y_train.median():.2f}")
    logger.info(f"  Min:            {y_train.min():.0f}")
    logger.info(f"  Max:            {y_train.max():.0f}")
    logger.info(f"  Q1:             {y_train.quantile(0.25):.2f}")
    logger.info(f"  Q3:             {y_train.quantile(0.75):.2f}")
    logger.info(f"  Valores únicos: {y_train.nunique()}")
    logger.info(f"  Total muestras: {len(y_train)}")
    
    # Estadísticas de test
    logger.info("\n📊 CONJUNTO DE TEST:")
    logger.info(f"  Media:          {np.mean(y_test):.2f}")
    logger.info(f"  Std Dev:        {np.std(y_test):.2f}")
    logger.info(f"  Mediana:        {np.median(y_test):.2f}")
    logger.info(f"  Min:            {np.min(y_test):.0f}")
    logger.info(f"  Max:            {np.max(y_test):.0f}")
    logger.info(f"  Q1:             {np.percentile(y_test, 25):.2f}")
    logger.info(f"  Q3:             {np.percentile(y_test, 75):.2f}")
    logger.info(f"  Valores únicos: {len(np.unique(y_test))}")
    logger.info(f"  Total muestras: {len(y_test)}")
    
    # Crear visualización
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograma de train
    axes[0].hist(y_train, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(y_train.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {y_train.mean():.2f}')
    axes[0].axvline(y_train.median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {y_train.median():.2f}')
    axes[0].set_xlabel('Nota Final', fontsize=12)
    axes[0].set_ylabel('Frecuencia', fontsize=12)
    axes[0].set_title('Distribución - Conjunto de Entrenamiento', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Histograma de test
    axes[1].hist(y_test, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1].axvline(np.mean(y_test), color='red', linestyle='--', linewidth=2, label=f'Media: {np.mean(y_test):.2f}')
    axes[1].axvline(np.median(y_test), color='green', linestyle='--', linewidth=2, label=f'Mediana: {np.median(y_test):.2f}')
    axes[1].set_xlabel('Nota Final', fontsize=12)
    axes[1].set_ylabel('Frecuencia', fontsize=12)
    axes[1].set_title('Distribución - Conjunto de Test', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'distribucion_calificaciones.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico guardado en: {save_path}")


def _analyze_predictions_detailed(y_test, y_pred, output_dir: Path, logger):
    """Analiza las predicciones vs valores reales."""
    logger.info("\n" + "="*70)
    logger.info("=== ANÁLISIS DE PREDICCIONES ===")
    logger.info("="*70)
    
    # Calcular métricas
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"\n📊 MÉTRICAS DEL MODELO:")
    logger.info(f"  RMSE:  {rmse:.4f}")
    logger.info(f"  MAE:   {mae:.4f}")
    logger.info(f"  R²:    {r2:.4f}")
    
    # Estadísticas de predicciones
    logger.info(f"\n📊 ESTADÍSTICAS DE PREDICCIONES:")
    logger.info(f"  Media:        {y_pred.mean():.2f}")
    logger.info(f"  Std Dev:      {y_pred.std():.2f}")
    logger.info(f"  Min:          {y_pred.min():.2f}")
    logger.info(f"  Max:          {y_pred.max():.2f}")
    
    # Calcular residuos
    residuos = y_test - y_pred
    
    logger.info(f"\n📊 ESTADÍSTICAS DE RESIDUOS:")
    logger.info(f"  Media:        {residuos.mean():.4f}")
    logger.info(f"  Std Dev:      {residuos.std():.4f}")
    logger.info(f"  Min:          {residuos.min():.2f}")
    logger.info(f"  Max:          {residuos.max():.2f}")
    logger.info(f"  Q1:           {np.quantile(residuos, 0.25):.2f}")
    logger.info(f"  Q3:           {np.quantile(residuos, 0.75):.2f}")
    
    # Crear visualizaciones
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Scatter plot: Predicciones vs Reales
    axes[0].scatter(y_test, y_pred, alpha=0.5, s=30, color='steelblue')
    
    # Línea de referencia perfecta
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción perfecta')
    
    axes[0].set_xlabel('Nota Real', fontsize=12)
    axes[0].set_ylabel('Nota Predicha', fontsize=12)
    axes[0].set_title(f'Predicciones vs Real\n(R² = {r2:.4f}, MAE = {mae:.4f})', 
                      fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Histograma de residuos
    axes[1].hist(residuos, bins=40, alpha=0.7, color='coral', edgecolor='black')
    axes[1].axvline(0, color='black', linestyle='--', linewidth=2, label='Error = 0')
    axes[1].axvline(residuos.mean(), color='blue', linestyle='--', linewidth=2, 
                    label=f'Media = {residuos.mean():.2f}')
    axes[1].set_xlabel('Error (Real - Predicción)', fontsize=12)
    axes[1].set_ylabel('Frecuencia', fontsize=12)
    axes[1].set_title('Distribución de Residuos', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'predicciones_vs_real.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico guardado en: {save_path}")
    
    return rmse, mae, r2


def _analyze_residuals_detailed(y_test, y_pred, output_dir: Path, logger):
    """Análisis detallado de residuos."""
    logger.info("\n" + "="*70)
    logger.info("=== ANÁLISIS DETALLADO DE RESIDUOS ===")
    logger.info("="*70)
    
    residuos = y_test - y_pred
    
    # Crear figura con 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Residuos vs Predicciones
    axes[0, 0].scatter(y_pred, residuos, alpha=0.5, s=30, color='purple')
    axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Nota Predicha', fontsize=11)
    axes[0, 0].set_ylabel('Residuo (Real - Pred)', fontsize=11)
    axes[0, 0].set_title('Residuos vs Predicciones', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Q-Q Plot
    stats.probplot(residuos, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot (Normalidad de Residuos)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residuos vs Valores Reales
    axes[1, 0].scatter(y_test, residuos, alpha=0.5, s=30, color='teal')
    axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Nota Real', fontsize=11)
    axes[1, 0].set_ylabel('Residuo (Real - Pred)', fontsize=11)
    axes[1, 0].set_title('Residuos vs Valores Reales', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Residuos Absolutos vs Predicciones
    residuos_abs = np.abs(residuos)
    axes[1, 1].scatter(y_pred, residuos_abs, alpha=0.5, s=30, color='orange')
    axes[1, 1].set_xlabel('Nota Predicha', fontsize=11)
    axes[1, 1].set_ylabel('|Residuo|', fontsize=11)
    axes[1, 1].set_title('Residuos Absolutos vs Predicciones', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'analisis_residuos_detallado.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test de normalidad
    _, p_value = stats.shapiro(residuos)
    logger.info(f"\n📊 TEST DE NORMALIDAD (Shapiro-Wilk):")
    logger.info(f"  p-value: {p_value:.6f}")
    if p_value > 0.05:
        logger.info(f"  ✅ Los residuos parecen seguir una distribución normal")
    else:
        logger.info(f"  ⚠️  Los residuos NO siguen una distribución normal")
    
    logger.info(f"✅ Gráfico guardado en: {save_path}")


def _compare_with_baseline(y_train, y_test, mae_model, r2_model, logger):
    """Compara el modelo con un baseline simple."""
    logger.info("\n" + "="*70)
    logger.info("=== COMPARACIÓN CON BASELINE ===")
    logger.info("="*70)
    
    # Baseline: predecir siempre la media del train
    baseline_pred = np.full(len(y_test), y_train.mean())
    
    mae_baseline = mean_absolute_error(y_test, baseline_pred)
    r2_baseline = r2_score(y_test, baseline_pred)
    rmse_baseline = np.sqrt(mean_squared_error(y_test, baseline_pred))
    
    logger.info(f"\n📊 BASELINE (Predecir siempre la media del train = {y_train.mean():.2f}):")
    logger.info(f"  RMSE:  {rmse_baseline:.4f}")
    logger.info(f"  MAE:   {mae_baseline:.4f}")
    logger.info(f"  R²:    {r2_baseline:.4f}")
    
    logger.info(f"\n📊 COMPARACIÓN:")
    logger.info(f"  MAE - Modelo:    {mae_model:.4f}")
    logger.info(f"  MAE - Baseline:  {mae_baseline:.4f}")
    logger.info(f"  Mejora en MAE:   {((mae_baseline - mae_model)/mae_baseline * 100):.2f}%")
    
    logger.info(f"\n  R² - Modelo:     {r2_model:.4f}")
    logger.info(f"  R² - Baseline:   {r2_baseline:.4f}")
    if r2_baseline < 0:
        logger.info(f"  Mejora en R²:    {r2_model - r2_baseline:.4f} puntos (baseline negativo)")
    else:
        logger.info(f"  Mejora en R²:    {((r2_model - r2_baseline)/abs(r2_baseline) * 100):.2f}%")
    
    # Interpretación
    logger.info(f"\n💡 INTERPRETACIÓN:")
    if mae_model < mae_baseline:
        mejora_pct = ((mae_baseline - mae_model)/mae_baseline * 100)
        logger.info(f"  ✅ El modelo es {mejora_pct:.1f}% mejor que el baseline en MAE")
        logger.info(f"  ✅ El modelo reduce el error en {mae_baseline - mae_model:.2f} puntos")
    else:
        logger.info(f"  ⚠️  El modelo NO supera al baseline")
    
    if r2_model > 0.5:
        logger.info(f"  ✅ R² > 0.5: El modelo explica más del 50% de la varianza")
    elif r2_model > 0.3:
        logger.info(f"  ⚠️  R² moderado: El modelo explica ~{r2_model*100:.0f}% de la varianza")
    else:
        logger.info(f"  ⚠️  R² bajo: El modelo tiene dificultades para explicar la varianza")


def _analyze_errors_by_range_detailed(y_test, y_pred, output_dir: Path, logger):
    """Analiza los errores segmentados por rangos de notas."""
    logger.info("\n" + "="*70)
    logger.info("=== ANÁLISIS DE ERRORES POR RANGO DE NOTAS ===")
    logger.info("="*70)
    
    # Crear rangos apropiados para escala 0-100
    bins = [0, 60, 70, 80, 90, 100]
    labels = ['0-60', '60-70', '70-80', '80-90', '90-100']
    
    # Calcular errores estándar y weighted
    error_standard = np.abs(y_test - y_pred)
    error_weighted = []
    
    # Calcular error weighted para cada muestra
    for i in range(len(y_test)):
        weight = 2.0 if y_test[i] <= 60 else 1.0
        error_weighted.append(weight * np.abs(y_test[i] - y_pred[i]))
    
    error_weighted = np.array(error_weighted)
    
    df_analysis = pd.DataFrame({
        'real': y_test,
        'pred': y_pred,
        'error_standard': error_standard,
        'error_weighted': error_weighted,
        'rango': pd.cut(y_test, bins=bins, labels=labels, include_lowest=True)
    })
    
    logger.info("\n📊 ERRORES POR RANGO DE NOTAS:")
    for rango in labels:
        subset = df_analysis[df_analysis['rango'] == rango]
        if len(subset) > 0:
            mae_rango = subset['error_standard'].mean()
            wmae_rango = subset['error_weighted'].mean()
            logger.info(f"\n  {rango}:")
            logger.info(f"    Cantidad:         {len(subset)}")
            logger.info(f"    MAE:              {mae_rango:.4f}")
            logger.info(f"    Weighted MAE:     {wmae_rango:.4f}")
            logger.info(f"    Error máximo:     {subset['error_standard'].max():.4f}")
            logger.info(f"    Error mínimo:     {subset['error_standard'].min():.4f}")
    
    # Crear visualización comparativa
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Preparar datos para ambos gráficos
    rangos_con_datos_std = df_analysis.groupby('rango', observed=False)['error_standard'].apply(list)
    rangos_con_datos_wmae = df_analysis.groupby('rango', observed=False)['error_weighted'].apply(list)
    
    positions = []
    data_to_plot_std = []
    data_to_plot_wmae = []
    labels_to_plot = []
    
    for i, rango in enumerate(labels):
        if rango in rangos_con_datos_std.index:
            positions.append(i)
            data_to_plot_std.append(rangos_con_datos_std[rango])
            data_to_plot_wmae.append(rangos_con_datos_wmae[rango])
            labels_to_plot.append(rango)
    
    # Gráfico 1: MAE estándar
    bp1 = axes[0].boxplot(data_to_plot_std, positions=positions, tick_labels=labels_to_plot, patch_artist=True)
    
    # Colorear las cajas
    colors = ['#ff9999', '#ffcc99', '#99ccff', '#99ff99', '#cc99ff']
    for patch, color in zip(bp1['boxes'], colors[:len(bp1['boxes'])]):
        patch.set_facecolor(color)
    
    axes[0].set_xlabel('Rango de Notas Reales', fontsize=12)
    axes[0].set_ylabel('Error Absoluto (MAE)', fontsize=12)
    axes[0].set_title('Distribución de Errores - MAE Estándar', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=15)
    
    # Gráfico 2: Weighted MAE
    bp2 = axes[1].boxplot(data_to_plot_wmae, positions=positions, tick_labels=labels_to_plot, patch_artist=True)
    
    # Colorear las cajas con colores más intensos para weighted
    colors_weighted = ['#ff6666', '#ff9966', '#6699ff', '#66ff66', '#cc66ff']
    for patch, color in zip(bp2['boxes'], colors_weighted[:len(bp2['boxes'])]):
        patch.set_facecolor(color)
    
    axes[1].set_xlabel('Rango de Notas Reales', fontsize=12)
    axes[1].set_ylabel('Error Ponderado (Weighted MAE)', fontsize=12)
    axes[1].set_title('Distribución de Errores - Weighted MAE (2x para 0-60)', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=15)
    
    # Resaltar el rango crítico en ambos gráficos
    for ax in axes:
        ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Rango Crítico')
        ax.legend()
    
    plt.tight_layout()
    save_path = output_dir / 'errores_por_rango_comparativo.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico comparativo guardado en: {save_path}")
    
    # Crear gráfico adicional: Comparación directa de promedios por rango
    _create_mae_comparison_by_range(df_analysis, labels, output_dir, logger)


def _create_mae_comparison_by_range(df_analysis, labels, output_dir: Path, logger):
    """Crea gráfico de barras comparando MAE vs Weighted MAE por rango."""
    
    # Calcular promedios por rango
    mae_by_range = []
    wmae_by_range = []
    ranges_with_data = []
    sample_counts = []
    
    for rango in labels:
        subset = df_analysis[df_analysis['rango'] == rango]
        if len(subset) > 0:
            mae_by_range.append(subset['error_standard'].mean())
            wmae_by_range.append(subset['error_weighted'].mean())
            ranges_with_data.append(rango)
            sample_counts.append(len(subset))
    
    if not ranges_with_data:
        logger.warning("No hay datos para crear gráfico de comparación por rango")
        return
    
    # Crear gráfico de barras comparativo
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(ranges_with_data))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, mae_by_range, width, label='MAE Estándar', 
                   alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, wmae_by_range, width, label='Weighted MAE (2x para 0-60)', 
                   alpha=0.8, color='coral')
    
    # Agregar valores en las barras
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.05,
                f'{height1:.2f}', ha='center', va='bottom', fontsize=10)
        ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.05,
                f'{height2:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Agregar número de muestras
        ax.text(i, -0.5, f'n={sample_counts[i]}', ha='center', va='top', 
                fontsize=9, style='italic', color='gray')
    
    ax.set_xlabel('Rango de Notas Reales', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error Promedio', fontsize=12, fontweight='bold')
    ax.set_title('Comparación MAE vs Weighted MAE por Rango\n(Weighted MAE penaliza 2x los errores en rango 0-60)', 
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ranges_with_data, rotation=20, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Resaltar el rango crítico
    if '0-60' in ranges_with_data:
        critical_idx = ranges_with_data.index('0-60')
        ax.axvspan(critical_idx - 0.4, critical_idx + 0.4, alpha=0.2, color='red', 
                   label='Rango Crítico')
        ax.legend(fontsize=11)
    
    plt.tight_layout()
    save_path = output_dir / 'mae_vs_weighted_mae_por_rango.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico de comparación MAE guardado en: {save_path}")


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

def analyze_all_models_by_range(y_test, predictions, results_df, logger):
    """Analiza métricas de todos los modelos por rango de calificaciones."""
    
    # Definir rangos
    ranges = [
        (0, 60, "0-60"),
        (60, 70, "60-70"),
        (70, 80, "70-80"),
        (80, 90, "80-90"),
        (90, 100, "90-100")
    ]
    
    # Crear directorio para guardar resultados
    output_dir = project_root / 'models' / 'summary'
    output_dir.mkdir(exist_ok=True)
    
    # Diccionario para almacenar resultados por modelo y rango
    all_results = []
    
    for model_name, y_pred in predictions.items():
        if y_pred is None:
            continue
            
        logger.info(f"\n📊 Analizando {model_name}...")
        
        for min_val, max_val, label in ranges:
            # Filtrar datos del rango
            mask = (y_test >= min_val) & (y_test < max_val) if max_val < 100 else (y_test >= min_val) & (y_test <= max_val)
            
            if mask.sum() == 0:
                continue
                
            y_true_range = y_test[mask]
            y_pred_range = y_pred[mask]
            
            # Calcular métricas
            n_samples = len(y_true_range)
            mae = mean_absolute_error(y_true_range, y_pred_range)
            rmse = np.sqrt(mean_squared_error(y_true_range, y_pred_range))
            w_mae = weighted_mae(y_true_range, y_pred_range, weight_low=2.0, weight_high=1.0, threshold=60)
            
            if y_true_range.std() > 0:
                r2 = r2_score(y_true_range, y_pred_range)
            else:
                r2 = np.nan
            
            all_results.append({
                'Modelo': model_name,
                'Rango': label,
                'N': n_samples,
                '%': (n_samples / len(y_test)) * 100,
                'MAE': mae,
                'RMSE': rmse,
                'Weighted_MAE': w_mae,
                'R²': r2
            })
    
    # Crear DataFrame con todos los resultados
    df_all_results = pd.DataFrame(all_results)
    
    # Guardar CSV
    csv_path = output_dir / 'metricas_por_rango_todos_modelos.csv'
    df_all_results.to_csv(csv_path, index=False)
    logger.info(f"\n✅ Resultados por rango guardados en: {csv_path}")
    
    # Crear visualizaciones comparativas
    create_comparison_visualizations(df_all_results, output_dir, logger)
    
    # Mostrar resumen en consola
    logger.info("\n" + "="*80)
    logger.info("📊 RESUMEN POR RANGO - TODOS LOS MODELOS")
    logger.info("="*80)
    
    for rango in ["0-60", "60-70", "70-80", "80-90", "90-100"]:
        df_rango = df_all_results[df_all_results['Rango'] == rango]
        if len(df_rango) > 0:
            logger.info(f"\n{rango}:")
            logger.info(f"  Mejor MAE:          {df_rango.loc[df_rango['MAE'].idxmin(), 'Modelo']} ({df_rango['MAE'].min():.4f})")
            logger.info(f"  Mejor Weighted MAE: {df_rango.loc[df_rango['Weighted_MAE'].idxmin(), 'Modelo']} ({df_rango['Weighted_MAE'].min():.4f})")
            if not df_rango['R²'].isna().all():
                logger.info(f"  Mejor R²:           {df_rango.loc[df_rango['R²'].idxmax(), 'Modelo']} ({df_rango['R²'].max():.4f})")


def create_comparison_visualizations(df_results, output_dir, logger):
    """Crea visualizaciones comparativas de todos los modelos."""
    
    logger.info("\n📊 Generando visualizaciones comparativas...")
    
    # Configurar estilo
    sns.set_style("whitegrid")
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    # 1. Comparación MAE por rango para todos los modelos
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1.1 MAE por rango
    rangos_unicos = df_results['Rango'].unique()
    x = np.arange(len(rangos_unicos))
    width = 0.15
    
    modelos = df_results['Modelo'].unique()
    
    for i, modelo in enumerate(modelos):
        df_modelo = df_results[df_results['Modelo'] == modelo]
        mae_values = [df_modelo[df_modelo['Rango'] == r]['MAE'].values[0] if r in df_modelo['Rango'].values else 0 
                      for r in rangos_unicos]
        axes[0, 0].bar(x + i * width, mae_values, width, label=modelo, alpha=0.8)
    
    axes[0, 0].set_xlabel('Rango de Calificaciones', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('MAE', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('MAE por Rango - Comparación de Modelos', fontsize=13, fontweight='bold')
    axes[0, 0].set_xticks(x + width * (len(modelos) - 1) / 2)
    axes[0, 0].set_xticklabels(rangos_unicos, rotation=20, ha='right')
    axes[0, 0].legend(loc='best')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].axvline(0.5, color='red', linestyle='--', alpha=0.3, linewidth=2)
    
    # 1.2 Weighted MAE por rango
    for i, modelo in enumerate(modelos):
        df_modelo = df_results[df_results['Modelo'] == modelo]
        wmae_values = [df_modelo[df_modelo['Rango'] == r]['Weighted_MAE'].values[0] if r in df_modelo['Rango'].values else 0 
                       for r in rangos_unicos]
        axes[0, 1].bar(x + i * width, wmae_values, width, label=modelo, alpha=0.8)
    
    axes[0, 1].set_xlabel('Rango de Calificaciones', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Weighted MAE', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Weighted MAE por Rango - Comparación de Modelos', fontsize=13, fontweight='bold')
    axes[0, 1].set_xticks(x + width * (len(modelos) - 1) / 2)
    axes[0, 1].set_xticklabels(rangos_unicos, rotation=20, ha='right')
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].axvline(0.5, color='red', linestyle='--', alpha=0.3, linewidth=2)
    
    # 1.3 R² por rango
    df_r2 = df_results[~df_results['R²'].isna()].copy()
    if len(df_r2) > 0:
        rangos_r2 = df_r2['Rango'].unique()
        x_r2 = np.arange(len(rangos_r2))
        
        for i, modelo in enumerate(modelos):
            df_modelo = df_r2[df_r2['Modelo'] == modelo]
            r2_values = [df_modelo[df_modelo['Rango'] == r]['R²'].values[0] if r in df_modelo['Rango'].values else 0 
                         for r in rangos_r2]
            axes[1, 0].bar(x_r2 + i * width, r2_values, width, label=modelo, alpha=0.8)
        
        axes[1, 0].set_xlabel('Rango de Calificaciones', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('R²', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('R² por Rango - Comparación de Modelos', fontsize=13, fontweight='bold')
        axes[1, 0].set_xticks(x_r2 + width * (len(modelos) - 1) / 2)
        axes[1, 0].set_xticklabels(rangos_r2, rotation=20, ha='right')
        axes[1, 0].legend(loc='best')
        axes[1, 0].axhline(0, color='black', linestyle='-', linewidth=0.8)
        axes[1, 0].axhline(0.5, color='green', linestyle='--', alpha=0.5, label='R²=0.5')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 1.4 Heatmap de MAE
    pivot_mae = df_results.pivot(index='Modelo', columns='Rango', values='MAE')
    sns.heatmap(pivot_mae, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=axes[1, 1], 
                cbar_kws={'label': 'MAE'}, linewidths=0.5)
    axes[1, 1].set_title('Heatmap: MAE por Modelo y Rango', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Rango de Calificaciones', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Modelo', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    save_path = output_dir / 'comparacion_todos_modelos_por_rango.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Visualización guardada en: {save_path}")
    
    # 2. Gráfico específico para rango crítico (0-60)
    create_critical_range_visualization(df_results, output_dir, logger)


def create_critical_range_visualization(df_results, output_dir, logger):
    """Crea visualización enfocada en el rango crítico 0-60."""
    
    df_critical = df_results[df_results['Rango'] == '0-60 (CRÍTICO)'].copy()
    
    if len(df_critical) == 0:
        logger.warning("⚠️  No hay datos en el rango crítico para visualizar")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 2.1 MAE vs Weighted MAE en rango crítico
    x = np.arange(len(df_critical))
    width = 0.35
    
    axes[0].bar(x - width/2, df_critical['MAE'], width, label='MAE estándar', alpha=0.8, color='steelblue')
    axes[0].bar(x + width/2, df_critical['Weighted_MAE'], width, label='Weighted MAE (2x)', alpha=0.8, color='coral')
    
    axes[0].set_xlabel('Modelo', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Error', fontsize=12, fontweight='bold')
    axes[0].set_title('Rango CRÍTICO (0-60): MAE vs Weighted MAE', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_critical['Modelo'], rotation=25, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 2.2 Porcentaje de muestras + mejor modelo
    models = df_critical['Modelo'].tolist()
    mae_values = df_critical['MAE'].tolist()
    
    colors_bar = ['green' if mae == min(mae_values) else 'orange' for mae in mae_values]
    
    axes[1].barh(models, mae_values, alpha=0.8, color=colors_bar)
    axes[1].set_xlabel('MAE', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Modelo', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Mejor Modelo en Rango Crítico\n({df_critical.iloc[0]["N"]} muestras, {df_critical.iloc[0]["%"]:.1f}%)', 
                      fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].invert_yaxis()
    
    # Anotar el mejor
    best_idx = df_critical['MAE'].idxmin()
    best_model = df_critical.loc[best_idx, 'Modelo']
    best_mae = df_critical.loc[best_idx, 'MAE']
    axes[1].text(best_mae + 0.1, models.index(best_model), f'✓ Mejor: {best_mae:.2f}', 
                 fontweight='bold', color='green', va='center')
    
    plt.tight_layout()
    save_path = output_dir / 'analisis_rango_critico.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Análisis de rango crítico guardado en: {save_path}")


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
        
        # Calcular predicciones para Weighted MAE
        logger.info(f"🔮 Calculando predicciones para métricas adicionales...")
        import joblib
        model_path = project_root / 'models' / model.model_name / ts / 'model.pkl'
        model_data = joblib.load(model_path)
        pipeline = model_data['pipeline']
        y_pred = pipeline.predict(X_test)
        
        # Calcular Weighted MAE
        w_mae = weighted_mae(y_test.values, y_pred, weight_low=2.0, weight_high=1.0, threshold=60)
        metrics['weighted_mae'] = w_mae

        logger.info(f"✅ {model_name} evaluado exitosamente")
        logger.info(f"   RMSE:         {metrics['rmse']:.4f}")
        logger.info(f"   MAE:          {metrics['mae']:.4f}")
        logger.info(f"   Weighted MAE: {metrics['weighted_mae']:.4f}")
        logger.info(f"   R²:           {metrics['r2']:.4f}")

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
            mlflow.log_metric("weighted_mae_test", float(metrics['weighted_mae']))
            mlflow.log_metric("r2_test", float(metrics['r2']))

            logger.info(f"Run {run_id} completado para {model_name}")
        logger.info(f"{model_name} evaluación completada y registrada en MLflow")
        return metrics, y_pred

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
        
        # Obtener predicciones para Weighted MAE
        logger.info(f"🔮 Calculando predicciones para métricas adicionales...")
        test_h2o = h2o.H2OFrame(test_df)
        
        # Cargar modelo
        import joblib
        metadata_path = project_root / 'models' / 'h2o' / ts / 'model_metadata.pkl'
        with open(metadata_path, 'rb') as f:
            metadata = joblib.load(f)
        model_path = metadata['model_path']
        h2o_model = h2o.load_model(model_path)
        
        predictions_h2o = h2o_model.predict(test_h2o)
        y_pred = predictions_h2o.as_data_frame()['predict'].values
        y_test = test_df[TARGET_FEATURE].values
        
        # Calcular Weighted MAE
        w_mae = weighted_mae(y_test, y_pred, weight_low=2.0, weight_high=1.0, threshold=60)
        metrics['weighted_mae'] = w_mae

        logger.info(f"✅ {model_name} evaluado exitosamente")
        logger.info(f"   RMSE:         {metrics['rmse']:.4f}")
        logger.info(f"   MAE:          {metrics['mae']:.4f}")
        logger.info(f"   Weighted MAE: {metrics['weighted_mae']:.4f}")
        logger.info(f"   R²:           {metrics['r2']:.4f}")

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
            mlflow.log_metric("weighted_mae_test", float(metrics['weighted_mae']))
            mlflow.log_metric("r2_test", float(metrics['r2']))

            logger.info(f"Run {run_id} completado para {model_name}")

        logger.info(f"{model_name} evaluación completada y registrada en MLflow")
        return metrics, y_pred

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
        (LinearRegressionPipeline, "LinearRegression", "20251029_014923"),
        (ElasticNetPipeline, "ElasticNet", "20251030_115639"),  # Modelo con weighted_mae
        (RandomForestPipeline, "RandomForest", "20251030_115706"),
        (CatBoostPipeline, "CatBoost", "20251030_123954")
    ]

    # Diccionario para almacenar resultados
    results = {}
    predictions = {}  # Guardar predicciones para análisis posterior

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

        # Cargar y_train para comparación con baseline en diagnósticos
        logger.info("\n📊 Cargando datos de entrenamiento para baseline...")
        train_df = pd.read_csv('data/processed/train_moodle.csv')
        y_train = train_df[TARGET_FEATURE]
        logger.info(f"✅ Train set cargado: {len(y_train)} muestras")
        
        for model_class, model_name, ts in models_config:
            try:
                metrics, y_pred = evaluate_model(model_class, model_name, ts, X_test, y_test, logger)
                results[model_name] = metrics
                predictions[model_name] = y_pred
                
                # Realizar diagnóstico completo para este modelo
                # Guardar en el directorio timestamp del modelo
                logger.info(f"\n{'='*80}")
                logger.info(f"🔬 EJECUTANDO DIAGNÓSTICO COMPLETO PARA {model_name}")
                logger.info(f"{'='*80}")
                
                # Crear instancia del modelo para obtener el nombre del directorio
                model_instance = model_class(random_state=42)
                model_dir = project_root / 'models' / model_instance.model_name / ts
                
                diagnose_model_predictions(
                    model_name=model_name,
                    y_train=y_train,
                    y_test=y_test.values,
                    y_pred=y_pred,
                    model_dir=model_dir,
                    logger=logger
                )
                
            except Exception as e:
                logger.error(f"Error evaluando {model_name}, continuando con el siguiente...")
                logger.error(traceback.format_exc())
                results[model_name] = None
                predictions[model_name] = None

        # Evaluar H2O AutoML
        logger.info("\nEvaluando H2O AutoML...")
        h2o_timestamp = "20251030_155646"
        try:
            h2o_metrics, h2o_pred = evaluate_h2o_model(df_test, h2o_timestamp, logger)
            results["H2O_AutoML"] = h2o_metrics
            predictions["H2O_AutoML"] = h2o_pred
            
            # Realizar diagnóstico completo para H2O
            # Guardar en el directorio timestamp del modelo H2O
            logger.info(f"\n{'='*80}")
            logger.info(f"🔬 EJECUTANDO DIAGNÓSTICO COMPLETO PARA H2O_AutoML")
            logger.info(f"{'='*80}")
            
            h2o_model_dir = project_root / 'models' / 'h2o' / h2o_timestamp
            
            diagnose_model_predictions(
                model_name="H2O_AutoML",
                y_train=y_train,
                y_test=y_test.values,
                y_pred=h2o_pred,
                model_dir=h2o_model_dir,
                logger=logger
            )
            
        except Exception as e:
            logger.error(f"Error evaluando H2O AutoML, continuando...")
            logger.error(traceback.format_exc())
            results["H2O_AutoML"] = None
            predictions["H2O_AutoML"] = None

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
                'Weighted_MAE': metrics['weighted_mae'],
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
        
        # Análisis adicional: Comparación de MAE vs Weighted MAE
        logger.info("\n" + "="*70)
        logger.info("📊 ANÁLISIS: MAE vs WEIGHTED MAE")
        logger.info("="*70)
        
        for _, row in results_df.iterrows():
            ratio = row['Weighted_MAE'] / row['MAE']
            logger.info(f"\n{row['Modelo']}:")
            logger.info(f"  MAE:          {row['MAE']:.4f}")
            logger.info(f"  Weighted MAE: {row['Weighted_MAE']:.4f}")
            logger.info(f"  Ratio:        {ratio:.2f}x")
            if ratio > 1.5:
                logger.info(f"  ⚠️  Alto impacto de penalización → Errores significativos en rango 0-60")
            elif ratio > 1.2:
                logger.info(f"  ⚠️  Impacto moderado → Algunos errores en rango crítico")
            else:
                logger.info(f"  ✅ Bajo impacto → Buen rendimiento en rango crítico")
        
        # Generar análisis por rangos para todos los modelos
        logger.info("\n" + "="*70)
        logger.info("📊 GENERANDO ANÁLISIS DETALLADO POR RANGOS...")
        logger.info("="*70)
        
        analyze_all_models_by_range(y_test, predictions, results_df, logger)

    logger.info("\n✅ Evaluación de todos los modelos completada")
    logger.info(f"Los resultados se guardaron en el experimento: All_Models_Test_Evaluation")
    logger.info(f"Para ver los resultados: cd {os.path.dirname(mlflow_dir_abs)} && mlflow ui")
    
    # Resumen final de archivos generados
    logger.info("\n" + "="*80)
    logger.info("📁 ARCHIVOS GENERADOS")
    logger.info("="*80)
    logger.info(f"\n1. Resultados Generales:")
    logger.info(f"   • {project_root / 'models' / 'test_evaluation_results.csv'}")
    logger.info(f"\n2. Análisis por Rangos (Todos los modelos):")
    logger.info(f"   • {project_root / 'models' / 'summary' / 'metricas_por_rango_todos_modelos.csv'}")
    logger.info(f"   • {project_root / 'models' / 'summary' / 'comparacion_todos_modelos_por_rango.png'}")
    logger.info(f"   • {project_root / 'models' / 'summary' / 'analisis_rango_critico.png'}")
    logger.info(f"\n3. Diagnósticos Detallados (guardados en cada modelo):")
    logger.info(f"   Estructura: models/[tipo_modelo]/[timestamp]/diagnostics/")
    logger.info(f"   Ejemplos:")
    logger.info(f"     • models/elasticnet/20251030_111313/diagnostics/")
    logger.info(f"     • models/h2o/20251030_155646/diagnostics/")
    logger.info(f"   Archivos por modelo:")
    logger.info(f"     • distribucion_calificaciones.png")
    logger.info(f"     • predicciones_vs_real.png")
    logger.info(f"     • analisis_residuos_detallado.png")
    logger.info(f"     • errores_por_rango.png")


if __name__ == "__main__":
    main()

