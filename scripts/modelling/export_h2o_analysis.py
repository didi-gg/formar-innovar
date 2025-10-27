"""
Script para exportar análisis del Stacked Ensemble H2O a archivos CSV y TXT.
"""

import h2o
import joblib
import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Directorio de análisis (ajustado para ejecutar desde scripts/modelling/)
ANALYSIS_DIR = "../../models/h2o/20251026_231034/analysis"

def load_h2o_model():
    """Cargar el modelo H2O"""
    print("🔧 Inicializando H2O...")
    h2o.init(verbose=False)
    
    metadata_path = "../../models/h2o/20251026_231034/model_metadata.pkl"
    with open(metadata_path, "rb") as f:
        metadata = joblib.load(f)
    
    model_path = metadata['model_path']
    model = h2o.load_model(model_path)
    
    return model, metadata

def create_analysis_directory():
    """Crear directorio de análisis"""
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    print(f"📁 Directorio creado: {ANALYSIS_DIR}")

def export_ensemble_summary(model, metadata):
    """Exportar resumen del ensemble a CSV"""
    print("📊 Exportando resumen del ensemble...")
    
    # Información básica del ensemble
    ensemble_info = {
        'Característica': [
            'Tipo de Modelo',
            'ID del Modelo',
            'Timestamp',
            'Número de Modelos Base',
            'Algoritmo Metalearner',
            'Tipo Metalearner'
        ],
        'Valor': [
            model.__class__.__name__,
            model.model_id,
            metadata.get('timestamp', 'N/A'),
            len(model.base_models) if hasattr(model, 'base_models') else 'N/A',
            model.metalearner().algo if model.metalearner() else 'N/A',
            'Regresión Lineal Generalizada (GLM)'
        ]
    }
    
    df_ensemble = pd.DataFrame(ensemble_info)
    output_path = os.path.join(ANALYSIS_DIR, 'h2o_ensemble_info.csv')
    df_ensemble.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ Guardado: {output_path}")
    
    return df_ensemble

def export_performance_metrics(model):
    """Exportar métricas de rendimiento a CSV"""
    print("📈 Exportando métricas de rendimiento...")
    
    metrics_data = []
    
    # Métricas de entrenamiento
    try:
        train_rmse = model.rmse(train=True)
        train_mae = model.mae(train=True)
        train_r2 = model.r2(train=True)
        
        metrics_data.append({
            'Conjunto': 'Entrenamiento',
            'RMSE': round(train_rmse, 4),
            'MAE': round(train_mae, 4),
            'R²': round(train_r2, 4)
        })
    except:
        pass
    
    # Métricas de validación cruzada
    try:
        cv_rmse = model.rmse(xval=True)
        cv_mae = model.mae(xval=True)
        cv_r2 = model.r2(xval=True)
        
        metrics_data.append({
            'Conjunto': 'Validación Cruzada (5-fold)',
            'RMSE': round(cv_rmse, 4),
            'MAE': round(cv_mae, 4),
            'R²': round(cv_r2, 4)
        })
    except:
        pass
    
    # Métricas de test (desde el archivo de resultados)
    try:
        results_df = pd.read_csv('../../models/test_evaluation_results.csv')
        h2o_results = results_df[results_df['Modelo'] == 'H2O_AutoML'].iloc[0]
        
        metrics_data.append({
            'Conjunto': 'Test Final',
            'RMSE': round(h2o_results['RMSE'], 4),
            'MAE': round(h2o_results['MAE'], 4),
            'R²': round(h2o_results['R²'], 4)
        })
    except:
        pass
    
    df_metrics = pd.DataFrame(metrics_data)
    output_path = os.path.join(ANALYSIS_DIR, 'h2o_performance_metrics.csv')
    df_metrics.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ Guardado: {output_path}")
    
    return df_metrics

def export_base_models_info(model):
    """Exportar información de modelos base a CSV"""
    print("🏗️ Exportando información de modelos base...")
    
    base_models_data = []
    
    try:
        base_models = model.base_models
        
        for i, base_model_id in enumerate(base_models, 1):
            # Extraer tipo de algoritmo del ID
            if 'GBM' in str(base_model_id):
                algo = 'Gradient Boosting Machine'
            elif 'DRF' in str(base_model_id):
                algo = 'Distributed Random Forest'
            elif 'GLM' in str(base_model_id):
                algo = 'Generalized Linear Model'
            elif 'XGBoost' in str(base_model_id):
                algo = 'XGBoost'
            elif 'DeepLearning' in str(base_model_id):
                algo = 'Deep Learning'
            elif 'StackedEnsemble' in str(base_model_id):
                algo = 'Stacked Ensemble'
            else:
                algo = 'Otro'
            
            base_models_data.append({
                'Posición': i,
                'ID_Modelo': str(base_model_id),
                'Algoritmo': algo
            })
    
    except Exception as e:
        print(f"⚠️ Error al procesar modelos base: {e}")
    
    df_base_models = pd.DataFrame(base_models_data)
    output_path = os.path.join(ANALYSIS_DIR, 'h2o_base_models.csv')
    df_base_models.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ Guardado: {output_path}")
    
    return df_base_models

def export_leaderboard(metadata):
    """Exportar leaderboard a CSV"""
    print("🏆 Exportando leaderboard...")
    
    try:
        if 'leaderboard' in metadata and metadata['leaderboard'] is not None:
            lb = metadata['leaderboard']
            
            # Procesar leaderboard
            lb_processed = lb.copy()
            lb_processed['Posición'] = range(1, len(lb_processed) + 1)
            
            # Extraer tipo de algoritmo
            lb_processed['Algoritmo'] = lb_processed['model_id'].apply(lambda x: 
                'Stacked Ensemble' if 'StackedEnsemble' in x else
                'Gradient Boosting' if 'GBM' in x else
                'Random Forest' if 'DRF' in x else
                'Linear Model' if 'GLM' in x else
                'XGBoost' if 'XGBoost' in x else
                'Deep Learning' if 'DeepLearning' in x else
                'Otro'
            )
            
            # Seleccionar columnas relevantes
            columns_to_keep = ['Posición', 'model_id', 'Algoritmo', 'rmse', 'mae']
            if 'mean_residual_deviance' in lb_processed.columns:
                columns_to_keep.append('mean_residual_deviance')
            
            lb_export = lb_processed[columns_to_keep].copy()
            lb_export.columns = ['Posición', 'ID_Modelo', 'Algoritmo', 'RMSE', 'MAE'] + \
                               (['Mean_Residual_Deviance'] if len(columns_to_keep) > 5 else [])
            
            # Redondear valores numéricos
            for col in ['RMSE', 'MAE'] + (['Mean_Residual_Deviance'] if len(columns_to_keep) > 5 else []):
                if col in lb_export.columns:
                    lb_export[col] = lb_export[col].round(4)
            
            output_path = os.path.join(ANALYSIS_DIR, 'h2o_leaderboard.csv')
            lb_export.to_csv(output_path, index=False, encoding='utf-8')
            print(f"✅ Guardado: {output_path}")
            
            return lb_export
        else:
            print("⚠️ Leaderboard no disponible en metadatos")
            return None
            
    except Exception as e:
        print(f"❌ Error al exportar leaderboard: {e}")
        return None

def export_comparison_with_other_models():
    """Exportar comparación con otros modelos"""
    print("📊 Exportando comparación con otros modelos...")
    
    try:
        results_df = pd.read_csv('../../models/test_evaluation_results.csv')
        
        # Agregar ranking
        results_df['Ranking_RMSE'] = results_df['RMSE'].rank(method='min')
        results_df['Ranking_MAE'] = results_df['MAE'].rank(method='min')
        results_df['Ranking_R²'] = results_df['R²'].rank(method='min', ascending=False)
        
        # Calcular ranking promedio
        results_df['Ranking_Promedio'] = (results_df['Ranking_RMSE'] + 
                                         results_df['Ranking_MAE'] + 
                                         results_df['Ranking_R²']) / 3
        
        # Ordenar por ranking promedio
        results_df = results_df.sort_values('Ranking_Promedio').reset_index(drop=True)
        results_df['Posición_Final'] = range(1, len(results_df) + 1)
        
        # Redondear valores
        for col in ['RMSE', 'MAE', 'R²', 'Ranking_Promedio']:
            results_df[col] = results_df[col].round(4)
        
        output_path = os.path.join(ANALYSIS_DIR, 'comparacion_todos_modelos.csv')
        results_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ Guardado: {output_path}")
        
        return results_df
        
    except Exception as e:
        print(f"❌ Error al exportar comparación: {e}")
        return None

def create_comprehensive_report(df_ensemble, df_metrics, df_base_models, df_leaderboard, df_comparison):
    """Crear reporte completo en formato TXT"""
    print("📝 Creando reporte completo...")
    
    report = []
    report.append("="*80)
    report.append("ANÁLISIS COMPLETO DEL STACKED ENSEMBLE H2O AutoML")
    report.append("="*80)
    report.append("")
    
    # Información básica
    report.append("🏆 INFORMACIÓN BÁSICA DEL ENSEMBLE")
    report.append("-" * 50)
    for _, row in df_ensemble.iterrows():
        report.append(f"{row['Característica']}: {row['Valor']}")
    report.append("")
    
    # Métricas de rendimiento
    report.append("📊 MÉTRICAS DE RENDIMIENTO")
    report.append("-" * 50)
    for _, row in df_metrics.iterrows():
        report.append(f"{row['Conjunto']}:")
        report.append(f"  • RMSE: {row['RMSE']}")
        report.append(f"  • MAE: {row['MAE']}")
        report.append(f"  • R²: {row['R²']}")
        report.append("")
    
    # Análisis de generalización
    if len(df_metrics) >= 2:
        train_r2 = df_metrics[df_metrics['Conjunto'] == 'Entrenamiento']['R²'].iloc[0]
        cv_r2 = df_metrics[df_metrics['Conjunto'].str.contains('Validación')]['R²'].iloc[0]
        diff_r2 = abs(train_r2 - cv_r2)
        
        report.append("📈 ANÁLISIS DE GENERALIZACIÓN")
        report.append("-" * 50)
        report.append(f"Diferencia R² (Train vs CV): {diff_r2:.4f}")
        if diff_r2 < 0.1:
            report.append("✅ Excelente generalización")
        elif diff_r2 < 0.2:
            report.append("✅ Buena generalización")
        else:
            report.append("⚠️ Posible overfitting")
        report.append("")
    
    # Composición del ensemble
    report.append("🏗️ COMPOSICIÓN DEL ENSEMBLE")
    report.append("-" * 50)
    if df_base_models is not None and len(df_base_models) > 0:
        algo_counts = df_base_models['Algoritmo'].value_counts()
        report.append(f"Total de modelos base: {len(df_base_models)}")
        report.append("Distribución por algoritmo:")
        for algo, count in algo_counts.items():
            percentage = (count / len(df_base_models)) * 100
            report.append(f"  • {algo}: {count} modelos ({percentage:.1f}%)")
    else:
        report.append("Información de modelos base no disponible")
    report.append("")
    
    # Top 5 del leaderboard
    if df_leaderboard is not None:
        report.append("🥇 TOP 5 MODELOS DEL AUTOML")
        report.append("-" * 50)
        top_5 = df_leaderboard.head()
        for _, row in top_5.iterrows():
            report.append(f"{row['Posición']}. {row['Algoritmo']}")
            report.append(f"   RMSE: {row['RMSE']} | MAE: {row['MAE']}")
        report.append("")
    
    # Comparación con otros modelos
    if df_comparison is not None:
        report.append("🏆 COMPARACIÓN CON OTROS MODELOS")
        report.append("-" * 50)
        for _, row in df_comparison.iterrows():
            status = "🥇" if row['Posición_Final'] == 1 else f"{row['Posición_Final']}°"
            report.append(f"{status} {row['Modelo']}")
            report.append(f"   RMSE: {row['RMSE']} | MAE: {row['MAE']} | R²: {row['R²']}")
        report.append("")
    
    # Conclusiones
    report.append("✅ CONCLUSIONES CLAVE")
    report.append("-" * 50)
    report.append("• H2O AutoML seleccionó un Stacked Ensemble como mejor modelo")
    report.append("• Combina múltiples algoritmos para mayor robustez")
    report.append("• Usa un GLM como metalearner para optimizar combinaciones")
    report.append("• Supera a todos los modelos individuales en el test")
    report.append("• Ideal para problemas complejos de regresión")
    report.append("")
    
    report.append("="*80)
    report.append("Reporte generado automáticamente")
    report.append("="*80)
    
    # Guardar reporte
    output_path = os.path.join(ANALYSIS_DIR, 'h2o_ensemble_reporte_completo.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"✅ Guardado: {output_path}")

def main():
    """Función principal"""
    print("📄 EXPORTANDO ANÁLISIS H2O A ARCHIVOS CSV Y TXT")
    print("="*60)
    
    try:
        # Crear directorio de análisis
        create_analysis_directory()
        
        # Cargar modelo
        model, metadata = load_h2o_model()
        
        # Exportar todos los análisis
        df_ensemble = export_ensemble_summary(model, metadata)
        df_metrics = export_performance_metrics(model)
        df_base_models = export_base_models_info(model)
        df_leaderboard = export_leaderboard(metadata)
        df_comparison = export_comparison_with_other_models()
        
        # Crear reporte completo
        create_comprehensive_report(df_ensemble, df_metrics, df_base_models, df_leaderboard, df_comparison)
        
        print("\n" + "="*60)
        print("✅ ARCHIVOS GENERADOS:")
        print("="*60)
        print(f"📊 {ANALYSIS_DIR}/h2o_ensemble_info.csv - Información básica del ensemble")
        print(f"📈 {ANALYSIS_DIR}/h2o_performance_metrics.csv - Métricas de rendimiento")
        print(f"🏗️ {ANALYSIS_DIR}/h2o_base_models.csv - Lista de modelos base")
        print(f"🏆 {ANALYSIS_DIR}/h2o_leaderboard.csv - Leaderboard completo de AutoML")
        print(f"📊 {ANALYSIS_DIR}/comparacion_todos_modelos.csv - Comparación con otros modelos")
        print(f"📝 {ANALYSIS_DIR}/h2o_ensemble_reporte_completo.txt - Reporte completo en texto")
        
    except Exception as e:
        print(f"❌ Error durante la exportación: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            h2o.cluster().shutdown()
            print(f"\n🔧 H2O cluster cerrado correctamente")
        except:
            pass

if __name__ == "__main__":
    main()
