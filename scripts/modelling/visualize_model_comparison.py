import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd


# Configuración de estilo
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_results():
    results_file = Path("models/test_evaluation_results.csv")
    if not results_file.exists():
        raise FileNotFoundError(f"El archivo {results_file} no existe")

    df = pd.read_csv(results_file)
    print(f"✅ Resultados cargados: {len(df)} modelos")

    summary_dir = Path("models/summary")
    summary_dir.mkdir(exist_ok=True)
    print(f"✅ Directorio creado: {summary_dir}")

    return df

def create_metrics_comparison(df):
    """Gráfico de barras comparando todas las métricas"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 8))

    # Crear nombres cortos para las etiquetas del eje X
    short_names = []
    for modelo in df['Modelo']:
        if 'Stacked Ensemble' in modelo:
            short_names.append('Stacked\nEnsemble')
        elif 'H2O' in modelo:
            short_names.append('Stacked\nEnsemble')
        elif 'RandomForest' in modelo:
            short_names.append('Random\nForest')
        elif 'ElasticNet' in modelo:
            short_names.append('Elastic\nNet')
        elif 'LinearRegression' in modelo:
            short_names.append('Linear\nRegression')
        elif 'CatBoost' in modelo:
            short_names.append('CatBoost')
        else:
            # Para nombres muy largos, tomar las primeras palabras
            words = modelo.split()
            if len(words) > 2:
                short_names.append('\n'.join(words[:2]))
            else:
                short_names.append(modelo)

    # Colores para cada modelo
    colors = plt.cm.Set3(np.linspace(0, 1, len(df)))

    # RMSE (menor es mejor)
    axes[0].bar(short_names, df['RMSE'], color=colors)
    axes[0].set_title('RMSE por Modelo\n(Menor es Mejor)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('RMSE')
    axes[0].tick_params(axis='x', rotation=0, labelsize=10)

    # Agregar valores en las barras
    for i, v in enumerate(df['RMSE']):
        axes[0].text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

    # MAE (menor es mejor)
    axes[1].bar(short_names, df['MAE'], color=colors)
    axes[1].set_title('MAE por Modelo\n(Menor es Mejor)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('MAE')
    axes[1].tick_params(axis='x', rotation=0, labelsize=10)

    # Agregar valores en las barras
    for i, v in enumerate(df['MAE']):
        axes[1].text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

    # R² (mayor es mejor)
    axes[2].bar(short_names, df['R²'], color=colors)
    axes[2].set_title('R² por Modelo\n(Mayor es Mejor)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('R²')
    axes[2].tick_params(axis='x', rotation=0, labelsize=10)

    # Agregar valores en las barras
    for i, v in enumerate(df['R²']):
        axes[2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
    # R² (mayor es mejor)
    axes[3].bar(short_names, df['Weighted_MAE'], color=colors)
    axes[3].set_title('Weighted MAE por Modelo\n(Menor es Mejor)', fontsize=14, fontweight='bold')
    axes[3].set_ylabel('Weighted MAE')
    axes[3].tick_params(axis='x', rotation=0, labelsize=10)

    # Agregar valores en las barras
    for i, v in enumerate(df['Weighted_MAE']):
        axes[3].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('models/summary/comparacion_metricas.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico guardado: models/summary/comparacion_metricas.png")


def create_summary_table(df):
    """Crear tabla visual de resumen"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    # Ordenar por R² (mejor métrica general)
    df_sorted = df.sort_values('R²', ascending=False).reset_index(drop=True)

    # Crear nombres más cortos para la tabla
    def shorten_model_name(name):
        if 'Stacked Ensemble' in name:
            return 'Stacked Ensemble (H2O)'
        elif 'RandomForest' in name:
            return 'Random Forest'
        elif 'ElasticNet' in name:
            return 'Elastic Net'
        elif 'LinearRegression' in name:
            return 'Linear Regression'
        elif 'CatBoost' in name:
            return 'CatBoost'
        else:
            # Para otros nombres largos, limitar a 25 caracteres
            return name[:25] + '...' if len(name) > 25 else name

    # Crear tabla con formato
    table_data = []
    for i, row in df_sorted.iterrows():
        table_data.append([
            f"{i+1}°",  # Posición
            shorten_model_name(row['Modelo']),
            f"{row['RMSE']:.3f}",
            f"{row['MAE']:.3f}",
            f"{row['R²']:.3f}"
        ])

    # Crear tabla
    table = ax.table(cellText=table_data,
                    colLabels=['Ranking', 'Modelo', 'RMSE ↓', 'MAE ↓', 'R² ↑'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])

    # Estilizar tabla
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2.2)

    # Ajustar ancho de columnas
    cellDict = table.get_celld()
    for i in range(len(df_sorted) + 1):  # +1 para incluir header
        cellDict[(i, 0)].set_width(0.12)  # Ranking - más estrecho
        cellDict[(i, 1)].set_width(0.35)  # Modelo - más ancho
        cellDict[(i, 2)].set_width(0.18)  # RMSE
        cellDict[(i, 3)].set_width(0.18)  # MAE
        cellDict[(i, 4)].set_width(0.17)  # R²

    # Colorear header
    for i in range(5):
        table[(0, i)].set_facecolor('#9F5C9A')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Colorear filas alternadas
    for i in range(1, len(df_sorted) + 1):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(5):
            table[(i, j)].set_facecolor(color)

    # Resaltar el mejor modelo
    for j in range(5):
        table[(1, j)].set_facecolor('#57BE13')  # Verde para el mejor
        table[(1, j)].set_text_props(weight='bold')

    plt.title('Resumen de Resultados - Evaluación de Modelos\n(Ordenado por R²)', 
              fontsize=16, fontweight='bold', pad=20)

    plt.savefig('models/summary/tabla_resumen.png', dpi=300, bbox_inches='tight')
    print("✅ Tabla guardada: models/summary/tabla_resumen.png")


def main():
    """Función principal"""
    print("🎨 Generando gráficas de comparación de modelos...")
    print("=" * 60)

    # Cargar datos
    df = load_results()
    if df is None:
        return

    print(f"\n📊 Modelos evaluados:")
    for i, modelo in enumerate(df['Modelo'], 1):
        print(f"  {i}. {modelo}")

    print(f"\n📈 Generando gráficas...")

    # Crear todas las visualizaciones
    create_metrics_comparison(df)
    create_summary_table(df)

    print("\n" + "=" * 60)
    print("✅ ¡Todas las gráficas generadas exitosamente!")
    print("\n📁 Archivos generados:")
    print("  • models/summary/comparacion_metricas.png - Comparación de métricas")
    print("  • models/summary/tabla_resumen.png - Tabla de resumen")

    # Mostrar el mejor modelo
    best_model = df.loc[df['R²'].idxmax()]
    print(f"\n🏆 Mejor modelo: {best_model['Modelo']}")
    print(f"   • RMSE: {best_model['RMSE']:.3f}")
    print(f"   • MAE: {best_model['MAE']:.3f}")
    print(f"   • R²: {best_model['R²']:.3f}")

if __name__ == "__main__":
    main()
