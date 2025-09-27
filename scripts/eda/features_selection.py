"""Script para selección de características usando XGBoost y análisis SHAP."""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
import warnings
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Configuración de warnings y logging
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)

# Silenciar mensajes de debug adicionales
import matplotlib
matplotlib.set_loglevel("WARNING")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.preprocessing.encode_categorical_values import CategoricalEncoder

class XGBoostFeatureSelector:
    """Selector de características usando XGBoost y análisis SHAP."""

    excluded_features = [
        'sede', # No es relevante
        'year', # No es relevante
        'documento_identificación', # No es relevante
        'moodle_user_id', # No es relevante
        'edukrea_user_id', # No es relevante
        'fecha_nacimiento', # Ya esta representada en edad_estudiante
        'id_grado', # No es relevante
        'id_asignatura', # No es relevante
        'id_docente', # No es relevante
        'id_most_late_opened_module', # No es relevante
        'id_least_opened_module', # No es relevante
        'id_least_viewed_module', # No es relevante
        'years_experience_ficc', # Ya esta representada en teacher_experiencia_nivel_ficc
        'years_experience_total', # Ya esta representada en teacher_experiencia_nivel
        'nivel', # Variable objetivo categoría
        'resultado', # Variable objetivo numérica
        'nota_final', # Variable objetivo numérica
        'cog', # Colinealidad con nota_final
        'proc', # Colinealidad con nota_final
        'act', # Colinealidad con nota_final
        'axi', # Colinealidad con nota_final
        'valoración_emocional', # Es una evaluación en lenguaje natural
        'teacher_experiencia_ficc_percentil', # Ya esta representada en teacher_experiencia_nivel_ficc
        'teacher_experiencia_total_percentil', # Ya esta representada en teacher_experiencia_nivel
        'teacher_nivel_educativo_percentil', # Ya esta representada en teacher_experiencia_nivel
        'teacher_nivel_educativo_num', # Ya esta representada en teacher_experiencia_nivel
    ]

    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.level_mapping = {"Bajo": 0, "Básico": 1, "Alto": 2, "Superior": 3}

    def get_features(self, df):
        """Obtiene las características válidas del dataset."""
        return [col for col in df.columns if col not in self.excluded_features]

    def cargar_datos(self):
        """Carga los datos desde el archivo CSV."""
        df = pd.read_csv('data/interim/full_short_dataset_moodle.csv')
        print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df

    def preprocesar_datos(self, df):
        """Preprocesa los datos aplicando codificación categórica."""
        encoder = CategoricalEncoder()
        df_encoded, features = encoder.encode_categorical_variables(df)
        print(f"Datos preprocesados: {len(features)} variables categóricas codificadas")
        return df_encoded

    def entrenar_xgboost(self, X, y):
        """Entrena un modelo XGBoost."""
        # Codificar variable objetivo
        if y.dtype == 'object':
            y_encoded = y.map(self.level_mapping)
            le_target = self.level_mapping
        else:
            y_encoded = y
            le_target = None

        # Detectar variables categóricas
        categorical_features = []
        for col in X.columns:
            if (X[col].dtype == 'object' or X[col].dtype.name == 'category' or 
                (X[col].dtype in ['int64', 'float64'] and X[col].nunique() <= 10 and X[col].min() >= 0)):
                categorical_features.append(col)

        # Convertir a category para XGBoost
        X_processed = X.copy()
        for col in categorical_features:
            X_processed[col] = X_processed[col].astype('category')

        print(f"Variables categóricas: {len(categorical_features)}")

        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Entrenar modelo
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'enable_categorical': True,
            'tree_method': 'hist'
        }

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train)

        # Predicciones
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"📈 Precisión: {accuracy:.4f}")

        # Convertir etiquetas de vuelta
        if le_target:
            inv_map = {v: k for k, v in le_target.items()}
            y_test_original = y_test.map(inv_map)
            y_pred_original = pd.Series(y_pred).map(inv_map)
        else:
            y_test_original = y_test
            y_pred_original = y_pred

        return {
            'model': self.model,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_test_original': y_test_original,
            'y_pred_original': y_pred_original,
            'accuracy': accuracy,
            'feature_names': X_processed.columns.tolist()
        }

    def visualizar_resultados(self, results, top_n=30):
        """Visualiza los resultados del modelo."""
        model = results['model']
        feature_names = results['feature_names']
        y_test = results['y_test']
        y_pred = results['y_pred']

        # Importancia de características
        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        self.feature_importance = importance_df

        # Crear visualizaciones
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Análisis XGBoost', fontsize=16)

        # Top características
        top_features = importance_df.head(top_n)
        axes[0, 0].barh(range(len(top_features)), top_features['importance'])
        axes[0, 0].set_yticks(range(len(top_features)))
        axes[0, 0].set_yticklabels(top_features['feature'], fontsize=9)
        axes[0, 0].set_title(f'Top {top_n} Características')
        axes[0, 0].invert_yaxis()

        # Distribución de importancias
        axes[0, 1].hist(importance, bins=30, alpha=0.7)
        axes[0, 1].set_title('Distribución de Importancias')
        axes[0, 1].axvline(importance.mean(), color='red', linestyle='--')

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)

        # Etiquetas de clase más claras
        class_names = ['Bajo', 'Básico', 'Alto', 'Superior']

        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   cbar_kws={'label': 'Cantidad'},
                   xticklabels=class_names[:len(np.unique(y_test))],
                   yticklabels=class_names[:len(np.unique(y_test))],
                   ax=axes[1, 0])
        axes[1, 0].set_xlabel('Predicción', fontsize=12)
        axes[1, 0].set_ylabel('Valor Real', fontsize=12)
        axes[1, 0].set_title('Matriz de Confusión', fontsize=14)

        # Importancia acumulada
        importance_sorted = np.sort(importance)[::-1]
        cumulative = np.cumsum(importance_sorted)
        axes[1, 1].plot(range(1, len(cumulative) + 1), cumulative)
        axes[1, 1].axhline(y=0.8, color='red', linestyle='--', label='80%')
        axes[1, 1].set_title('Importancia Acumulada')
        axes[1, 1].legend()

        # Guardar cada gráfico por separado

        # 1. Top características
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        top_features = importance_df.head(top_n)
        ax1.barh(range(len(top_features)), top_features['importance'], color='steelblue')
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'], fontsize=9)
        ax1.set_xlabel('Importancia')
        ax1.set_title(f'Top {top_n} Características Más Importantes')
        ax1.invert_yaxis()
        plt.tight_layout()
        plt.savefig('reports/feature_importance_moodle/top_features.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Distribución de importancias
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.hist(importance, bins=30, color='lightcoral', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Importancia')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución de Importancias')
        ax2.axvline(importance.mean(), color='red', linestyle='--', 
                   label=f'Media: {importance.mean():.4f}')
        ax2.legend()
        plt.tight_layout()
        plt.savefig('reports/feature_importance_moodle/importance_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Matriz de confusión
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        class_names = ['Bajo', 'Básico', 'Alto', 'Superior']

        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   cbar_kws={'label': 'Cantidad'},
                   xticklabels=class_names[:len(np.unique(y_test))],
                   yticklabels=class_names[:len(np.unique(y_test))],
                   ax=ax3)
        ax3.set_xlabel('Predicción', fontsize=12)
        ax3.set_ylabel('Valor Real', fontsize=12)
        ax3.set_title('Matriz de Confusión', fontsize=14)
        plt.tight_layout()
        plt.savefig('reports/feature_importance_moodle/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Importancia acumulada
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        importance_sorted = np.sort(importance)[::-1]
        cumulative = np.cumsum(importance_sorted)
        ax4.plot(range(1, len(cumulative) + 1), cumulative, 'b-', linewidth=2)
        ax4.axhline(y=0.8, color='red', linestyle='--', label='80% de importancia')
        ax4.axhline(y=0.9, color='orange', linestyle='--', label='90% de importancia')
        ax4.set_xlabel('Número de Características')
        ax4.set_ylabel('Importancia Acumulada')
        ax4.set_title('Importancia Acumulada de Características')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('reports/feature_importance_moodle/cumulative_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Gráficos XGBoost guardados por separado:")
        print("  - Top características: top_features.png")
        print("  - Distribución: importance_distribution.png") 
        print("  - Matriz confusión: confusion_matrix.png")
        print("  - Importancia acumulada: cumulative_importance.png")

        return importance_df

    def visualizar_shap(self, X_sample):
        """Genera análisis SHAP."""
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        feature_names = X_sample.columns.tolist()

        # Plot de barras
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)

        # Solo personalizar las etiquetas de la leyenda existente si existe
        ax = plt.gca()
        legend = ax.get_legend()
        if legend:
            labels = ['Bajo (0)', 'Básico (1)', 'Alto (2)', 'Superior (3)']
            for i, text in enumerate(legend.get_texts()):
                if i < len(labels):
                    text.set_text(labels[i])
            legend.set_title('Niveles de Rendimiento')

        plt.tight_layout()
        plt.savefig('reports/feature_importance_moodle/shap_bar_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Gráficos SHAP guardados con leyendas personalizadas")
        return shap_values

    def analizar(self, df_processed, feature_columns):
        """Ejecuta el análisis completo."""
        print("Distribución de 'nivel':")
        print(df_processed['nivel'].value_counts().sort_index())

        X = df_processed[feature_columns].copy()
        y = df_processed['nivel'].copy()
        print(f"Datos: X{X.shape}, y{y.shape}")

        # Entrenar modelo
        results = self.entrenar_xgboost(X, y)

        # Visualizar resultados
        importance_df = self.visualizar_resultados(results)

        # Análisis SHAP
        X_sample = pd.DataFrame(results['X_test'], columns=results['feature_names']).sample(
            n=min(200, len(results['X_test'])), random_state=42
        )
        self.visualizar_shap(X_sample)

        # Guardar importancia
        importance_df.to_csv('reports/feature_importance_moodle/feature_importance.csv', index=False)
        print("Resultados guardados")

        return {
            'model': results['model'],
            'accuracy': results['accuracy'],
            'feature_importance': importance_df,
            'classification_report': classification_report(
                results['y_test_original'], 
                results['y_pred_original']
            ),
            'datos_totales': df_processed.shape[0]
        }

    def main(self):
        """Ejecuta el pipeline completo."""
        print("Iniciando análisis de características")

        # Cargar y preprocesar datos
        df = self.cargar_datos()
        df_processed = self.preprocesar_datos(df)

        # Obtener características
        features = self.get_features(df_processed)
        print(f"Características seleccionadas: {len(features)}")

        # Crear directorio si no existe
        os.makedirs('reports/feature_importance_moodle', exist_ok=True)

        # Analizar
        resultados = self.analizar(df_processed, features)

        print("\n Análisis completado")
        print(f"Precisión final: {resultados['accuracy']:.4f}")
        print(f"Datos analizados: {resultados['datos_totales']}")

        return resultados


if __name__ == "__main__":
    selector = XGBoostFeatureSelector()
    selector.main()