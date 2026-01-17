"""
G-SMOTE para Regresión: Implementación basada en Camacho et al. (2022)
https://doi.org/10.1016/j.eswa.2021.116387
"""

import os
import time
import numpy as np
import pandas as pd
import logging
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import euclidean
from typing import Tuple
from scipy.spatial.distance import pdist, cdist, euclidean
import traceback
import time




class RelevanceFunction:
    """Función de relevancia automática usando box-and-whisker (Ribeiro 2011) https://doi.org/10.1007/978-3-540-74976-9_63"""

    def __init__(self, y: np.ndarray, extremes: str = "both"):
        self.y = y
        self.extremes = extremes
        self._compute_relevance()


    def _compute_relevance(self):
        """Calcula valores del box-whisker para interpolación."""
        q1 = np.percentile(self.y, 25)
        q3 = np.percentile(self.y, 75)
        iqr = q3 - q1

        self.median = np.percentile(self.y, 50)
        self.lower_adjacent = q1 - 1.5 * iqr
        self.upper_adjacent = q3 + 1.5 * iqr

    def compute_relevance(self, values: np.ndarray) -> np.ndarray:
        """Relevancia via interpolación cúbica Hermite."""
        relevance = np.zeros_like(values, dtype=float)

        if self.extremes == "both":
            # Región inferior: lower_adjacent → median (relevancia: 1 → 0)
            mask_lower = (values >= self.lower_adjacent) & (values <= self.median)
            if mask_lower.sum() > 0:
                t = (values[mask_lower] - self.lower_adjacent) / (self.median - self.lower_adjacent)
                h00 = 2 * t**3 - 3 * t**2 + 1
                h01 = -2 * t**3 + 3 * t**2
                relevance[mask_lower] = h00 * 1.0 + h01 * 0.0

            # Región superior: median → upper_adjacent (relevancia: 0 → 1)
            mask_upper = (values > self.median) & (values <= self.upper_adjacent)
            if mask_upper.sum() > 0:
                t = (values[mask_upper] - self.median) / (self.upper_adjacent - self.median)
                h00 = 2 * t**3 - 3 * t**2 + 1
                h01 = -2 * t**3 + 3 * t**2
                relevance[mask_upper] = h00 * 0.0 + h01 * 1.0

            # Extremos
            relevance[values < self.lower_adjacent] = 1.0
            relevance[values > self.upper_adjacent] = 1.0

        elif self.extremes == "upper":
            mask = (values >= self.median) & (values <= self.upper_adjacent)
            if mask.sum() > 0:
                t = (values[mask] - self.median) / (self.upper_adjacent - self.median)
                h01 = -2 * t**3 + 3 * t**2
                relevance[mask] = h01
            relevance[values > self.upper_adjacent] = 1.0

        elif self.extremes == "lower":
            mask = (values >= self.lower_adjacent) & (values <= self.median)
            if mask.sum() > 0:
                t = (values[mask] - self.lower_adjacent) / (self.median - self.lower_adjacent)
                h00 = 2 * t**3 - 3 * t**2 + 1
                relevance[mask] = h00
            relevance[values < self.lower_adjacent] = 1.0

        return np.clip(relevance, 0, 1)


class GSMOTERegressor:
    """
    G-SMOTE para Regresión (Camacho et al. 2022) - Versión AJUSTADA.
    Genera registros sintéticos para balancear registros "raros".
    """

    def __init__(
        self,
        relevance_threshold: float = 0.8,
        phi: float = 0.0,
        deformation_factor: float = 0.5,
        selection_strategy: str = "combined",
        k_neighbors: int = 5,
        extremes: str = "both",
        n_synthetic_multiplier: float = 1.0,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        n_synthetic_multiplier : float
            Multiplicador de sintéticas respecto a raros.
            1.0 = generar igual cantidad que raros
            2.0 = generar 2x cantidad de raros
            etc.
        """
        self.relevance_threshold = relevance_threshold
        self.phi = phi
        self.deformation_factor = deformation_factor
        self.selection_strategy = selection_strategy
        self.k_neighbors = k_neighbors
        self.extremes = extremes
        self.n_synthetic_multiplier = n_synthetic_multiplier
        self.random_state = random_state
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()

        self._fit_resample_calls = 0
        if hasattr(self.logger, "propagate"):
            self.logger.propagate = False 

        np.random.seed(random_state)

    def _setup_logging(self):
        """Configurar logging básico."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

            # Handler para archivo
            log_file = "g_smote.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def fit_resample(self, X: pd.DataFrame, y: pd.Series):
        t0 = time.time()
        self._fit_resample_calls += 1
        call_id = self._fit_resample_calls

        # Cabecera del log para esta llamada (útil dentro de CV/learning_curve)
        self.logger.info(f"🔁 G-SMOTE call #{call_id}: inicio | X={X.shape}, y={y.shape} | "
                        f"strategy={self.selection_strategy} | k={self.k_neighbors} | "
                        f"multiplier={self.n_synthetic_multiplier} | thr={self.relevance_threshold}")

        # Copias base
        X_array = X.values
        y_array = y.values

        # 1) Relevancia y máscaras
        rel_func = RelevanceFunction(y_array, extremes=self.extremes)
        relevance = rel_func.compute_relevance(y_array)
        rare_mask = relevance >= self.relevance_threshold
        normal_mask = ~rare_mask

        n_rare = int(rare_mask.sum())
        n_normal = int(normal_mask.sum())
        n_total = int(len(y_array))

        self.logger.info(
            f"📊 Recuento inicial → Raros: {n_rare} ({100 * n_rare / max(1, n_total):.1f}%), "
            f"Normales: {n_normal} ({100 * n_normal / max(1, n_total):.1f}%)"
        )

        if n_rare == 0:
            self.logger.warning("⚠️  Sin muestras raras. No se aplica G-SMOTE.")
            return X, y

        # 2) Selección de columnas numéricas para KNN
        num_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = [c for c in X.columns if c not in num_cols]
        self.logger.info(f"🧮 Cols numéricas={len(num_cols)} | categóricas={len(cat_cols)}")
        if len(num_cols) == 0:
            self.logger.warning("⚠️  No hay columnas numéricas; KNN no es viable. Se retorna sin resampling.")
            return X, y

        # 3) Imputación temporal solo para KNN (no se devuelve)
        X_num = X[num_cols].copy()
        n_nan_before = int(X_num.isna().sum().sum())
        medians = X_num.median()
        X_num_imputed = X_num.fillna(medians)
        n_nan_after = int(X_num_imputed.isna().sum().sum())
        self.logger.info(f"🧰 Imputación temporal (mediana) para KNN: NaNs antes={n_nan_before}, después={n_nan_after}")

        # 4) Ajustar KNN en espacio numérico imputado
        X_num_rare_imputed   = X_num_imputed.values[rare_mask]
        X_num_normal_imputed = X_num_imputed.values[normal_mask]

        # Guardas por si hay clases muy pequeñas
        k_rare   = max(1, min(self.k_neighbors, X_num_rare_imputed.shape[0]))
        k_normal = max(1, min(self.k_neighbors, X_num_normal_imputed.shape[0]))
        self.logger.info(f"🤝 KNN vecinos: raros={k_rare}, normales={k_normal}")

        try:
            nn_rare = NearestNeighbors(n_neighbors=k_rare).fit(X_num_rare_imputed)
            nn_normal = NearestNeighbors(n_neighbors=k_normal).fit(X_num_normal_imputed)
        except Exception as e:
            self.logger.error(f"💥 Error ajustando KNN: {e}. Se retorna sin resampling.")
            return X, y

        # 5) Generación de sintéticos
        n_synthetic = int(n_rare * self.n_synthetic_multiplier)
        self.logger.info(f"⚙️  Sintéticas a generar: {n_synthetic} (multiplier={self.n_synthetic_multiplier})")
        if n_synthetic <= 0:
            self.logger.warning("⚠️  n_synthetic = 0. Retornando datos originales.")
            return X, y

        X_synth_rows = []
        y_synth_vals = []

        rng = np.random.default_rng(self.random_state + call_id)  # micro-variación por llamada
        X_array_rare = X_array[rare_mask]
        X_array_norm = X_array[normal_mask]

        # Contadores de estrategia (para saber de dónde viene el vecino)
        c_from_rare = 0
        c_from_norm = 0

        for s in range(n_synthetic):
            # Instancia rara base
            i = int(rng.integers(0, n_rare))
            xi_num = X_num_rare_imputed[i]
            xi_full = X_array_rare[i]
            yi = y_array[rare_mask][i]

            # Vecino según estrategia
            choose_rare_neighbor = (
                (self.selection_strategy == "combined" and rng.random() < 0.5) or
                (self.selection_strategy == "minority")
            )

            if choose_rare_neighbor:
                _, idxs = nn_rare.kneighbors([xi_num])
                j = int(rng.choice(idxs[0][1:] if len(idxs[0]) > 1 else idxs[0]))
                xj_num = X_num_rare_imputed[j]
                xj_full = X_array_rare[j]
                yj = y_array[rare_mask][j]
                c_from_rare += 1
            else:
                _, idxs = nn_normal.kneighbors([xi_num])
                j = int(rng.choice(idxs[0]))
                xj_num = X_num_normal_imputed[j]
                xj_full = X_array_norm[j]
                yj = y_array[normal_mask][j]
                c_from_norm += 1

            # Parte numérica (hiperesfera) en espacio imputado
            x_syn_num, y_syn = self._sample_hypersphere(xi_num, xj_num, yi, yj)

            # Reconstrucción: numéricas = x_syn_num; categóricas = copiar de xi/xj
            row_dict = {}
            # numéricas
            for k, col in enumerate(num_cols):
                row_dict[col] = x_syn_num[k]
            # categóricas: tomar de xi_full o xj_full (al azar)
            source_full = xi_full if rng.random() < 0.5 else xj_full
            for col in cat_cols:
                col_idx = X.columns.get_loc(col)
                row_dict[col] = source_full[col_idx]

            X_synth_rows.append(row_dict)
            y_synth_vals.append(y_syn)

            # Muestra de depuración ocasional (no en cada iteración)
            if s in (0, n_synthetic - 1):
                self.logger.debug(f"🧪 Ejemplo sintético #{s+1}: y={y_syn:.4f} | "
                                f"num_sampled(first3)={[row_dict[c] for c in list(num_cols)[:3]]}")

        # 6) Devolver con mismas columnas
        X_synth_df = pd.DataFrame(X_synth_rows, columns=X.columns)
        y_synth_sr = pd.Series(y_synth_vals, name=y.name)

        X_res = pd.concat([X, X_synth_df], axis=0, ignore_index=True)
        y_res = pd.concat([y, y_synth_sr], axis=0, ignore_index=True)

        # === 7) Análisis de diversidad de sintéticos ===
        self._analyze_synthetic_diversity(X_num_imputed, X_synth_df, num_cols, n_synthetic, call_id)

        # === 8) Análisis de distribución del target antes/después ===
        self._analyze_target_distribution(y, y_res, call_id)

        # Resumen final
        elapsed = time.time() - t0
        self.logger.info(f"✅ G-SMOTE call #{call_id}: fin en {elapsed:.2f}s | "
                        f"sintéticas={n_synthetic} (minority={c_from_rare}, majority={c_from_norm}) | "
                        f"X_res={X_res.shape}, y_res={y_res.shape}")

        return X_res, y_res

    def _analyze_target_distribution(self, y_original: pd.Series, y_resampled: pd.Series, call_id: int):
        """
        Analiza la distribución del target antes y después del resampling.
        Especial atención a estudiantes con bajas calificaciones.
        """
        try:
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"📊 ANÁLISIS DE DISTRIBUCIÓN DEL TARGET (Call #{call_id})")
            self.logger.info(f"{'='*70}")

            # Bandas de calificación
            bands = {
                'Muy Bajo (0-40)': (0, 40),
                'Bajo (40-60)': (40, 60),
                'Medio (60-70)': (60, 70),
                'Alto (70-80)': (70, 80),
                'Muy Alto (80-100)': (80, 100)
            }

            self.logger.info(f"\n🔹 Distribución ANTES del resampling:")
            y_orig_array = y_original.values
            total_orig = len(y_orig_array)

            for band_name, (min_val, max_val) in bands.items():
                if max_val == 100:
                    mask = (y_orig_array >= min_val) & (y_orig_array <= max_val)
                else:
                    mask = (y_orig_array >= min_val) & (y_orig_array < max_val)
                count = int(mask.sum())
                pct = 100 * count / total_orig if total_orig > 0 else 0
                self.logger.info(f"   {band_name:20s}: {count:5d} ({pct:5.1f}%)")

            self.logger.info(f"\n🔹 Distribución DESPUÉS del resampling:")
            y_res_array = y_resampled.values
            total_res = len(y_res_array)

            for band_name, (min_val, max_val) in bands.items():
                if max_val == 100:
                    mask = (y_res_array >= min_val) & (y_res_array <= max_val)
                else:
                    mask = (y_res_array >= min_val) & (y_res_array < max_val)
                count = int(mask.sum())
                pct = 100 * count / total_res if total_res > 0 else 0
                increment = count - int((y_orig_array >= min_val) & (y_orig_array < (max_val if max_val < 100 else 101)).sum())
                self.logger.info(f"   {band_name:20s}: {count:5d} ({pct:5.1f}%) [+{increment:+5d}]")

            # Análisis específico de bajas calificaciones
            low_grades_mask_orig = y_orig_array < 60
            low_grades_mask_res = y_res_array < 60

            n_low_orig = int(low_grades_mask_orig.sum())
            n_low_res = int(low_grades_mask_res.sum())
            pct_low_orig = 100 * n_low_orig / total_orig if total_orig > 0 else 0
            pct_low_res = 100 * n_low_res / total_res if total_res > 0 else 0

            self.logger.info(f"\n🔹 Análisis de BAJAS CALIFICACIONES (<60):")
            self.logger.info(f"   Antes:  {n_low_orig:5d} ({pct_low_orig:5.1f}%)")
            self.logger.info(f"   Después: {n_low_res:5d} ({pct_low_res:5.1f}%)")
            self.logger.info(f"   Incremento: +{n_low_res - n_low_orig:+5d} ({100*(n_low_res/n_low_orig - 1) if n_low_orig > 0 else 0:+.1f}%)")

            if n_low_orig < 10:
                self.logger.warning(f"   ⚠️  PROBLEMA: Muy pocos estudiantes con bajas calificaciones ({n_low_orig})")
                self.logger.warning(f"   ⚠️  Esto puede causar que los sintéticos sean muy similares entre sí")

            # Estadísticas descriptivas
            self.logger.info(f"\n🔹 Estadísticas descriptivas:")
            self.logger.info(f"   Antes:  Media={y_orig_array.mean():.2f}, Mediana={np.median(y_orig_array):.2f}, "
                           f"Std={y_orig_array.std():.2f}")
            self.logger.info(f"   Después: Media={y_res_array.mean():.2f}, Mediana={np.median(y_res_array):.2f}, "
                           f"Std={y_res_array.std():.2f}")

            # Verificar si la distribución mejoró
            if pct_low_res < pct_low_orig:
                self.logger.warning(f"   ⚠️  PROBLEMA: El porcentaje de bajas calificaciones DISMINUYÓ después del resampling")

            self.logger.info(f"{'='*70}\n")

            # Guardar métricas para diagnóstico
            self._last_distribution_metrics = {
                'n_low_orig': n_low_orig,
                'n_low_res': n_low_res,
                'pct_low_orig': pct_low_orig,
                'pct_low_res': pct_low_res,
                'mean_orig': float(y_orig_array.mean()),
                'mean_res': float(y_res_array.mean()),
                'std_orig': float(y_orig_array.std()),
                'std_res': float(y_res_array.std())
            }

        except Exception as e:
            self.logger.warning(f"⚠️  Error analizando distribución del target: {e}")
            self.logger.debug(traceback.format_exc())

    def _analyze_synthetic_diversity(self, X_original_num: pd.DataFrame, X_synth: pd.DataFrame, 
                                     num_cols: pd.Index, n_synthetic: int, call_id: int):
        """
        Analiza qué tan diversos son los datos sintéticos generados.
        """
        if n_synthetic < 2:
            self.logger.info("📊 Diversidad: Solo 1 sintético generado, análisis omitido.")
            return

        try:
            # Extraer solo columnas numéricas de sintéticos (ya vienen imputadas)
            X_synth_num = X_synth[num_cols].values
            X_orig_num = X_original_num.values

            # Muestrear para eficiencia (si hay muchos datos)
            max_samples = 500
            if len(X_orig_num) > max_samples:
                sample_idx = np.random.choice(len(X_orig_num), max_samples, replace=False)
                X_orig_sample = X_orig_num[sample_idx]
            else:
                X_orig_sample = X_orig_num

            # 1. Distancias entre sintéticos
            if n_synthetic >= 2:
                synth_distances = pdist(X_synth_num, metric='euclidean')
                mean_synth_dist = float(np.mean(synth_distances))
                min_synth_dist = float(np.min(synth_distances))
                max_synth_dist = float(np.max(synth_distances))
                std_synth_dist = float(np.std(synth_distances))
                median_synth_dist = float(np.median(synth_distances))
                p25_synth_dist = float(np.percentile(synth_distances, 25))
                p75_synth_dist = float(np.percentile(synth_distances, 75))
            else:
                mean_synth_dist = min_synth_dist = max_synth_dist = std_synth_dist = median_synth_dist = p25_synth_dist = p75_synth_dist = 0.0

            # 2. Distancias entre originales (muestra)
            orig_distances = pdist(X_orig_sample, metric='euclidean')
            mean_orig_dist = float(np.mean(orig_distances))
            std_orig_dist = float(np.std(orig_distances))
            median_orig_dist = float(np.median(orig_distances))

            # 3. Distancia promedio de cada sintético a su vecino original más cercano
            dist_to_nearest_orig = cdist(X_synth_num, X_orig_num, metric='euclidean').min(axis=1)
            mean_dist_to_orig = float(np.mean(dist_to_nearest_orig))
            min_dist_to_orig = float(np.min(dist_to_nearest_orig))
            max_dist_to_orig = float(np.max(dist_to_nearest_orig))
            median_dist_to_orig = float(np.median(dist_to_nearest_orig))

            # 4. Detectar potenciales duplicados (distancia < umbral)
            threshold_duplicate = 1e-6
            threshold_similar = 0.01  # Umbral para datos muy similares (1% de la distancia media)
            n_near_duplicates = int(np.sum(synth_distances < threshold_duplicate)) if n_synthetic >= 2 else 0
            n_similar = int(np.sum(synth_distances < threshold_similar)) if n_synthetic >= 2 else 0

            # 5. Análisis de varianza por característica
            feature_variance_orig = np.var(X_orig_num, axis=0)
            feature_variance_synth = np.var(X_synth_num, axis=0)
            variance_ratio = feature_variance_synth / (feature_variance_orig + 1e-10)

            # 6. Análisis de solapamiento: cuántos sintéticos están muy cerca de originales
            overlap_threshold = np.percentile(orig_distances, 10)  # Percentil 10 de distancias originales
            n_overlapping = int(np.sum(dist_to_nearest_orig < overlap_threshold))
            pct_overlapping = 100 * n_overlapping / n_synthetic if n_synthetic > 0 else 0

            # Log de resultados
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"📊 ANÁLISIS DE DIVERSIDAD - Sintéticos Generados (Call #{call_id})")
            self.logger.info(f"{'='*70}")
            self.logger.info(f"Registros creados: {n_synthetic}")
            self.logger.info(f"\n🔹 Distancias ENTRE sintéticos:")
            self.logger.info(f"   Media:  {mean_synth_dist:.4f} ± {std_synth_dist:.4f}")
            self.logger.info(f"   Mediana: {median_synth_dist:.4f}")
            self.logger.info(f"   Percentiles: P25={p25_synth_dist:.4f}, P75={p75_synth_dist:.4f}")
            self.logger.info(f"   Rango:  [{min_synth_dist:.4f}, {max_synth_dist:.4f}]")

            self.logger.info(f"\n🔹 Distancias en datos ORIGINALES (referencia):")
            self.logger.info(f"   Media:  {mean_orig_dist:.4f} ± {std_orig_dist:.4f}")
            self.logger.info(f"   Mediana: {median_orig_dist:.4f}")

            self.logger.info(f"\n🔹 Distancia de sintéticos al original más cercano:")
            self.logger.info(f"   Media:  {mean_dist_to_orig:.4f}")
            self.logger.info(f"   Mediana: {median_dist_to_orig:.4f}")
            self.logger.info(f"   Rango:  [{min_dist_to_orig:.4f}, {max_dist_to_orig:.4f}]")

            # Ratio de diversidad
            if mean_orig_dist > 1e-10:
                diversity_ratio = mean_synth_dist / mean_orig_dist
                self.logger.info(f"\n🔹 Ratio de diversidad (synth/orig): {diversity_ratio:.3f}")
                if diversity_ratio < 0.3:
                    self.logger.warning(f"   ⚠️  Los sintéticos están MUY AGRUPADOS (ratio < 0.3)")
                    self.logger.warning(f"   ⚠️  PROBLEMA: Los datos sintéticos son muy similares entre sí")
                elif diversity_ratio > 1.5:
                    self.logger.warning(f"   ⚠️  Los sintéticos están MUY DISPERSOS (ratio > 1.5)")
                else:
                    self.logger.info(f"   ✓ Diversidad apropiada (0.3 < ratio < 1.5)")

            # Duplicados y similares
            if n_near_duplicates > 0:
                pct_duplicates = 100 * n_near_duplicates / (n_synthetic * (n_synthetic - 1) / 2)
                self.logger.warning(f"\n⚠️  Posibles duplicados exactos: {n_near_duplicates} pares "
                                  f"({pct_duplicates:.1f}% de todos los pares)")

            if n_similar > 0:
                pct_similar = 100 * n_similar / (n_synthetic * (n_synthetic - 1) / 2)
                self.logger.warning(f"⚠️  Datos muy similares (<0.01): {n_similar} pares "
                                  f"({pct_similar:.1f}% de todos los pares)")

            if n_near_duplicates == 0 and n_similar == 0:
                self.logger.info(f"\n✓ No se detectaron duplicados ni datos muy similares")

            # Análisis de varianza
            low_variance_features = np.sum(variance_ratio < 0.5)
            high_variance_features = np.sum(variance_ratio > 2.0)
            self.logger.info(f"\n🔹 Análisis de varianza por característica:")
            self.logger.info(f"   Características con varianza baja (<50% original): {low_variance_features}/{len(variance_ratio)}")
            self.logger.info(f"   Características con varianza alta (>200% original): {high_variance_features}/{len(variance_ratio)}")
            if low_variance_features > len(variance_ratio) * 0.3:
                self.logger.warning(f"   ⚠️  PROBLEMA: Muchas características tienen poca variación en sintéticos")

            # Solapamiento con originales
            self.logger.info(f"\n🔹 Solapamiento con datos originales:")
            self.logger.info(f"   Sintéticos muy cerca de originales (<P10 distancia): {n_overlapping} ({pct_overlapping:.1f}%)")
            if pct_overlapping > 50:
                self.logger.warning(f"   ⚠️  PROBLEMA: Más del 50% de sintéticos están muy cerca de originales")
                self.logger.warning(f"   ⚠️  Esto sugiere que los sintéticos no aportan nueva información")

            self.logger.info(f"{'='*70}\n")

            # Guardar métricas para diagnóstico
            self._last_diversity_metrics = {
                'n_synthetic': n_synthetic,
                'mean_synth_dist': mean_synth_dist,
                'mean_orig_dist': mean_orig_dist,
                'diversity_ratio': diversity_ratio if mean_orig_dist > 1e-10 else None,
                'n_duplicates': n_near_duplicates,
                'n_similar': n_similar,
                'pct_overlapping': pct_overlapping,
                'low_variance_features': low_variance_features,
                'mean_dist_to_orig': mean_dist_to_orig
            }

        except Exception as e:
            self.logger.warning(f"⚠️  Error analizando diversidad: {e}")
            self.logger.debug(traceback.format_exc())

    def _sample_hypersphere(self, xi: np.ndarray, xj: np.ndarray, yi: float, yj: float) -> Tuple[np.ndarray, float]:
        """
        Muestrea punto uniforme en hipersfera truncada y deformada.

        Centro: c = xi + φ * (xj - xi)
        Radio: r = ||xj - xi|| * (1 - ψ) / 2
        """

        d = xj - xi
        norm_d = np.linalg.norm(d)

        if norm_d < 1e-10:
            return xi.copy(), yi

        # Centro de la hipersfera
        center = xi + self.phi * d

        # Radio (deformado)
        radius = norm_d * (1 - self.deformation_factor) / 2

        if radius < 1e-10:
            return center.copy(), (yi + yj) / 2

        # Muestrear punto uniforme en bola n-dimensional
        n_features = len(xi)
        z = np.random.randn(n_features)
        z = z / np.linalg.norm(z)

        # Radio uniforme en bola: r^(1/n)
        u = np.random.uniform(0, 1)
        r_uniform = radius * (u ** (1.0 / n_features))

        x_synthetic = center + r_uniform * z

        # Target: interpolación ponderada por distancia
        dist_xi_syn = euclidean(x_synthetic, xi)
        dist_xj_syn = euclidean(x_synthetic, xj)

        total_dist = dist_xi_syn + dist_xj_syn
        if total_dist > 1e-10:
            w_i = dist_xj_syn / total_dist
            w_j = dist_xi_syn / total_dist
        else:
            w_i = 0.5
            w_j = 0.5

        y_synthetic = w_i * yi + w_j * yj

        return x_synthetic, y_synthetic

class GSMOTEBalancedRegressor:
    """
    G-SMOTE balanceado por bandas fijas del target.
    Bandas por defecto:
      - bajo:     1–59
      - básico:   60–79
      - alto:     80–94
      - superior: 95–100

    Estrategia de balanceo:
      Oversampling hacia la banda mayor (todas las bandas alcanzan el tamaño de la banda con más ejemplos).
      Opcionalmente, puedes limitar el total con n_synthetic_multiplier relativo a la cantidad de 'raros'
      (según relevance_threshold), pero no es necesario.
    """

    def __init__(
        self,
        phi: float = 0.0,
        deformation_factor: float = 0.5,
        selection_strategy: str = "combined",  # "minority", "majority", "combined"
        k_neighbors: int = 5,
        relevance_threshold: float = 0.8,      # solo para el CAP opcional
        n_synthetic_multiplier: float | None = None,
        random_state: int = 42,
    ):
        # Bandas fijas internas
        self.bands = {
            "bajo":     {"min": 1,  "max": 59},
            "basico":   {"min": 60, "max": 79},
            "alto":     {"min": 80, "max": 94},
            "superior": {"min": 95, "max": 100},
        }

        self.phi = phi
        self.deformation_factor = deformation_factor
        self.selection_strategy = selection_strategy
        self.k_neighbors = k_neighbors
        self.relevance_threshold = relevance_threshold
        self.n_synthetic_multiplier = n_synthetic_multiplier
        self.random_state = random_state

        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()
        self._fit_resample_calls = 0
        if hasattr(self.logger, "propagate"):
            self.logger.propagate = False

        np.random.seed(random_state)

    def _setup_logging(self):
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(fmt)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

            file_handler = logging.FileHandler("g_smote.log", encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(fmt)
            self.logger.addHandler(file_handler)

    # --- Utilidades ---
    def _band_masks_counts(self, y: np.ndarray):
        masks, counts = {}, {}
        for name, spec in self.bands.items():
            m = (y >= spec["min"]) & (y <= spec["max"])
            masks[name] = m
            counts[name] = int(m.sum())
        return masks, counts

    def _compute_relevance(self, y: np.ndarray) -> np.ndarray:
        rel = RelevanceFunction(y, extremes="both").compute_relevance(y)
        return rel

    # --- Métodos principales ---
    def fit_resample(self, X: pd.DataFrame, y: pd.Series):
        t0 = time.time()
        self._fit_resample_calls += 1
        call_id = self._fit_resample_calls

        self.logger.info(f"🔁 GSMOTEBalanced call #{call_id}: inicio | X={X.shape}, y={y.shape} | "
                         f"strategy={self.selection_strategy} | k={self.k_neighbors}")

        X_array = X.values
        y_array = y.values
        n_total = len(y_array)

        # Distribución actual por banda
        band_masks, band_counts = self._band_masks_counts(y_array)
        max_count = max(band_counts.values()) if band_counts else 0

        self.logger.info("📊 Conteo inicial por bandas:")
        for b in self.bands:
            self.logger.info(f"   - {b:9s}: {band_counts[b]}")

        # Cuántas sintéticas necesita cada banda para alcanzar a la mayor
        band_synth = {b: max(0, max_count - band_counts[b]) for b in self.bands}
        total_needed = sum(band_synth.values())

        # CAP opcional por multiplier (relativo a "raros" por relevance)
        if self.n_synthetic_multiplier is not None:
            relevance = self._compute_relevance(y_array)
            rare_mask = relevance >= self.relevance_threshold
            n_rare = int(rare_mask.sum())
            cap_total = int(n_rare * self.n_synthetic_multiplier)
            if total_needed > cap_total and total_needed > 0:
                factor = cap_total / total_needed
                band_synth = {b: int(round(v * factor)) for b, v in band_synth.items()}
                # al menos 1 si había déficit
                for b in band_synth:
                    if band_synth[b] == 0 and max_count > band_counts[b]:
                        band_synth[b] = 1
                total_needed = sum(band_synth.values())

        if total_needed == 0:
            self.logger.info("✅ Ya está balanceado por bandas (no se generan sintéticas).")
            return X, y

        self.logger.info("🧮 Sintéticas a crear por banda:")
        for b in self.bands:
            self.logger.info(f"   - {b:9s}: +{band_synth[b]} (actual={band_counts[b]}, objetivo={max_count})")
        self.logger.info(f"Total sintéticas: {total_needed}")

        # KNN solo en numéricas (imputación temporal por mediana)
        num_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = [c for c in X.columns if c not in num_cols]
        if len(num_cols) == 0:
            self.logger.warning("⚠️ Sin columnas numéricas; no se puede muestrear en hiperesfera. Retorno original.")
            return X, y

        X_num = X[num_cols].copy()
        n_nan_before = int(X_num.isna().sum().sum())
        medians = X_num.median()
        X_num_imp = X_num.fillna(medians)
        n_nan_after = int(X_num_imp.isna().sum().sum())
        self.logger.info(f"🧰 Imputación temporal para KNN: NaNs antes={n_nan_before}, después={n_nan_after}")

        X_num_vals = X_num_imp.values

        # Índices por banda
        band_indices = {b: np.where(band_masks[b])[0] for b in self.bands}

        # Preparar KNN global por banda (solo si hay >=1 punto)
        knn_by_band = {}
        for b, idxs in band_indices.items():
            k_eff = max(1, min(self.k_neighbors, len(idxs)))
            if len(idxs) > 0:
                knn_by_band[b] = NearestNeighbors(n_neighbors=k_eff).fit(X_num_vals[idxs])
            else:
                knn_by_band[b] = None

        rng = np.random.default_rng(self.random_state + call_id)
        X_synth_rows, y_synth_vals = [], []

        # Generación por banda
        for b, n_add in band_synth.items():
            if n_add <= 0 or len(band_indices[b]) == 0:
                continue

            idx_band = band_indices[b]
            knn_band = knn_by_band[b]

            for _ in range(n_add):
                # Ancla: un punto cualquiera de la banda
                i_global = int(rng.choice(idx_band))
                xi_num = X_num_vals[i_global]
                xi_full = X_array[i_global]
                yi = y_array[i_global]

                # Elegir vecino: dentro de la misma banda si es posible
                if knn_band is not None and len(idx_band) > 1:
                    dists, neighs = knn_band.kneighbors([xi_num])
                    # neighs están en índices locales; elegir uno (evitar el mismo si es posible)
                    local_choices = neighs[0]
                    if len(local_choices) > 1:
                        local_choices = local_choices[1:]
                    j_local = int(rng.choice(local_choices))
                    j_global = int(idx_band[j_local])
                else:
                    # Fallback: otro punto de la banda (o el mismo si no hay más)
                    j_global = int(rng.choice(idx_band))

                xj_num = X_num_vals[j_global]
                xj_full = X_array[j_global]
                yj = y_array[j_global]

                # Muestreo en hiperesfera (numéricas)
                x_syn_num, y_syn = self._sample_hypersphere(xi_num, xj_num, yi, yj)

                # Reconstrucción fila completa
                row = {}
                for k, col in enumerate(num_cols):
                    row[col] = x_syn_num[k]
                src = xi_full if rng.random() < 0.5 else xj_full
                for col in cat_cols:
                    col_idx = X.columns.get_loc(col)
                    row[col] = src[col_idx]

                X_synth_rows.append(row)
                y_synth_vals.append(y_syn)

        # Ensamble final
        X_synth_df = pd.DataFrame(X_synth_rows, columns=X.columns)
        y_synth_sr = pd.Series(y_synth_vals, name=y.name)

        X_res = pd.concat([X, X_synth_df], axis=0, ignore_index=True)
        y_res = pd.concat([y, y_synth_sr], axis=0, ignore_index=True)

        # === Análisis de diversidad de sintéticos ===
        if len(X_synth_rows) > 0:
            self._analyze_synthetic_diversity(X_num_imp, X_synth_df, num_cols, len(X_synth_rows), call_id)

        elapsed = time.time() - t0
        self.logger.info(f"✅ GSMOTEBalanced call #{call_id}: fin en {elapsed:.2f}s | "
                         f"sintéticas={len(X_synth_rows)} | X_res={X_res.shape}, y_res={y_res.shape}")

        # Distribución final por banda
        _, final_counts = self._band_masks_counts(y_res.values)
        self.logger.info("📈 Distribución FINAL por bandas:")
        for b in self.bands:
            self.logger.info(f"   - {b:9s}: {final_counts[b]}")

        return X_res, y_res

    def _analyze_synthetic_diversity(self, X_original_num: pd.DataFrame, X_synth: pd.DataFrame, 
                                     num_cols: pd.Index, n_synthetic: int, call_id: int):
        """
        Analiza qué tan diversos son los datos sintéticos generados.
        """
        if n_synthetic < 2:
            self.logger.info("📊 Diversidad: Solo 1 sintético generado, análisis omitido.")
            return

        try:
            # Extraer solo columnas numéricas de sintéticos (ya vienen imputadas)
            X_synth_num = X_synth[num_cols].values
            X_orig_num = X_original_num.values

            # Muestrear para eficiencia (si hay muchos datos)
            max_samples = 500
            if len(X_orig_num) > max_samples:
                sample_idx = np.random.choice(len(X_orig_num), max_samples, replace=False)
                X_orig_sample = X_orig_num[sample_idx]
            else:
                X_orig_sample = X_orig_num

            # 1. Distancias entre sintéticos
            if n_synthetic >= 2:
                synth_distances = pdist(X_synth_num, metric='euclidean')
                mean_synth_dist = float(np.mean(synth_distances))
                min_synth_dist = float(np.min(synth_distances))
                max_synth_dist = float(np.max(synth_distances))
                std_synth_dist = float(np.std(synth_distances))
                median_synth_dist = float(np.median(synth_distances))
                p25_synth_dist = float(np.percentile(synth_distances, 25))
                p75_synth_dist = float(np.percentile(synth_distances, 75))
            else:
                mean_synth_dist = min_synth_dist = max_synth_dist = std_synth_dist = median_synth_dist = p25_synth_dist = p75_synth_dist = 0.0

            # 2. Distancias entre originales (muestra)
            orig_distances = pdist(X_orig_sample, metric='euclidean')
            mean_orig_dist = float(np.mean(orig_distances))
            std_orig_dist = float(np.std(orig_distances))
            median_orig_dist = float(np.median(orig_distances))

            # 3. Distancia promedio de cada sintético a su vecino original más cercano
            dist_to_nearest_orig = cdist(X_synth_num, X_orig_num, metric='euclidean').min(axis=1)
            mean_dist_to_orig = float(np.mean(dist_to_nearest_orig))
            min_dist_to_orig = float(np.min(dist_to_nearest_orig))
            max_dist_to_orig = float(np.max(dist_to_nearest_orig))
            median_dist_to_orig = float(np.median(dist_to_nearest_orig))

            # 4. Detectar potenciales duplicados (distancia < umbral)
            threshold_duplicate = 1e-6
            threshold_similar = 0.01  # Umbral para datos muy similares (1% de la distancia media)
            n_near_duplicates = int(np.sum(synth_distances < threshold_duplicate)) if n_synthetic >= 2 else 0
            n_similar = int(np.sum(synth_distances < threshold_similar)) if n_synthetic >= 2 else 0

            # 5. Análisis de varianza por característica
            feature_variance_orig = np.var(X_orig_num, axis=0)
            feature_variance_synth = np.var(X_synth_num, axis=0)
            variance_ratio = feature_variance_synth / (feature_variance_orig + 1e-10)

            # 6. Análisis de solapamiento: cuántos sintéticos están muy cerca de originales
            overlap_threshold = np.percentile(orig_distances, 10)  # Percentil 10 de distancias originales
            n_overlapping = int(np.sum(dist_to_nearest_orig < overlap_threshold))
            pct_overlapping = 100 * n_overlapping / n_synthetic if n_synthetic > 0 else 0

            # Log de resultados
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"📊 ANÁLISIS DE DIVERSIDAD - Sintéticos Generados (Call #{call_id})")
            self.logger.info(f"{'='*70}")
            self.logger.info(f"Registros creados: {n_synthetic}")
            self.logger.info(f"\n🔹 Distancias ENTRE sintéticos:")
            self.logger.info(f"   Media:  {mean_synth_dist:.4f} ± {std_synth_dist:.4f}")
            self.logger.info(f"   Mediana: {median_synth_dist:.4f}")
            self.logger.info(f"   Percentiles: P25={p25_synth_dist:.4f}, P75={p75_synth_dist:.4f}")
            self.logger.info(f"   Rango:  [{min_synth_dist:.4f}, {max_synth_dist:.4f}]")

            self.logger.info(f"\n🔹 Distancias en datos ORIGINALES (referencia):")
            self.logger.info(f"   Media:  {mean_orig_dist:.4f} ± {std_orig_dist:.4f}")
            self.logger.info(f"   Mediana: {median_orig_dist:.4f}")

            self.logger.info(f"\n🔹 Distancia de sintéticos al original más cercano:")
            self.logger.info(f"   Media:  {mean_dist_to_orig:.4f}")
            self.logger.info(f"   Mediana: {median_dist_to_orig:.4f}")
            self.logger.info(f"   Rango:  [{min_dist_to_orig:.4f}, {max_dist_to_orig:.4f}]")

            # Ratio de diversidad
            if mean_orig_dist > 1e-10:
                diversity_ratio = mean_synth_dist / mean_orig_dist
                self.logger.info(f"\n🔹 Ratio de diversidad (synth/orig): {diversity_ratio:.3f}")
                if diversity_ratio < 0.3:
                    self.logger.warning(f"   ⚠️  Los sintéticos están MUY AGRUPADOS (ratio < 0.3)")
                    self.logger.warning(f"   ⚠️  PROBLEMA: Los datos sintéticos son muy similares entre sí")
                elif diversity_ratio > 1.5:
                    self.logger.warning(f"   ⚠️  Los sintéticos están MUY DISPERSOS (ratio > 1.5)")
                else:
                    self.logger.info(f"   ✓ Diversidad apropiada (0.3 < ratio < 1.5)")

            # Duplicados y similares
            if n_near_duplicates > 0:
                pct_duplicates = 100 * n_near_duplicates / (n_synthetic * (n_synthetic - 1) / 2)
                self.logger.warning(f"\n⚠️  Posibles duplicados exactos: {n_near_duplicates} pares "
                                  f"({pct_duplicates:.1f}% de todos los pares)")

            if n_similar > 0:
                pct_similar = 100 * n_similar / (n_synthetic * (n_synthetic - 1) / 2)
                self.logger.warning(f"⚠️  Datos muy similares (<0.01): {n_similar} pares "
                                  f"({pct_similar:.1f}% de todos los pares)")

            if n_near_duplicates == 0 and n_similar == 0:
                self.logger.info(f"\n✓ No se detectaron duplicados ni datos muy similares")

            # Análisis de varianza
            low_variance_features = np.sum(variance_ratio < 0.5)
            high_variance_features = np.sum(variance_ratio > 2.0)
            self.logger.info(f"\n🔹 Análisis de varianza por característica:")
            self.logger.info(f"   Características con varianza baja (<50% original): {low_variance_features}/{len(variance_ratio)}")
            self.logger.info(f"   Características con varianza alta (>200% original): {high_variance_features}/{len(variance_ratio)}")
            if low_variance_features > len(variance_ratio) * 0.3:
                self.logger.warning(f"   ⚠️  PROBLEMA: Muchas características tienen poca variación en sintéticos")

            # Solapamiento con originales
            self.logger.info(f"\n🔹 Solapamiento con datos originales:")
            self.logger.info(f"   Sintéticos muy cerca de originales (<P10 distancia): {n_overlapping} ({pct_overlapping:.1f}%)")
            if pct_overlapping > 50:
                self.logger.warning(f"   ⚠️  PROBLEMA: Más del 50% de sintéticos están muy cerca de originales")
                self.logger.warning(f"   ⚠️  Esto sugiere que los sintéticos no aportan nueva información")

            self.logger.info(f"{'='*70}\n")

            # Guardar métricas para diagnóstico
            self._last_diversity_metrics = {
                'n_synthetic': n_synthetic,
                'mean_synth_dist': mean_synth_dist,
                'mean_orig_dist': mean_orig_dist,
                'diversity_ratio': diversity_ratio if mean_orig_dist > 1e-10 else None,
                'n_duplicates': n_near_duplicates,
                'n_similar': n_similar,
                'pct_overlapping': pct_overlapping,
                'low_variance_features': low_variance_features,
                'mean_dist_to_orig': mean_dist_to_orig
            }

        except Exception as e:
            self.logger.warning(f"⚠️  Error analizando diversidad: {e}")
            self.logger.debug(traceback.format_exc())

    def _sample_hypersphere(self, xi: np.ndarray, xj: np.ndarray, yi: float, yj: float) -> Tuple[np.ndarray, float]:
        d = xj - xi
        norm_d = np.linalg.norm(d)
        if norm_d < 1e-10:
            return xi.copy(), yi

        center = xi + self.phi * d
        radius = norm_d * (1 - self.deformation_factor) / 2
        if radius < 1e-10:
            return center.copy(), (yi + yj) / 2

        n_features = len(xi)
        z = np.random.randn(n_features)
        z = z / np.linalg.norm(z)

        u = np.random.uniform(0, 1)
        r_uniform = radius * (u ** (1.0 / n_features))

        x_syn = center + r_uniform * z

        d_i = euclidean(x_syn, xi)
        d_j = euclidean(x_syn, xj)
        tot = d_i + d_j
        wi = d_j / tot if tot > 1e-10 else 0.5
        wj = d_i / tot if tot > 1e-10 else 0.5
        y_syn = wi * yi + wj * yj
        return x_syn, y_syn