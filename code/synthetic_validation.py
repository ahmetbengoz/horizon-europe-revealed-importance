"""
synthetic_validation.py

Synthetic noise sensitivity experiment for Supplementary Figure S1:
Effect of noise injection on synthetic validation performance.

This script:
1) Generates synthetic ground-truth importance vectors (Dirichlet).
2) Creates context features (Year, Evaluator, Scenario) similar to the main paper setup.
3) Trains a multi-output regressor (RandomForestRegressor) to learn importance vectors.
4) Injects increasing Gaussian noise into targets and evaluates prediction MSE via K-fold CV.
5) Saves:
   - Figure_S1_NoiseSensitivity.png
   - (optional) CSV with noise vs mean/std MSE

Designed to be lightweight and reproducible.

Run:
  python code/synthetic_validation.py

Requirements:
  pandas, numpy, scikit-learn, matplotlib
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


@dataclass
class SyntheticConfig:
    random_seed: int = 42
    n_samples: int = 6000          # synthetic records
    n_criteria: int = 8            # C1–C8
    n_years: int = 4               # e.g., 2021–2024 (synthetic timeline only)
    n_evaluators: int = 6
    n_scenarios: int = 10
    k_folds: int = 5
    rf_n_estimators: int = 250
    rf_max_depth: int | None = None
    rf_min_samples_leaf: int = 2

    # Noise levels for Figure S1 (match your caption: 0.0 ... 0.5)
    noise_levels: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.22, 0.28, 0.35, 0.40, 0.45, 0.50)

    # Output locations
    out_fig_path: str = "results/figures/Figure_S1_NoiseSensitivity.png"
    out_csv_path: str = "results/supplementary/Supplementary_Figure_S1_noise_sensitivity.csv"


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)


def minmax_0_1(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Min-max normalize per-column to [0,1]."""
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    denom = np.maximum(x_max - x_min, eps)
    return (x - x_min) / denom


def generate_synthetic_dataset(cfg: SyntheticConfig) -> pd.DataFrame:
    """
    Generate synthetic context features and ground-truth importance vectors.

    Context:
      - Year index (mapped to e.g. 2021..)
      - Evaluator ID
      - Scenario ID

    Targets:
      - importance vector w (Dirichlet)
    """
    rng = np.random.default_rng(cfg.random_seed)

    # Context variables
    years = rng.integers(0, cfg.n_years, size=cfg.n_samples)
    evaluators = rng.integers(1, cfg.n_evaluators + 1, size=cfg.n_samples)
    scenarios = rng.integers(1, cfg.n_scenarios + 1, size=cfg.n_samples)

    # Ground-truth importance via Dirichlet.
    # To create structured variation by scenario, make concentration depend on scenario group.
    base_alpha = np.ones(cfg.n_criteria) * 2.0

    # Add mild scenario-based tilt (structured signal):
    # scenarios 1..n_scenarios map to one of 4 "policy regimes"
    regime = (scenarios - 1) % 4
    alphas = np.zeros((cfg.n_samples, cfg.n_criteria), dtype=float)

    for i in range(cfg.n_samples):
        a = base_alpha.copy()
        if regime[i] == 0:
            a[3] += 3.0  # emphasize C4
        elif regime[i] == 1:
            a[0] += 2.0  # emphasize C1
            a[2] += 1.0  # emphasize C3
        elif regime[i] == 2:
            a[6] += 2.5  # emphasize C7
        else:
            a[7] += 2.0  # emphasize C8
        # small year drift
        a = a * (1.0 + 0.05 * years[i])
        alphas[i, :] = a

    # Sample importance vectors
    W = np.vstack([rng.dirichlet(alphas[i, :]) for i in range(cfg.n_samples)])

    df = pd.DataFrame({
        "Year": years,  # keep as index; we can map to actual year later if needed
        "Evaluator": evaluators,
        "Scenario": scenarios,
    })
    for j in range(cfg.n_criteria):
        df[f"C{j+1}"] = W[:, j]

    return df


def evaluate_noise_sensitivity(
    df: pd.DataFrame,
    cfg: SyntheticConfig,
) -> pd.DataFrame:
    """
    For each noise level:
      - Add Gaussian noise to targets, re-normalize to simplex-like vector
      - Train/test using K-fold CV
      - Collect mean/std MSE across folds
    """
    rng = np.random.default_rng(cfg.random_seed)

    X = df[["Year", "Evaluator", "Scenario"]].to_numpy()
    Y_true = df[[f"C{i+1}" for i in range(cfg.n_criteria)]].to_numpy()

    # Scale features to [0,1] to reduce arbitrary magnitude dominance
    X_scaled = minmax_0_1(X.astype(float))

    kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.random_seed)

    results = []
    for nl in cfg.noise_levels:
        fold_mse: List[float] = []

        # Create noisy targets per noise level (same noise for all folds for comparability)
        noise = rng.normal(loc=0.0, scale=nl, size=Y_true.shape)
        Y_noisy = Y_true + noise

        # Clip negatives and renormalize rows to sum to 1 (importance vectors)
        Y_noisy = np.clip(Y_noisy, 0.0, None)
        row_sums = np.sum(Y_noisy, axis=1, keepdims=True)
        # Avoid division by zero: if a row is all zeros after clipping, revert to true vector
        zero_rows = (row_sums.squeeze() == 0.0)
        if np.any(zero_rows):
            Y_noisy[zero_rows, :] = Y_true[zero_rows, :]
            row_sums = np.sum(Y_noisy, axis=1, keepdims=True)

        Y_noisy = Y_noisy / row_sums

        for train_idx, test_idx in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = Y_noisy[train_idx], Y_noisy[test_idx]

            model = RandomForestRegressor(
                n_estimators=cfg.rf_n_estimators,
                random_state=cfg.random_seed,
                max_depth=cfg.rf_max_depth,
                min_samples_leaf=cfg.rf_min_samples_leaf,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            mse = mean_squared_error(y_test, pred)
            fold_mse.append(float(mse))

        results.append({
            "noise_level": float(nl),
            "mse_mean": float(np.mean(fold_mse)),
            "mse_std": float(np.std(fold_mse, ddof=1)) if len(fold_mse) > 1 else 0.0,
        })

    return pd.DataFrame(results)


def plot_noise_sensitivity(df_res: pd.DataFrame, cfg: SyntheticConfig) -> None:
    """
    Produce Supplementary Figure S1.
    """
    ensure_dirs(cfg.out_fig_path)

    x = df_res["noise_level"].to_numpy()
    y = df_res["mse_mean"].to_numpy()
    yerr = df_res["mse_std"].to_numpy()

    plt.figure()
    plt.plot(x, y, marker="o")
    # optional uncertainty band
    if np.any(yerr > 0):
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.2)

    plt.title("Effect of noise injection on synthetic validation performance")
    plt.xlabel("Noise level added to synthetic importance vectors")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.tight_layout()
    plt.savefig(cfg.out_fig_path, dpi=300)
    plt.close()


def main() -> None:
    cfg = SyntheticConfig()
    ensure_dirs(cfg.out_fig_path, cfg.out_csv_path)

    df = generate_synthetic_dataset(cfg)
    df_res = evaluate_noise_sensitivity(df, cfg)

    # Save CSV for supplementary reproducibility
    df_res.to_csv(cfg.out_csv_path, index=False)

    # Plot Figure S1
    plot_noise_sensitivity(df_res, cfg)

    print("Synthetic noise sensitivity experiment completed.")
    print(f"Saved: {cfg.out_fig_path}")
    print(f"Saved: {cfg.out_csv_path}")


if __name__ == "__main__":
    main()

