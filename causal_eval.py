"""
EconoCausal - Causal Uplift & Qini Evaluation Engine
Week 2 Module: Rigorous Out-of-Sample Uplift Analysis, Qini Metrics, and Observational Diagnostics.

Formulas & Conventions:
1. Radcliffe Qini Curve (Radcliffe, 2007):
   Q(k) = Y_t(k) - Y_c(k) * (N_t(k) / N_c(k))   [Local Ratio]
   or
   Q(k) = Y_t(k) - Y_c(k) * (N_t_total / N_c_total) [Global Ratio]
   where observations are sorted descending by predicted CATE (tau_hat).

2. Random Targeting Baseline:
   Baseline(p) = p * Q(N)  for fraction p in [0, 1].

3. Cumulative Uplift (ATE in top k):
   Uplift(k) = (Y_t(k) / N_t(k)) - (Y_c(k) / N_c(k))

4. Inverse Propensity Weighted (IPW) Causal Gain:
   Y_i^* = (T_i * Y_i / e(X_i)) - ((1 - T_i) * Y_i / (1 - e(X_i)))
   Q_ipw(k) = sum_{i=1}^k Y_i^*

All functions handle zero denominators, small groups, ties, and edge cases safely.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


@dataclass
class QiniResult:
    """Stores Qini curve evaluation results and summary metrics."""
    curve_df: pd.DataFrame
    qini_score: float               # Net Qini Area (AUQC_model - AUQC_random)
    model_auc: float                # Area Under Model Qini Curve
    random_auc: float               # Area Under Random Baseline
    oracle_auc: Optional[float]     # Area Under Oracle Qini Curve (if provided)
    qini_ratio: Optional[float]     # Normalized Qini Ratio vs Oracle
    total_qini_gain: float          # Qini gain at 100% targeting
    convention: str                 # 'radcliffe_global', 'radcliffe_local', or 'ipw'


@dataclass
class UpliftResult:
    """Stores Cumulative Uplift curve evaluation results."""
    curve_df: pd.DataFrame
    sample_ate: float               # Overall evaluation sample ATE (Y_t/N_t - Y_c/N_c)
    max_uplift: float               # Maximum observed uplift across fractions
    best_targeting_fraction: float  # Population fraction achieving maximum cumulative uplift


def estimate_propensities(
    W_train: np.ndarray,
    T_train: np.ndarray,
    W_eval: np.ndarray,
    clip_bounds: Tuple[float, float] = (0.02, 0.98),
    random_state: int = 42,
) -> np.ndarray:
    """
    Fits a standardized, calibrated Logistic Regression propensity model on training confounders
    and generates clipped propensity scores e(W) = P(T=1|W) for the evaluation sample.
    """
    if len(np.unique(T_train)) < 2:
        raise ValueError("Treatment array T_train must contain both treated (1) and control (0) units.")

    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=random_state, solver="lbfgs")
    )
    pipeline.fit(W_train, T_train)
    
    # Predict probabilities of treatment
    raw_props = pipeline.predict_proba(W_eval)[:, 1]
    
    # Clip to avoid extreme inverse weights
    clipped_props = np.clip(raw_props, clip_bounds[0], clip_bounds[1])
    return clipped_props


def compute_qini_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    pred_effects: np.ndarray,
    convention: str = "radcliffe_global",
    propensities: Optional[np.ndarray] = None,
    true_effects: Optional[np.ndarray] = None,
    n_bins: int = 100,
) -> QiniResult:
    """
    Computes the Qini curve, random baseline, theoretical oracle (if known), and Qini score.

    Parameters:
    -----------
    y_true : np.ndarray
        Observed binary outcomes Y in {0, 1}.
    treatment : np.ndarray
        Observed binary treatment indicators T in {0, 1}.
    pred_effects : np.ndarray
        Model-estimated treatment effects (ITE / CATE) used to rank observations descending.
    convention : str
        'radcliffe_global': Q(k) = Y_t(k) - Y_c(k) * (N_t_total / N_c_total) (Standard)
        'radcliffe_local' : Q(k) = Y_t(k) - Y_c(k) * (N_t(k) / N_c(k))
        'ipw'             : Horvitz-Thompson inverse propensity weighted cumulative causal gain
    propensities : Optional[np.ndarray]
        Propensity scores e(X) required if convention == 'ipw'.
    true_effects : Optional[np.ndarray]
        Ground-truth ITEs (available in controlled synthetic experiments) for Oracle benchmark.
    n_bins : int
        Number of evaluation points (quantiles) along the curve.

    Returns:
    --------
    QiniResult containing curve dataframe, AUQC, baseline AUC, and Qini score.
    """
    y_true = np.asarray(y_true).astype(float)
    treatment = np.asarray(treatment).astype(int)
    pred_effects = np.asarray(pred_effects).astype(float)

    n_samples = len(y_true)
    if n_samples == 0:
        raise ValueError("Cannot evaluate Qini curve on an empty sample.")
    if len(treatment) != n_samples or len(pred_effects) != n_samples:
        raise ValueError("Arrays y_true, treatment, and pred_effects must have identical lengths.")

    total_nt = int(np.sum(treatment))
    total_nc = n_samples - total_nt

    if total_nt == 0 or total_nc == 0:
        raise ValueError(
            f"Evaluation sample must contain both treated and control units. "
            f"Found N_t={total_nt}, N_c={total_nc}."
        )

    # Sort evaluation observations descending by predicted treatment effect
    sort_idx_model = np.argsort(-pred_effects, kind="mergesort")
    
    # Internal evaluation helper for a given sorting order
    def _evaluate_order(sort_indices: np.ndarray) -> np.ndarray:
        s_y = y_true[sort_indices]
        s_t = treatment[sort_indices]

        if convention == "ipw":
            if propensities is None:
                raise ValueError("Propensities must be provided for IPW convention.")
            s_prop = propensities[sort_indices]
            # Horvitz-Thompson transformed outcome
            y_star = (s_t * s_y / s_prop) - ((1 - s_t) * s_y / (1 - s_prop))
            return np.cumsum(y_star)

        cum_yt = np.cumsum(s_y * s_t)
        cum_yc = np.cumsum(s_y * (1 - s_t))
        cum_nt = np.cumsum(s_t)
        cum_nc = np.cumsum(1 - s_t)

        if convention == "radcliffe_local":
            ratio = np.divide(
                cum_nt,
                np.maximum(cum_nc, 1),
                out=np.full_like(cum_nt, total_nt / max(total_nc, 1), dtype=float),
                where=(cum_nc > 0)
            )
            qini = cum_yt - cum_yc * ratio
        else:  # radcliffe_global (default, standard in literature)
            global_ratio = total_nt / total_nc
            qini = cum_yt - cum_yc * global_ratio

        return qini

    raw_qini_model = _evaluate_order(sort_idx_model)
    total_qini_gain = float(raw_qini_model[-1])

    # Sample into n_bins quantiles for consistent plotting and smooth curves
    step = max(1, n_samples // n_bins)
    eval_indices = np.arange(step - 1, n_samples, step)
    if eval_indices[-1] != n_samples - 1:
        eval_indices = np.append(eval_indices, n_samples - 1)

    pop_fractions = (eval_indices + 1) / n_samples
    model_qini_sampled = raw_qini_model[eval_indices]
    random_qini_sampled = pop_fractions * total_qini_gain

    # Compute Oracle if true effects are provided
    oracle_qini_sampled = None
    oracle_auc = None
    qini_ratio = None
    if true_effects is not None:
        sort_idx_oracle = np.argsort(-np.asarray(true_effects), kind="mergesort")
        raw_qini_oracle = _evaluate_order(sort_idx_oracle)
        oracle_qini_sampled = raw_qini_oracle[eval_indices]

    # Prepend origin (0, 0)
    pop_fractions_full = np.concatenate([[0.0], pop_fractions])
    model_qini_full = np.concatenate([[0.0], model_qini_sampled])
    random_qini_full = np.concatenate([[0.0], random_qini_sampled])

    # Compute areas via numerical trapezoidal integration
    model_auc = float(trapezoid(model_qini_full, pop_fractions_full))
    random_auc = float(trapezoid(random_qini_full, pop_fractions_full))
    qini_score = model_auc - random_auc

    curve_data = {
        "fraction_targeted": pop_fractions_full,
        "population_percent": pop_fractions_full * 100.0,
        "n_targeted": (pop_fractions_full * n_samples).round().astype(int),
        "model_qini": model_qini_full,
        "random_baseline": random_qini_full,
    }

    if oracle_qini_sampled is not None:
        oracle_qini_full = np.concatenate([[0.0], oracle_qini_sampled])
        curve_data["oracle_qini"] = oracle_qini_full
        oracle_auc = float(trapezoid(oracle_qini_full, pop_fractions_full))
        oracle_net = oracle_auc - random_auc
        if abs(oracle_net) > 1e-9:
            qini_ratio = float(qini_score / oracle_net)

    curve_df = pd.DataFrame(curve_data)

    return QiniResult(
        curve_df=curve_df,
        qini_score=qini_score,
        model_auc=model_auc,
        random_auc=random_auc,
        oracle_auc=oracle_auc,
        qini_ratio=qini_ratio,
        total_qini_gain=total_qini_gain,
        convention=convention,
    )


def compute_uplift_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    pred_effects: np.ndarray,
    n_bins: int = 100,
) -> UpliftResult:
    """
    Computes cumulative average treatment effect (Uplift) across increasing fractions
    of the population ranked by predicted treatment effect.

    Formula:
    Uplift(k) = (Y_t(k) / N_t(k)) - (Y_c(k) / N_c(k))
    Incremental Gain(k) = k * Uplift(k)
    """
    y_true = np.asarray(y_true).astype(float)
    treatment = np.asarray(treatment).astype(int)
    pred_effects = np.asarray(pred_effects).astype(float)

    n_samples = len(y_true)
    if n_samples == 0:
        raise ValueError("Cannot evaluate uplift on empty sample.")

    total_nt = int(np.sum(treatment))
    total_nc = n_samples - total_nt
    if total_nt == 0 or total_nc == 0:
        raise ValueError("Sample must contain both treated and control units.")

    sample_yt = np.sum(y_true[treatment == 1])
    sample_yc = np.sum(y_true[treatment == 0])
    sample_ate = float((sample_yt / total_nt) - (sample_yc / total_nc))

    # Sort descending by predicted effect
    sort_idx = np.argsort(-pred_effects, kind="mergesort")
    s_y = y_true[sort_idx]
    s_t = treatment[sort_idx]

    cum_yt = np.cumsum(s_y * s_t)
    cum_yc = np.cumsum(s_y * (1 - s_t))
    cum_nt = np.cumsum(s_t)
    cum_nc = np.cumsum(1 - s_t)

    # Safe rates with zero-division handling
    rate_t = np.divide(cum_yt, np.maximum(cum_nt, 1), out=np.zeros_like(cum_yt), where=(cum_nt > 0))
    rate_c = np.divide(cum_yc, np.maximum(cum_nc, 1), out=np.zeros_like(cum_yc), where=(cum_nc > 0))
    raw_uplift = rate_t - rate_c

    step = max(1, n_samples // n_bins)
    eval_indices = np.arange(step - 1, n_samples, step)
    if eval_indices[-1] != n_samples - 1:
        eval_indices = np.append(eval_indices, n_samples - 1)

    pop_fractions = (eval_indices + 1) / n_samples
    sampled_uplift = raw_uplift[eval_indices]
    sampled_nt = cum_nt[eval_indices]
    sampled_nc = cum_nc[eval_indices]
    sampled_yt = cum_yt[eval_indices]
    sampled_yc = cum_yc[eval_indices]

    # Prepend origin
    pop_fractions_full = np.concatenate([[0.0], pop_fractions])
    # Uplift at fraction 0 initialized to top bin uplift or sample ATE
    uplift_full = np.concatenate([[sampled_uplift[0]], sampled_uplift])
    random_baseline_full = np.full_like(pop_fractions_full, sample_ate)
    incremental_gain_full = pop_fractions_full * n_samples * uplift_full

    curve_df = pd.DataFrame({
        "fraction_targeted": pop_fractions_full,
        "population_percent": pop_fractions_full * 100.0,
        "n_targeted": (pop_fractions_full * n_samples).round().astype(int),
        "cumulative_uplift": uplift_full,
        "random_baseline": random_baseline_full,
        "incremental_gain": incremental_gain_full,
        "n_treated": np.concatenate([[0], sampled_nt]),
        "n_control": np.concatenate([[0], sampled_nc]),
        "y_treated": np.concatenate([[0.0], sampled_yt]),
        "y_control": np.concatenate([[0.0], sampled_yc]),
    })

    # Identify best targeting fraction (maximum cumulative uplift)
    valid_mask = pop_fractions > 0.05  # Look beyond initial noise
    if np.any(valid_mask):
        best_idx = np.argmax(sampled_uplift[valid_mask])
        best_frac = float(pop_fractions[valid_mask][best_idx])
        max_uplift = float(sampled_uplift[valid_mask][best_idx])
    else:
        best_frac = 1.0
        max_uplift = sample_ate

    return UpliftResult(
        curve_df=curve_df,
        sample_ate=sample_ate,
        max_uplift=max_uplift,
        best_targeting_fraction=best_frac,
    )


def compute_decile_uplift(
    y_true: np.ndarray,
    treatment: np.ndarray,
    pred_effects: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Computes binned performance across deciles (or quantiles) ranked by predicted treatment effect.

    Each row represents a customer decile (Decile 1 = highest predicted benefit,
    Decile 10 = lowest / negative predicted benefit).
    """
    y_true = np.asarray(y_true).astype(float)
    treatment = np.asarray(treatment).astype(int)
    pred_effects = np.asarray(pred_effects).astype(float)

    n_samples = len(y_true)
    if n_samples < n_bins:
        raise ValueError(f"Sample size ({n_samples}) must be at least the number of bins ({n_bins}).")

    # Sort descending
    sort_idx = np.argsort(-pred_effects, kind="mergesort")
    s_y = y_true[sort_idx]
    s_t = treatment[sort_idx]
    s_pred = pred_effects[sort_idx]

    # Assign bin labels (1 = highest predicted effect)
    bin_sizes = np.full(n_bins, n_samples // n_bins)
    bin_sizes[: n_samples % n_bins] += 1
    bin_ids = np.repeat(np.arange(1, n_bins + 1), bin_sizes)

    decile_rows = []
    cum_incremental = 0.0

    for b in range(1, n_bins + 1):
        mask = (bin_ids == b)
        b_y = s_y[mask]
        b_t = s_t[mask]
        b_pred = s_pred[mask]

        n_b = len(b_y)
        nt_b = int(np.sum(b_t))
        nc_b = n_b - nt_b

        yt_b = float(np.sum(b_y[b_t == 1])) if nt_b > 0 else 0.0
        yc_b = float(np.sum(b_y[b_t == 0])) if nc_b > 0 else 0.0

        rate_t = yt_b / nt_b if nt_b > 0 else 0.0
        rate_c = yc_b / nc_b if nc_b > 0 else 0.0
        uplift = rate_t - rate_c
        incremental_units = uplift * n_b
        cum_incremental += incremental_units

        decile_rows.append({
            "decile": f"Decile {b}",
            "decile_num": b,
            "customers": n_b,
            "treated_count": nt_b,
            "control_count": nc_b,
            "treatment_rate": nt_b / n_b if n_b > 0 else 0.0,
            "treated_conversion_rate": rate_t,
            "control_conversion_rate": rate_c,
            "observed_uplift": uplift,
            "incremental_conversions": incremental_units,
            "cum_incremental_conversions": cum_incremental,
            "mean_predicted_effect": float(np.mean(b_pred)),
            "min_predicted_effect": float(np.min(b_pred)),
            "max_predicted_effect": float(np.max(b_pred)),
        })

    return pd.DataFrame(decile_rows)
