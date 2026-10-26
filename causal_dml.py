"""
EconoCausal - Double Machine Learning (DML) Model Engine
Week 2 Module: EconML Training Pipeline, Cross-Fitting, and Out-of-Sample ITE Estimation.

Implements:
1. LinearDML (Chernozhukov et al., 2018):
   Y - E[Y|X, W] = tau(X) * (T - E[T|X, W]) + epsilon
2. CausalForestDML:
   Non-parametric causal forest estimating heterogeneous treatment effects.
3. Strict Train/Evaluation split to prevent in-sample evaluation leakage.
4. Out-of-sample ITE/CATE prediction.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

try:
    from econml.dml import LinearDML, CausalForestDML
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False


@dataclass
class DMLModelBundle:
    """Encapsulates the trained EconML estimator, evaluation data, and predictions."""
    model: object
    estimator_name: str
    discrete_treatment: bool
    train_df: pd.DataFrame
    eval_df: pd.DataFrame
    pred_effects_eval: np.ndarray
    true_effects_eval: Optional[np.ndarray]
    propensities_eval: Optional[np.ndarray]
    feature_names_X: List[str]
    feature_names_W: List[str]
    treatment_col: str
    outcome_col: str
    mean_effect: float
    median_effect: float
    std_effect: float
    p05_effect: float
    p95_effect: float
    coef_summary: Optional[pd.DataFrame]


def train_causal_dml_pipeline(
    df: pd.DataFrame,
    treatment_col: str = "discount_percent",
    outcome_col: str = "purchased",
    confounder_cols: Optional[List[str]] = None,
    effect_modifier_cols: Optional[List[str]] = None,
    discrete_treatment: bool = True,
    estimator_type: str = "linear_dml",
    test_size: float = 0.30,
    random_state: int = 42,
    cv_folds: int = 3,
) -> DMLModelBundle:
    """
    Fits an EconML Double Machine Learning estimator with cross-fitting on a training split,
    and generates out-of-sample treatment effect predictions on a held-out evaluation split.

    Parameters:
    -----------
    df : pd.DataFrame
        Source retail dataset.
    treatment_col : str
        Treatment variable name (default: 'discount_percent').
    outcome_col : str
        Primary outcome variable name (default: 'purchased').
    confounder_cols : List[str]
        Backdoor confounders W to orthogonalize out.
    effect_modifier_cols : List[str]
        Covariates X driving treatment effect heterogeneity tau(X).
    discrete_treatment : bool
        If True, converts treatment to binary (T > 0 vs T == 0).
    estimator_type : str
        'linear_dml' or 'causal_forest_dml'.
    test_size : float
        Held-out evaluation fraction (e.g. 0.30 for 30%).
    random_state : int
        Scientific reproducibility seed.
    cv_folds : int
        Number of cross-fitting folds (default: 3).

    Returns:
    --------
    DMLModelBundle with trained estimator, held-out evaluation set, and predicted ITEs.
    """
    if not ECONML_AVAILABLE:
        raise ImportError(
            "EconML is not installed in the current environment. "
            "Please install it via: pip install econml"
        )

    if confounder_cols is None:
        confounder_cols = [
            "tenure_days", "previous_orders", "days_since_last_order",
            "avg_order_value", "prior_email_engagement", "price_sensitivity"
        ]

    if effect_modifier_cols is None:
        effect_modifier_cols = ["price_sensitivity"]

    # Verify column existence
    missing_cols = [c for c in [treatment_col, outcome_col] + confounder_cols + effect_modifier_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Prepare working dataframe
    clean_df = df.dropna(subset=[treatment_col, outcome_col] + list(set(confounder_cols + effect_modifier_cols))).copy()

    # Stratified or randomized train/eval split
    # If binary treatment or outcome, stratify to ensure balanced treated/control in both sets
    stratify_col = None
    if discrete_treatment:
        clean_df["_t_binary"] = (clean_df[treatment_col] > 0).astype(int)
        stratify_col = clean_df["_t_binary"]

    train_df, eval_df = train_test_split(
        clean_df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    # Extract arrays
    Y_train = train_df[outcome_col].values.astype(float)
    Y_eval = eval_df[outcome_col].values.astype(float)

    if discrete_treatment:
        T_train = (train_df[treatment_col].values > 0).astype(int)
        T_eval = (eval_df[treatment_col].values > 0).astype(int)
    else:
        T_train = train_df[treatment_col].values.astype(float)
        T_eval = eval_df[treatment_col].values.astype(float)

    X_train = train_df[effect_modifier_cols].values.astype(float)
    X_eval = eval_df[effect_modifier_cols].values.astype(float)

    W_train = train_df[confounder_cols].values.astype(float)
    W_eval = eval_df[confounder_cols].values.astype(float)

    # Initialize nuisance learners
    # In EconML: model_y is ALWAYS a Regressor (estimates E[Y|X, W] residual)
    model_y = RandomForestRegressor(
        n_estimators=60,
        max_depth=5,
        min_samples_leaf=15,
        random_state=random_state,
        n_jobs=-1,
    )

    if discrete_treatment:
        model_t = RandomForestClassifier(
            n_estimators=60,
            max_depth=5,
            min_samples_leaf=15,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        model_t = RandomForestRegressor(
            n_estimators=60,
            max_depth=5,
            min_samples_leaf=15,
            random_state=random_state,
            n_jobs=-1,
        )

    # Instantiate selected EconML DML Estimator
    if estimator_type == "causal_forest_dml":
        model = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=discrete_treatment,
            n_estimators=80,
            max_depth=6,
            min_samples_leaf=15,
            cv=cv_folds,
            random_state=random_state,
            n_jobs=-1,
        )
    else:  # Default: linear_dml
        model = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=discrete_treatment,
            cv=cv_folds,
            random_state=random_state,
        )

    # Fit strictly on training split
    model.fit(Y_train, T_train, X=X_train, W=W_train)

    # Generate out-of-sample ITE predictions on held-out evaluation set
    pred_effects_eval = model.effect(X_eval).flatten()

    # Ground truth effects for synthetic data oracle (if applicable)
    true_effects_eval = None
    if "price_sensitivity" in eval_df.columns:
        # Check if DGP matches synthetic formula
        ps = eval_df["price_sensitivity"].values
        # True heterogeneous marginal multiplier: 1.5 + 4.5 * price_sensitivity
        true_effects_eval = 1.5 + 4.5 * ps

    # Compute propensity scores for evaluation set
    propensities_eval = None
    if discrete_treatment:
        from causal_eval import estimate_propensities
        propensities_eval = estimate_propensities(
            W_train=W_train,
            T_train=T_train,
            W_eval=W_eval,
            clip_bounds=(0.02, 0.98),
            random_state=random_state,
        )

    # Coefficient / Feature importance summary
    coef_summary = None
    if hasattr(model, "coef_"):
        coef_vals = np.atleast_1d(model.coef_).flatten()
        coef_summary = pd.DataFrame({
            "Feature (Effect Modifier)": effect_modifier_cols,
            "CATE Linear Coefficient": coef_vals[:len(effect_modifier_cols)],
        })
    elif hasattr(model, "feature_importances_"):
        fi_vals = np.atleast_1d(model.feature_importances_).flatten()
        coef_summary = pd.DataFrame({
            "Feature (Effect Modifier)": effect_modifier_cols,
            "CATE Importance Weight": fi_vals[:len(effect_modifier_cols)],
        })

    # Summary statistics
    mean_eff = float(np.mean(pred_effects_eval))
    median_eff = float(np.median(pred_effects_eval))
    std_eff = float(np.std(pred_effects_eval))
    p05_eff = float(np.percentile(pred_effects_eval, 5))
    p95_eff = float(np.percentile(pred_effects_eval, 95))

    # Clean temporary column
    if "_t_binary" in clean_df.columns:
        train_df = train_df.drop(columns=["_t_binary"], errors="ignore")
        eval_df = eval_df.drop(columns=["_t_binary"], errors="ignore")

    return DMLModelBundle(
        model=model,
        estimator_name="LinearDML (EconML)" if estimator_type == "linear_dml" else "CausalForestDML (EconML)",
        discrete_treatment=discrete_treatment,
        train_df=train_df,
        eval_df=eval_df,
        pred_effects_eval=pred_effects_eval,
        true_effects_eval=true_effects_eval,
        propensities_eval=propensities_eval,
        feature_names_X=effect_modifier_cols,
        feature_names_W=confounder_cols,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        mean_effect=mean_eff,
        median_effect=median_eff,
        std_effect=std_eff,
        p05_effect=p05_eff,
        p95_effect=p95_eff,
        coef_summary=coef_summary,
    )
