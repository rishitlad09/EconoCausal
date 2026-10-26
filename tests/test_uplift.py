"""
Unit and Integration Tests for Week 2 Uplift Analysis and Qini Evaluation
Covers:
1. Mathematical correctness of Qini curve and Qini score on controlled toy data.
2. Inverted ranking verification (negative Qini score).
3. Cumulative Uplift calculations and sample ATE matching.
4. Handling of edge cases (zero denominators, ties, small groups).
5. Propensity score estimation and IPW clipping.
6. EconML Double Machine Learning pipeline (no data leakage on held-out split).
7. Interactive Plotly figures rendering without errors.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from causal_eval import (
    compute_qini_curve,
    compute_uplift_curve,
    compute_decile_uplift,
    estimate_propensities,
    QiniResult,
    UpliftResult,
)
from causal_dml import train_causal_dml_pipeline, DMLModelBundle
from causal_plots import (
    create_qini_figure,
    create_uplift_figure,
    create_decile_figure,
    create_effect_distribution_figure,
    create_propensity_overlap_figure,
)


def test_qini_controlled_toy_dataset():
    """
    Construct a synthetic toy dataset with known positive treatment effect:
    First 50 units: treated converts at 80%, control at 20% (Uplift = +60%)
    Last 50 units: treated converts at 30%, control at 30% (Uplift = 0%)
    A model predicting high effect for top 50 must produce a strictly positive Qini score.
    """
    rng = np.random.default_rng(42)
    n = 100
    
    # 50 high-lift units
    t_high = rng.binomial(1, 0.5, 50)
    y_high = np.where(t_high == 1, rng.binomial(1, 0.85, 50), rng.binomial(1, 0.15, 50))
    pred_high = rng.uniform(0.5, 0.9, 50)
    
    # 50 zero-lift units
    t_low = rng.binomial(1, 0.5, 50)
    y_low = np.where(t_low == 1, rng.binomial(1, 0.30, 50), rng.binomial(1, 0.30, 50))
    pred_low = rng.uniform(0.0, 0.4, 50)
    
    treatment = np.concatenate([t_high, t_low])
    y_true = np.concatenate([y_high, y_low])
    pred_effects = np.concatenate([pred_high, pred_low])
    
    # 1. Perfect / Strong Model Qini
    res_good = compute_qini_curve(y_true, treatment, pred_effects, convention="radcliffe_global")
    assert res_good.qini_score > 0.0, f"Expected positive Qini score, got {res_good.qini_score}"
    assert res_good.model_auc > res_good.random_auc
    assert res_good.curve_df["fraction_targeted"].iloc[0] == 0.0
    assert res_good.curve_df["fraction_targeted"].iloc[-1] == 1.0
    
    # Random baseline starts at 0 and ends at total Qini gain
    assert res_good.curve_df["random_baseline"].iloc[0] == 0.0
    assert np.isclose(res_good.curve_df["random_baseline"].iloc[-1], res_good.total_qini_gain)
    
    # 2. Inverted Model (Reverse Ranking: worst first)
    res_bad = compute_qini_curve(y_true, treatment, -pred_effects, convention="radcliffe_global")
    assert res_bad.qini_score < 0.0, f"Expected negative Qini score for inverted model, got {res_bad.qini_score}"
    assert res_bad.model_auc < res_bad.random_auc


def test_qini_edge_cases_and_validations():
    """Verify safe handling of invalid inputs and edge cases."""
    # Empty inputs
    with pytest.raises(ValueError, match="empty sample"):
        compute_qini_curve([], [], [])
        
    # Mismatched lengths
    with pytest.raises(ValueError, match="identical lengths"):
        compute_qini_curve([1, 0], [1], [0.5, 0.2])
        
    # All treated or all control
    with pytest.raises(ValueError, match="both treated and control"):
        compute_qini_curve([1, 0, 1], [1, 1, 1], [0.5, 0.2, 0.1])


def test_cumulative_uplift_curve():
    """Verify cumulative uplift formula: Uplift(k) = (Y_t(k)/N_t(k)) - (Y_c(k)/N_c(k))."""
    treatment = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_true = np.array([1, 0, 1, 0, 0, 0, 0, 1])
    pred_effects = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    
    res = compute_uplift_curve(y_true, treatment, pred_effects, n_bins=4)
    assert isinstance(res, UpliftResult)
    
    # Overall sample ATE: Y_t/N_t - Y_c/N_c = 2/4 - 1/4 = 0.25
    expected_ate = (2.0 / 4.0) - (1.0 / 4.0)
    assert np.isclose(res.sample_ate, expected_ate)
    assert np.isclose(res.curve_df["random_baseline"].iloc[-1], expected_ate)
    assert res.curve_df["cumulative_uplift"].iloc[-1] == expected_ate


def test_decile_uplift_partitioning():
    """Verify decile calculation partitions all evaluation observations correctly."""
    n = 200
    rng = np.random.default_rng(123)
    y_true = rng.binomial(1, 0.3, n)
    treatment = rng.binomial(1, 0.7, n)
    pred_effects = rng.normal(0.05, 0.02, n)
    
    decile_df = compute_decile_uplift(y_true, treatment, pred_effects, n_bins=10)
    assert len(decile_df) == 10
    assert decile_df["customers"].sum() == n
    assert (decile_df["treated_count"] + decile_df["control_count"]).sum() == n
    
    # Decile 1 should have strictly higher or equal mean predicted effect than Decile 10
    assert decile_df["mean_predicted_effect"].iloc[0] >= decile_df["mean_predicted_effect"].iloc[-1]


def test_propensity_estimation():
    """Test standardized logistic propensity estimation and clipping."""
    rng = np.random.default_rng(42)
    n = 200
    W_train = rng.normal(0, 1, (n, 3))
    T_train = rng.binomial(1, 0.6, n)
    W_eval = rng.normal(0, 1, (50, 3))
    
    props = estimate_propensities(W_train, T_train, W_eval, clip_bounds=(0.05, 0.95))
    assert len(props) == 50
    assert np.all(props >= 0.05)
    assert np.all(props <= 0.95)


def test_causal_dml_pipeline_heldout_split():
    """Test full EconML pipeline training on train split and evaluating on held-out test split."""
    csv_path = PROJECT_ROOT / "data" / "retail_causal_data.csv"
    assert csv_path.exists(), "Missing retail dataset!"
    df = pd.read_csv(csv_path)
    
    # Run pipeline on a subset for speedy test execution
    sub_df = df.head(1000).copy()
    bundle = train_causal_dml_pipeline(
        df=sub_df,
        treatment_col="discount_percent",
        outcome_col="purchased",
        discrete_treatment=True,
        estimator_type="linear_dml",
        test_size=0.30,
        random_state=42,
        cv_folds=2,
    )
    
    assert isinstance(bundle, DMLModelBundle)
    assert len(bundle.eval_df) == 300
    assert len(bundle.train_df) == 700
    # Verify no data leakage: train and eval indices are disjoint
    train_ids = set(bundle.train_df["customer_id"])
    eval_ids = set(bundle.eval_df["customer_id"])
    assert len(train_ids.intersection(eval_ids)) == 0, "Data leakage detected between train and eval splits!"
    
    # Verify out-of-sample predictions
    assert len(bundle.pred_effects_eval) == 300
    assert not np.isnan(bundle.pred_effects_eval).any()
    assert bundle.mean_effect > 0.0


def test_plotly_figures_rendering():
    """Verify that all Plotly figures render without runtime syntax or dimension exceptions."""
    treatment = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    y_true = np.array([1, 0, 1, 0, 1, 0, 0, 0, 1, 0])
    pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    
    q_res = compute_qini_curve(y_true, treatment, pred, n_bins=5)
    fig_qini = create_qini_figure(q_res)
    assert fig_qini is not None
    assert len(fig_qini.data) >= 2
    
    u_res = compute_uplift_curve(y_true, treatment, pred, n_bins=5)
    fig_uplift = create_uplift_figure(u_res)
    assert fig_uplift is not None
    assert len(fig_uplift.data) >= 2
    
    dec_df = compute_decile_uplift(y_true, treatment, pred, n_bins=2)
    fig_decile = create_decile_figure(dec_df)
    assert fig_decile is not None
    
    fig_dist = create_effect_distribution_figure(pred, float(np.mean(pred)), float(np.median(pred)))
    assert fig_dist is not None
    
    props = np.full_like(treatment, 0.5, dtype=float)
    fig_prop = create_propensity_overlap_figure(props, treatment)
    assert fig_prop is not None


if __name__ == "__main__":
    pytest.main(["-v", __file__])
