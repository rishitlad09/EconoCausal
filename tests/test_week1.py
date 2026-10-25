"""
Unit and Integration Tests for EconoCausal Week 1 Deliverables
Covers:
1. Synthetic dataset integrity and schema compliance.
2. Structural Causal DAG validity and acyclicity.
3. Microsoft DoWhy causal inference workflow (Model, Identify, Estimate, Refute).
4. Streamlit dashboard utility functions (audit_dataset, validate_budget_constraints).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import audit_dataset, validate_budget_constraints, REQUIRED_SCHEMA


def test_dataset_exists_and_integrity():
    """Verify retail dataset exists and has expected dimensions and null status."""
    csv_path = PROJECT_ROOT / "data" / "retail_causal_data.csv"
    assert csv_path.exists(), f"Missing dataset at {csv_path}"
    
    df = pd.read_csv(csv_path)
    assert len(df) == 5000, f"Expected 5,000 rows, got {len(df)}"
    assert len(df.columns) == 10, f"Expected 10 columns, got {len(df.columns)}"
    assert df.isnull().sum().sum() == 0, "Dataset contains unexpected null values"
    assert df.duplicated().sum() == 0, "Dataset contains duplicate records"


def test_dashboard_dataset_audit():
    """Test schema auditing on valid and invalid dataframes."""
    csv_path = PROJECT_ROOT / "data" / "retail_causal_data.csv"
    df = pd.read_csv(csv_path)
    
    # Valid dataset audit
    res_valid = audit_dataset(df)
    assert res_valid["is_valid"] is True
    assert len(res_valid["missing_cols"]) == 0
    assert len(res_valid["type_issues"]) == 0
    assert res_valid["total_nulls"] == 0
    
    # Invalid dataset: missing columns
    df_missing = df.drop(columns=["price_sensitivity", "discount_percent"])
    res_missing = audit_dataset(df_missing)
    assert res_missing["is_valid"] is False
    assert "price_sensitivity" in res_missing["missing_cols"]
    assert "discount_percent" in res_missing["missing_cols"]
    
    # Invalid dataset: non-numeric type where numeric required
    df_bad_type = df.copy()
    df_bad_type["tenure_days"] = df_bad_type["tenure_days"].astype(str)
    res_bad_type = audit_dataset(df_bad_type)
    assert res_bad_type["is_valid"] is False
    assert len(res_bad_type["type_issues"]) > 0


def test_budget_constraints_validation():
    """Test mathematical and feasibility checks for enterprise budget configuration."""
    # Feasible budget
    ok, errors = validate_budget_constraints(total=10000.0, num_units=5, min_unit=500.0, max_unit=3000.0)
    assert ok is True
    assert len(errors) == 0
    
    # Infeasible: total budget less than minimum required
    ok, errors = validate_budget_constraints(total=2000.0, num_units=5, min_unit=500.0, max_unit=3000.0)
    assert ok is False
    assert any("strictly less than the minimum required" in e for e in errors)
    
    # Infeasible: total budget exceeds maximum possible
    ok, errors = validate_budget_constraints(total=20000.0, num_units=5, min_unit=500.0, max_unit=3000.0)
    assert ok is False
    assert any("exceeds the maximum possible" in e for e in errors)
    
    # Infeasible: min > max
    ok, errors = validate_budget_constraints(total=10000.0, num_units=5, min_unit=4000.0, max_unit=3000.0)
    assert ok is False
    assert any("cannot exceed maximum" in e for e in errors)
    
    # Negative budget
    ok, errors = validate_budget_constraints(total=-500.0, num_units=5, min_unit=100.0, max_unit=1000.0)
    assert ok is False
    assert any("must be non-negative" in e for e in errors)


def test_structural_causal_dag():
    """Verify structural DAG properties and acyclicity."""
    causal_edges = [
        ("tenure_days", "previous_orders"),
        ("previous_orders", "days_since_last_order"),
        ("previous_orders", "avg_order_value"),
        ("tenure_days", "discount_percent"),
        ("previous_orders", "discount_percent"),
        ("days_since_last_order", "discount_percent"),
        ("avg_order_value", "discount_percent"),
        ("prior_email_engagement", "discount_percent"),
        ("price_sensitivity", "discount_percent"),
        ("tenure_days", "purchased"),
        ("previous_orders", "purchased"),
        ("days_since_last_order", "purchased"),
        ("avg_order_value", "purchased"),
        ("prior_email_engagement", "purchased"),
        ("price_sensitivity", "purchased"),
        ("discount_percent", "purchased"),
        ("discount_percent", "revenue"),
        ("purchased", "revenue"),
        ("avg_order_value", "revenue"),
    ]
    G = nx.DiGraph()
    G.add_edges_from(causal_edges)
    
    assert nx.is_directed_acyclic_graph(G), "Causal graph is not a DAG!"
    assert G.number_of_nodes() == 9
    assert G.number_of_edges() == 19
    
    # Backdoor paths: All 6 confounders are common causes of discount_percent and purchased
    confounders = [
        "tenure_days", "previous_orders", "days_since_last_order",
        "avg_order_value", "prior_email_engagement", "price_sensitivity"
    ]
    for c in confounders:
        assert G.has_edge(c, "discount_percent")
        assert G.has_edge(c, "purchased")


def test_dowhy_causal_identification_and_estimation():
    """Test full DoWhy workflow: Model -> Identify -> Estimate -> Refute."""
    from dowhy import CausalModel
    
    csv_path = PROJECT_ROOT / "data" / "retail_causal_data.csv"
    df = pd.read_csv(csv_path)
    
    DAG_DOT = """digraph {
        tenure_days -> previous_orders;
        previous_orders -> days_since_last_order;
        previous_orders -> avg_order_value;

        tenure_days -> discount_percent;
        previous_orders -> discount_percent;
        days_since_last_order -> discount_percent;
        avg_order_value -> discount_percent;
        prior_email_engagement -> discount_percent;
        price_sensitivity -> discount_percent;

        tenure_days -> purchased;
        previous_orders -> purchased;
        days_since_last_order -> purchased;
        avg_order_value -> purchased;
        prior_email_engagement -> purchased;
        price_sensitivity -> purchased;

        discount_percent -> purchased;

        discount_percent -> revenue;
        purchased -> revenue;
        avg_order_value -> revenue;
    }"""
    
    model = CausalModel(
        data=df,
        treatment="discount_percent",
        outcome="purchased",
        graph=DAG_DOT
    )
    
    # 1. Identification
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    backdoor_vars = identified_estimand.get_backdoor_variables()
    assert "revenue" not in backdoor_vars, "Downstream revenue must NOT be in backdoor adjustment set!"
    assert len(backdoor_vars) == 6, f"Expected 6 confounders, got {len(backdoor_vars)}"
    
    # 2. Estimation
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        test_significance=False
    )
    assert estimate.value > 0.0, f"Expected positive discount treatment effect, got {estimate.value}"
    assert 0.40 <= estimate.value <= 0.80, f"Estimate {estimate.value} outside expected range [0.40, 0.80]"
    
    # 3. Refutation: Random Common Cause
    ref_random = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="random_common_cause",
        random_state=42
    )
    assert abs(ref_random.new_effect - estimate.value) < 0.05, "Estimate changed significantly under random noise!"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
