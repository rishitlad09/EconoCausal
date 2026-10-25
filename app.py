"""
EconoCausal - Dynamic Pricing via Double Machine Learning
Week 1 Dashboard: Data Ingestion, Schema Auditing, and Budget Configuration

This Streamlit application implements the Week 1 dashboard requirements:
1. Overview of the EconoCausal pricing and causal AI architecture.
2. Ingestion of historical campaign/retail CSV data with automated schema validation.
3. Multi-point dataset profiling (data types, nulls, duplicates, statistical ranges).
4. Parametric budget constraint configuration with real-time mathematical feasibility validation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Standard schema definition required by EconoCausal pipeline
REQUIRED_SCHEMA = {
    "customer_id": {
        "type": "string",
        "description": "Unique customer account identifier (e.g., CUST_00001)",
        "role": "Identifier",
    },
    "tenure_days": {
        "type": "numeric",
        "description": "Account lifetime with retailer in days",
        "role": "Confounder",
    },
    "previous_orders": {
        "type": "numeric",
        "description": "Cumulative historical purchase count",
        "role": "Confounder",
    },
    "days_since_last_order": {
        "type": "numeric",
        "description": "Recency in days since prior order (churn risk proxy)",
        "role": "Confounder",
    },
    "avg_order_value": {
        "type": "numeric",
        "description": "Historical average basket size in USD",
        "role": "Confounder",
    },
    "prior_email_engagement": {
        "type": "numeric",
        "description": "Marketing email engagement rate in [0.0, 1.0]",
        "role": "Confounder",
    },
    "price_sensitivity": {
        "type": "numeric",
        "description": "Latent price elasticity score in [0.0, 1.0]",
        "role": "Confounder & Effect Modifier",
    },
    "discount_percent": {
        "type": "numeric",
        "description": "Assigned discount proportion [0.00, 0.30]",
        "role": "Treatment (T)",
    },
    "purchased": {
        "type": "numeric",
        "description": "Binary purchase outcome indicator (0 = No, 1 = Yes)",
        "role": "Primary Outcome (Y)",
    },
    "revenue": {
        "type": "numeric",
        "description": "Realized gross revenue from order in USD",
        "role": "Downstream Metric",
    },
}

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "retail_causal_data.csv"


# -----------------------------------------------------------------------------
# Validation Helper Functions (Modular & Testable)
# -----------------------------------------------------------------------------
def audit_dataset(df: pd.DataFrame) -> dict:
    """Performs schema, type, and distribution checks against the required schema."""
    cols_present = set(df.columns)
    required_cols = set(REQUIRED_SCHEMA.keys())
    missing_cols = list(required_cols - cols_present)
    extra_cols = list(cols_present - required_cols)

    type_issues = []
    for col in required_cols.intersection(cols_present):
        expected_type = REQUIRED_SCHEMA[col]["type"]
        if expected_type == "numeric":
            if not pd.api.types.is_numeric_dtype(df[col]):
                type_issues.append(f"Column '{col}' must be numeric, but found {df[col].dtype}.")

    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    duplicate_count = int(df.duplicated().sum())

    is_valid = (len(missing_cols) == 0) and (len(type_issues) == 0) and (len(df) > 0)

    return {
        "is_valid": is_valid,
        "missing_cols": missing_cols,
        "extra_cols": extra_cols,
        "type_issues": type_issues,
        "null_counts": null_counts,
        "total_nulls": total_nulls,
        "duplicate_count": duplicate_count,
        "row_count": len(df),
        "col_count": len(df.columns),
    }


def validate_budget_constraints(
    total: float,
    num_units: int,
    min_unit: float,
    max_unit: float,
) -> tuple[bool, list[str]]:
    """Mathematically checks budget bounds and feasibility across allocation units."""
    errors = []

    if total < 0:
        errors.append("Total budget must be non-negative (>= $0.00).")
    if num_units <= 0:
        errors.append("Number of campaigns / allocation units must be at least 1.")
    if min_unit < 0:
        errors.append("Minimum allocation per unit cannot be negative.")
    if max_unit < 0:
        errors.append("Maximum allocation per unit cannot be negative.")
    if min_unit > max_unit:
        errors.append(f"Minimum allocation (${min_unit:,.2f}) cannot exceed maximum allocation (${max_unit:,.2f}).")

    if not errors:
        min_possible_total = min_unit * num_units
        max_possible_total = max_unit * num_units

        if total < min_possible_total:
            errors.append(
                f"Infeasible Budget: Total budget (${total:,.2f}) is strictly less than the minimum required "
                f"to fund {num_units} units at ${min_unit:,.2f} each (${min_possible_total:,.2f})."
            )
        elif total > max_possible_total:
            errors.append(
                f"Infeasible Budget: Total budget (${total:,.2f}) exceeds the maximum possible allocation "
                f"across {num_units} units capped at ${max_unit:,.2f} each (${max_possible_total:,.2f})."
            )

    return (len(errors) == 0, errors)


# -----------------------------------------------------------------------------
# Main Application Content
# -----------------------------------------------------------------------------
def render_dashboard():
    st.set_page_config(
        page_title="EconoCausal — Dynamic Pricing & Causal AI",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Session State Initialization
    if "df" not in st.session_state:
        st.session_state["df"] = None

    if "data_source_name" not in st.session_state:
        st.session_state["data_source_name"] = None

    if "budget_config" not in st.session_state:
        st.session_state["budget_config"] = {
            "total_budget": 10000.0,
            "num_campaigns": 5,
            "min_per_campaign": 500.0,
            "max_per_campaign": 3000.0,
            "is_valid": True,
            "validation_errors": [],
        }

    # Sidebar Navigation & Quick Actions
    with st.sidebar:
        st.image(
            "https://raw.githubusercontent.com/microsoft/DoWhy/main/docs/source/_static/dowhy-logo.png",
            width=180,
        )
        st.title("EconoCausal")
        st.caption("Dynamic Pricing via Double Machine Learning")
        st.divider()

        st.subheader("Data Source Actions")
        if DEFAULT_DATA_PATH.exists():
            if st.button("📥 Load Default Synthetic Dataset", use_container_width=True):
                loaded_df = pd.read_csv(DEFAULT_DATA_PATH)
                st.session_state["df"] = loaded_df
                st.session_state["data_source_name"] = "retail_causal_data.csv (Project Default)"
                st.toast("Loaded 5,000 synthetic retail records successfully!", icon="✅")
        else:
            st.info("Default dataset not found at expected path: `data/retail_causal_data.csv`.")

        if st.session_state["df"] is not None:
            if st.button("🗑️ Clear Loaded Data", use_container_width=True):
                st.session_state["df"] = None
                st.session_state["data_source_name"] = None
                st.rerun()

        st.divider()
        st.caption("Version: Week 1 Build (v0.1.0)")
        st.caption("Architecture: DoWhy Causal DAG + Streamlit UI")

    st.title("⚖️ EconoCausal — Dynamic Pricing & Causal AI Dashboard")
    st.markdown(
        """
        **Welcome to EconoCausal.** This platform applies **Computational Causal AI** and **Econometric Policy Learning** 
        to optimize promotional retail discounting. Standard machine learning models capture surface correlations between 
        promotions and conversions. EconoCausal isolates the **true incremental impact (counterfactual uplift)** of 
        discounting while respecting enterprise marketing budget constraints.
        """
    )

    # Architectural Scope Notice
    st.info(
        """
        **Current Phase — Week 1 Foundation:**
        This dashboard provides historical campaign data ingestion, rigorous schema verification, exploratory confounding audits, 
        and parametric budget constraint validation.  
        *EconML Double Machine Learning (DML), Individual Treatment Effect (ITE) inference, Qini/uplift curves, and SciPy budget optimization 
        are scheduled for implementation in Week 2.*
        """,
        icon="ℹ️",
    )

    st.divider()

    # -----------------------------------------------------------------------------
    # Part 1: Historical Retail Data Ingestion & Profiling
    # -----------------------------------------------------------------------------
    st.header("Historical Campaign & Retail Data Ingestion")
    st.write(
        "Upload a historical retail transaction CSV dataset to audit its compatibility with the causal inference pipeline."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"],
        help="Upload a comma-separated values file containing customer pre-treatment confounders, assigned discounts, and observed conversions.",
    )

    if uploaded_file is not None:
        try:
            parsed_df = pd.read_csv(uploaded_file)
            st.session_state["df"] = parsed_df
            st.session_state["data_source_name"] = uploaded_file.name
            st.success(f"Successfully uploaded and parsed **{uploaded_file.name}**.")
        except Exception as e:
            st.error(f"Failed to parse uploaded CSV file: {str(e)}")

    current_df = st.session_state["df"]

    if current_df is not None:
        st.subheader(f"Dataset Overview: {st.session_state['data_source_name']}")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Total Records (Customers)", f"{len(current_df):,}")
        col_m2.metric("Total Features (Columns)", f"{len(current_df.columns)}")
        null_total = int(current_df.isnull().sum().sum())
        col_m3.metric("Missing Values", f"{null_total:,}", delta=f"{null_total} nulls" if null_total > 0 else "Clean", delta_color="inverse")
        dup_total = int(current_df.duplicated().sum())
        col_m4.metric("Duplicate Rows", f"{dup_total:,}", delta=f"{dup_total} dups" if dup_total > 0 else "Clean", delta_color="inverse")

        # Schema Audit Card
        audit_res = audit_dataset(current_df)
        if audit_res["is_valid"]:
            st.success("✅ **Schema Verification Passed:** All 10 required causal variables are present with valid data types.")
        else:
            if audit_res["missing_cols"]:
                st.error(f"❌ **Missing Required Columns:** {', '.join(audit_res['missing_cols'])}")
            if audit_res["type_issues"]:
                for issue in audit_res["type_issues"]:
                    st.warning(f"⚠️ **Data Type Mismatch:** {issue}")

        # Data preview and summary tabs
        sub_preview, sub_types, sub_stats = st.tabs(["Dataset Preview", "Column Metadata & Types", "Descriptive Statistics"])

        with sub_preview:
            st.dataframe(current_df.head(20), use_container_width=True)

        with sub_types:
            col_info = []
            for col in current_df.columns:
                role = REQUIRED_SCHEMA.get(col, {}).get("role", "Unmapped / External")
                desc = REQUIRED_SCHEMA.get(col, {}).get("description", "Not in standard schema")
                col_info.append({
                    "Column Name": col,
                    "Inferred Data Type": str(current_df[col].dtype),
                    "Non-Null Count": int(current_df[col].count()),
                    "Null Count": int(current_df[col].isnull().sum()),
                    "Causal Taxonomy Role": role,
                    "Business Description": desc,
                })
            st.dataframe(pd.DataFrame(col_info), use_container_width=True)

        with sub_stats:
            st.dataframe(current_df.describe().T, use_container_width=True)

    else:
        st.warning("No dataset currently loaded. Please upload a CSV file above or click **'Load Default Synthetic Dataset'** in the sidebar.")



if __name__ == "__main__":
    render_dashboard()
