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


    st.divider()

    # Tabbed Interface
    tab_data, tab_budget, tab_arch = st.tabs([
        "📊 1. Historical Campaign Data",
        "💰 2. Budget Constraints",
        "📐 3. Causal Architecture & Schema",
    ])

    # -----------------------------------------------------------------------------
    # TAB 1: Historical Retail Data Ingestion & Profiling
    # -----------------------------------------------------------------------------
    with tab_data:
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

    # -----------------------------------------------------------------------------
    # TAB 2: Budget Constraints Configuration
    # -----------------------------------------------------------------------------
    with tab_budget:
        st.header("Enterprise Marketing Budget Constraints")
        st.write(
            """
            Configure the financial boundaries and allocation parameters for promotional discounting. 
            In Week 2, these parameters will govern the constrained optimization algorithm (SciPy) 
            allocating discounts based on estimated Individual Treatment Effects.
            """
        )

        prev_b = st.session_state["budget_config"]

        with st.form("budget_constraint_form"):
            col_b1, col_b2 = st.columns(2)

            with col_b1:
                total_budget = st.number_input(
                    "Total Promotional Budget ($ USD)",
                    min_value=0.0,
                    max_value=1_000_000_000.0,
                    value=float(prev_b["total_budget"]),
                    step=500.0,
                    format="%.2f",
                    help="Maximum aggregate capital available for all discount allocations across the campaign.",
                )

                num_campaigns = st.number_input(
                    "Number of Marketing Campaigns / Customer Tiers",
                    min_value=1,
                    max_value=100,
                    value=int(prev_b["num_campaigns"]),
                    step=1,
                    help="The number of distinct customer cohorts or campaign channels sharing this budget pool.",
                )

            with col_b2:
                min_per_campaign = st.number_input(
                    "Minimum Budget per Campaign / Cohort ($ USD)",
                    min_value=0.0,
                    max_value=1_000_000_000.0,
                    value=float(prev_b["min_per_campaign"]),
                    step=100.0,
                    format="%.2f",
                    help="Floor guarantee: the minimum capital that must be allocated to each cohort.",
                )

                max_per_campaign = st.number_input(
                    "Maximum Budget per Campaign / Cohort ($ USD)",
                    min_value=0.0,
                    max_value=1_000_000_000.0,
                    value=float(prev_b["max_per_campaign"]),
                    step=500.0,
                    format="%.2f",
                    help="Ceiling cap: the maximum promotional capital allowed for any single cohort.",
                )

            submitted = st.form_submit_button("💾 Validate & Save Budget Constraints", use_container_width=True)

        # Perform validation
        is_valid, validation_errors = validate_budget_constraints(
            total=total_budget,
            num_units=num_campaigns,
            min_unit=min_per_campaign,
            max_unit=max_per_campaign,
        )

        # Update state
        st.session_state["budget_config"] = {
            "total_budget": total_budget,
            "num_campaigns": num_campaigns,
            "min_per_campaign": min_per_campaign,
            "max_per_campaign": max_per_campaign,
            "is_valid": is_valid,
            "validation_errors": validation_errors,
        }

        if is_valid:
            st.success(
                f"✅ **Budget Constraints Validated:** Total available: **${total_budget:,.2f}** distributed across "
                f"**{num_campaigns}** campaigns (min: **${min_per_campaign:,.2f}**, max: **${max_per_campaign:,.2f}** per campaign)."
            )
            avg_alloc = total_budget / num_campaigns
            st.info(f"💡 Expected average allocation: **${avg_alloc:,.2f}** per campaign unit.", icon="📊")
        else:
            for err in validation_errors:
                st.error(f"❌ **Constraint Violation:** {err}")

        st.caption("Note: Budget parameters are validated and stored in session state. Optimization algorithms will be triggered during Week 2.")

    # -----------------------------------------------------------------------------
    # TAB 3: Causal Architecture & Schema
    # -----------------------------------------------------------------------------
    with tab_arch:
        st.header("Structural Causal Architecture & Directed Acyclic Graph (DAG)")
        st.markdown(
            """
            In causal AI, variables cannot be treated as interchangeable statistical predictors. 
            Every variable occupies an explicit structural position in a **Directed Acyclic Graph (DAG)** 
            that governs customer behavior, promotional targeting, and downstream financials.
            """
        )

        st.subheader("1. Interactive Structural Causal Graph (DAG)")
        st.caption(
            "Graph specification encoded in DOT language and validated via NetworkX (9 nodes, 19 directed causal edges, strictly acyclic)."
        )

        causal_dag_dot = """
        digraph CausalArchitecture {
            rankdir=LR;
            node [shape=box, style="filled,rounded", fontname="sans-serif", fontsize=10, margin="0.15,0.1"];
            edge [color="#34495e", arrowsize=0.7, penwidth=1.2];

            // Pre-treatment Confounders (Blue)
            node [fillcolor="#ebf5fb", color="#2980b9", fontcolor="#1b4f72"];
            tenure_days [label="Account Tenure\n(tenure_days)"];
            previous_orders [label="Prior Orders\n(previous_orders)"];
            days_since_last_order [label="Recency / Churn Risk\n(days_since_last_order)"];
            avg_order_value [label="Historical AOV\n(avg_order_value)"];
            prior_email_engagement [label="Email Engagement\n(prior_email_engagement)"];
            price_sensitivity [label="Price Sensitivity\n(price_sensitivity)"];

            // Policy Treatment (Orange)
            node [fillcolor="#fef5e7", color="#d35400", fontcolor="#7e5109", penwidth=2.2];
            discount_percent [label="POLICY TREATMENT (T)\n(discount_percent)"];

            // Primary Outcome (Green)
            node [fillcolor="#eafaf1", color="#27ae60", fontcolor="#145a32", penwidth=2.2];
            purchased [label="PRIMARY OUTCOME (Y)\n(purchased)"];

            // Downstream Financial Metric (Purple)
            node [fillcolor="#f5eef8", color="#8e44ad", fontcolor="#512e5f", penwidth=1.5];
            revenue [label="Downstream Financials\n(revenue)"];

            // Customer history mechanics
            tenure_days -> previous_orders;
            previous_orders -> days_since_last_order;
            previous_orders -> avg_order_value;

            // Confounders -> Treatment (Backdoor targeting paths)
            tenure_days -> discount_percent;
            previous_orders -> discount_percent;
            days_since_last_order -> discount_percent;
            avg_order_value -> discount_percent;
            prior_email_engagement -> discount_percent;
            price_sensitivity -> discount_percent;

            // Confounders -> Outcome (Organic purchase paths)
            tenure_days -> purchased;
            previous_orders -> purchased;
            days_since_last_order -> purchased;
            avg_order_value -> purchased;
            prior_email_engagement -> purchased;
            price_sensitivity -> purchased;

            // Primary Causal Lift Edge
            discount_percent -> purchased [color="#e67e22", penwidth=3.0, label=" Causal Lift (do-calculus)"];

            // Downstream revenue mechanics
            discount_percent -> revenue [style="dashed", color="#9b59b6"];
            purchased -> revenue [color="#27ae60", penwidth=2.0];
            avg_order_value -> revenue [style="dashed", color="#9b59b6"];
        }
        """

        st.graphviz_chart(causal_dag_dot, use_container_width=True)

        # Legend and taxonomy description
        col_l1, col_l2, col_l3, col_l4 = st.columns(4)
        col_l1.info(r"🔵 **Confounders ($W$):** Pre-treatment common causes driving both discount targeting and organic conversion.")
        col_l2.warning(r"🟠 **Treatment ($T$):** The actionable policy variable (`discount_percent` $\in [0.0, 0.30]$).")
        col_l3.success(r"🟢 **Primary Outcome ($Y$):** Counterfactual purchase conversion probability (`purchased` $\in \{0, 1\}$).")
        col_l4.error(r"🟣 **Downstream Metric:** Post-treatment financial cashflow (`revenue`). **Do NOT control for this!**")

        st.divider()

        st.subheader("2. Causal Engine Verification & Backdoor Identification (DoWhy)")
        st.markdown(
            """
            Using **Microsoft DoWhy** in `notebooks/02_causal_dag_dowhy.ipynb`, the causal graph was formally analyzed 
            under Pearl's **Backdoor Criterion** to isolate the causal effect of `discount_percent` on `purchased`:
            """
        )

        col_c1, col_c2, col_c3 = st.columns(3)
        col_c1.metric("Identified Estimand", "Backdoor Adjustment", "6 Confounders Blocked")
        col_c2.metric("Estimated ATE Slope (β̂)", "+0.5957", "p < 0.001 (Significant)")
        col_c3.metric("Random Noise Refuter", "p = 0.94", "Invariant to Noise (Passed)")

        st.markdown(
            r"""
            - **Backdoor Adjustment Set:** `{'avg_order_value', 'days_since_last_order', 'previous_orders', 'price_sensitivity', 'prior_email_engagement', 'tenure_days'}`
            - **Collider & Mediator Protection:** `revenue` is a post-treatment descendant. Conditioning on revenue would induce collider stratification bias and mediator blocking. DoWhy correctly excluded it from the adjustment set.
            - **Economic Interpretation:** A 10 percentage point increase in promotional discount (e.g., $5\% \to 15\%$) causes an estimated **$+5.96\%$** increase in baseline conversion probability.
            """
        )

        st.divider()

        st.subheader("3. Formal Variable Schema & Causal Taxonomy")
        schema_rows = []
        for col, meta in REQUIRED_SCHEMA.items():
            schema_rows.append({
                "Variable Name": f"`{col}`",
                "Data Type": meta["type"],
                "Causal Role": meta["role"],
                "Domain Description": meta["description"],
            })
        st.table(pd.DataFrame(schema_rows))

        st.subheader("4. Causal Identification Assumptions (Week 1)")
        st.markdown(
            r"""
            1. **Conditional Unconfoundedness (Ignorability):**  
               $(Y(t) \perp T) \mid \mathbf{W} \quad \forall t \in [0.0, 0.30]$  
               *All common causes of discount assignment and purchase conversion are captured within pre-treatment covariates $\mathbf{W}$.*
            2. **Positivity (Overlap):**  
               $0 < P(T \in \mathcal{A} \mid \mathbf{W} = \mathbf{w}) < 1 \quad \forall \mathbf{w}$  
               *Every customer segment has a non-zero probability of receiving any promotional discount level.*
            3. **Stable Unit Treatment Value Assumption (SUTVA):**  
               *No interference between customer purchasing decisions; consistent operational discount implementation.*
            """
        )


if __name__ == "__main__":
    render_dashboard()
