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

from causal_eval import (
    compute_qini_curve,
    compute_uplift_curve,
    compute_decile_uplift,
    estimate_propensities,
    QiniResult,
    UpliftResult,
)
from causal_dml import (
    train_causal_dml_pipeline,
    DMLModelBundle,
    ECONML_AVAILABLE,
)
from causal_plots import (
    create_qini_figure,
    create_uplift_figure,
    create_decile_figure,
    create_effect_distribution_figure,
    create_propensity_overlap_figure,
)

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

    if "dml_bundle" not in st.session_state:
        st.session_state["dml_bundle"] = None

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
                st.session_state["dml_bundle"] = None
                st.toast("Loaded 5,000 synthetic retail records successfully!", icon="✅")
        else:
            st.info("Default dataset not found at expected path: `data/retail_causal_data.csv`.")

        if st.session_state["df"] is not None:
            if st.button("🗑️ Clear Loaded Data", use_container_width=True):
                st.session_state["df"] = None
                st.session_state["data_source_name"] = None
                st.session_state["dml_bundle"] = None
                st.rerun()

        st.divider()
        st.caption("Version: Week 2 Build (v0.2.0)")
        st.caption("Architecture: DoWhy DAG + EconML DML + Streamlit & Plotly")

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
    tab_data, tab_budget, tab_arch, tab_uplift = st.tabs([
        "📊 1. Historical Campaign Data",
        "💰 2. Budget Constraints",
        "📐 3. Causal Architecture & Schema",
        "📈 4. Double ML & Uplift Evaluation",
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
                st.session_state["dml_bundle"] = None
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

    # -----------------------------------------------------------------------------
    # TAB 4: Double Machine Learning (EconML) & Uplift Evaluation
    # -----------------------------------------------------------------------------
    with tab_uplift:
        st.header("Double Machine Learning (EconML) & Causal Uplift Evaluation")
        st.markdown(
            """
            Evaluate whether the trained Double Machine Learning (DML) causal model can rank customers 
            to produce **superior treatment allocation efficiency** compared to a random promotional rollout. 
            All metrics and Plotly visualizations are computed strictly on a **held-out evaluation sample**.
            """
        )

        current_df = st.session_state["df"]

        if current_df is None:
            st.warning(
                "⚠️ **No dataset loaded.** Please upload a CSV dataset in Tab 1 or click "
                "**'Load Default Synthetic Dataset'** in the sidebar to run the DML causal pipeline."
            )
        else:
            audit_res = audit_dataset(current_df)
            if not audit_res["is_valid"]:
                st.error(
                    "❌ **Dataset Schema Incompatible:** The loaded dataset does not meet the required schema. "
                    "Please review the missing columns or data type mismatches in Tab 1."
                )
            else:
                # -------------------------------------------------------------
                # 1. Model Configuration & Training Controls
                # -------------------------------------------------------------
                with st.expander("⚙️ **Causal Model & Evaluation Setup**", expanded=(st.session_state["dml_bundle"] is None)):
                    col_cfg1, col_cfg2 = st.columns(2)

                    with col_cfg1:
                        treatment_mode = st.radio(
                            "Treatment Variable Formulation",
                            options=[
                                "Binary Policy (Discount > 0% vs. No Discount [0%])",
                                "Continuous Treatment (Discount Percentage [0.00, 0.30])",
                            ],
                            index=0,
                            help="Binary treatment enables standard Radcliffe Qini curves and uplift evaluation against an untreated control group.",
                        )
                        is_binary = "Binary" in treatment_mode

                        estimator_choice = st.selectbox(
                            "EconML DML Estimator",
                            options=[
                                "LinearDML (Chernozhukov et al., 2018)",
                                "CausalForestDML (Athey & Wager, 2019)",
                            ],
                            index=0,
                            help="LinearDML fits linear CATE models with cross-fitting. CausalForestDML fits non-parametric random causal forests.",
                        )
                        estimator_key = "linear_dml" if "LinearDML" in estimator_choice else "causal_forest_dml"

                        qini_convention = st.selectbox(
                            "Qini Curve Convention",
                            options=[
                                "Radcliffe Global Ratio [Standard] (Y_t - Y_c * N_t_total / N_c_total)",
                                "Radcliffe Local Ratio (Y_t - Y_c * N_t(k) / N_c(k))",
                                "Inverse Propensity Weighting (IPW Causal Adjustment)",
                            ],
                            index=0,
                            help="Standard Radcliffe convention scales control conversions by total treatment/control ratio. IPW adjusts for observational selection bias.",
                        )
                        if "Global" in qini_convention:
                            conv_key = "radcliffe_global"
                        elif "Local" in qini_convention:
                            conv_key = "radcliffe_local"
                        else:
                            conv_key = "ipw"

                    with col_cfg2:
                        test_split_pct = st.slider(
                            "Held-Out Evaluation Split Fraction (%)",
                            min_value=15,
                            max_value=40,
                            value=30,
                            step=5,
                            help="Percentage of the dataset withheld strictly for out-of-sample causal evaluation to prevent data leakage.",
                        )

                        default_modifiers = ["price_sensitivity"]
                        all_confounders = [
                            "tenure_days", "previous_orders", "days_since_last_order",
                            "avg_order_value", "prior_email_engagement", "price_sensitivity"
                        ]
                        selected_modifiers = st.multiselect(
                            "Heterogeneity Effect Modifiers (X)",
                            options=all_confounders,
                            default=default_modifiers,
                            help="Covariates that moderate customer responsiveness to discounts tau(X).",
                        )
                        if not selected_modifiers:
                            selected_modifiers = default_modifiers

                        st.caption(
                            f"**Backdoor Confounders (W):** `{', '.join(all_confounders)}` "
                            "(Orthogonalized out via cross-fitted nuisance models)"
                        )

                    train_clicked = st.button("🚀 Train Causal DML Model & Evaluate Out-of-Sample", use_container_width=True)

                if train_clicked:
                    with st.spinner("Training Double Machine Learning model with cross-fitting and evaluating on held-out split..."):
                        try:
                            bundle = train_causal_dml_pipeline(
                                df=current_df,
                                treatment_col="discount_percent",
                                outcome_col="purchased",
                                confounder_cols=all_confounders,
                                effect_modifier_cols=selected_modifiers,
                                discrete_treatment=is_binary,
                                estimator_type=estimator_key,
                                test_size=test_split_pct / 100.0,
                                random_state=42,
                                cv_folds=3,
                            )
                            st.session_state["dml_bundle"] = bundle
                            st.session_state["last_conv_key"] = conv_key
                            st.success(
                                f"✅ Successfully trained **{bundle.estimator_name}**! "
                                f"Trained on {len(bundle.train_df):,} records, evaluated on {len(bundle.eval_df):,} held-out records."
                            )
                        except Exception as e:
                            st.error(f"❌ Error during model training and evaluation: {str(e)}")

                bundle = st.session_state.get("dml_bundle", None)

                if bundle is not None:
                    # Extract evaluation vectors
                    eval_df = bundle.eval_df
                    y_eval = eval_df[bundle.outcome_col].values.astype(float)
                    if bundle.discrete_treatment:
                        t_eval = (eval_df[bundle.treatment_col].values > 0).astype(int)
                    else:
                        t_eval = eval_df[bundle.treatment_col].values.astype(float)

                    pred_effects = bundle.pred_effects_eval

                    # Verify binary treatment for Qini / Uplift curves
                    if not bundle.discrete_treatment:
                        st.info(
                            "ℹ️ **Continuous Treatment Notice:** The model was trained with continuous discounts. "
                            "For binary Qini and uplift curve calculations, observations with `discount_percent > 0` "
                            "are compared against untreated control (`discount_percent == 0`)."
                        )
                        t_eval_binary = (eval_df[bundle.treatment_col].values > 0).astype(int)
                    else:
                        t_eval_binary = t_eval

                    # Compute Qini, Uplift, and Decile metrics
                    qini_res = compute_qini_curve(
                        y_true=y_eval,
                        treatment=t_eval_binary,
                        pred_effects=pred_effects,
                        convention=conv_key,
                        propensities=bundle.propensities_eval,
                        true_effects=bundle.true_effects_eval,
                        n_bins=100,
                    )

                    uplift_res = compute_uplift_curve(
                        y_true=y_eval,
                        treatment=t_eval_binary,
                        pred_effects=pred_effects,
                        n_bins=100,
                    )

                    decile_df = compute_decile_uplift(
                        y_true=y_eval,
                        treatment=t_eval_binary,
                        pred_effects=pred_effects,
                        n_bins=10,
                    )

                    # ---------------------------------------------------------
                    # 2. Model & Evaluation Audit Information Card
                    # ---------------------------------------------------------
                    with st.expander("📋 **Audited Model & Evaluation Metadata**", expanded=False):
                        col_a1, col_a2 = st.columns(2)
                        with col_a1:
                            st.markdown(f"- **Trained Estimator:** `{bundle.estimator_name}` (3-Fold Cross-Fitting)")
                            st.markdown(f"- **Treatment Variable:** `{bundle.treatment_col}` (Mode: {'Binary {0, 1}' if bundle.discrete_treatment else 'Continuous [0.0, 0.30]'})")
                            st.markdown(f"- **Primary Outcome:** `{bundle.outcome_col}` (Binary Conversion Indicator)")
                            st.markdown(f"- **Effect Modifiers (X):** `{', '.join(bundle.feature_names_X)}`")
                        with col_a2:
                            st.markdown(f"- **Training Sample Size:** `{len(bundle.train_df):,}` observations")
                            st.markdown(f"- **Held-Out Evaluation Sample:** `{len(bundle.eval_df):,}` observations (Strictly out-of-sample)")
                            nt_eval = int(np.sum(t_eval_binary))
                            nc_eval = len(t_eval_binary) - nt_eval
                            st.markdown(f"- **Evaluation Balance:** `{nt_eval:,}` Treated ({nt_eval/len(t_eval_binary):.1%}), `{nc_eval:,}` Control ({nc_eval/len(t_eval_binary):.1%})")
                            st.markdown(f"- **Assignment Mechanism:** Observational with observable confounding (Backdoor adjusted)")

                    # ---------------------------------------------------------
                    # 3. Performance Summary Metrics
                    # ---------------------------------------------------------
                    st.subheader("1. Out-of-Sample Performance Summary")

                    nt_eval = int(np.sum(t_eval_binary))
                    nc_eval = len(t_eval_binary) - nt_eval
                    trt_rate = nt_eval / len(t_eval_binary)

                    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                    col_p1.metric(
                        "Evaluation Sample Size",
                        f"{len(eval_df):,}",
                        help="Number of strictly held-out customer records used for evaluation.",
                    )
                    col_p2.metric(
                        "Treated / Control Split",
                        f"{nt_eval:,} / {nc_eval:,}",
                        delta=f"{trt_rate:.1%} treated",
                        delta_color="off",
                    )
                    col_p3.metric(
                        "Mean Predicted CATE (Lift)",
                        f"{bundle.mean_effect * 100.0:+.2f}%",
                        delta=f"Median: {bundle.median_effect * 100.0:+.2f}%",
                        delta_color="normal",
                    )
                    col_p4.metric(
                        "Net Qini Score (AUQC - AURC)",
                        f"{qini_res.qini_score:+.2f}",
                        delta="Outperformed Random" if qini_res.qini_score > 0 else "Underperformed",
                        delta_color="normal" if qini_res.qini_score > 0 else "inverse",
                    )

                    # Honest Model-vs-Random Assessment
                    if qini_res.qini_score > 0:
                        st.success(
                            f"✅ **Causal Model Outperforms Random Targeting:** "
                            f"The model achieves a Net Qini Gain of **{qini_res.qini_score:+.2f}** "
                            f"(Model AUQC: **{qini_res.model_auc:.2f}** vs. Random AURC: **{qini_res.random_auc:.2f}**). "
                            f"Prioritizing promotions by predicted Individual Treatment Effect (CATE) successfully concentrates "
                            f"high-lift customers in the earliest rollout tiers."
                        )
                    else:
                        st.warning(
                            f"⚠️ **Causal Model Does Not Outperform Random Targeting:** "
                            f"The model produced a Net Qini Score of **{qini_res.qini_score:+.2f}**. "
                            f"The learned ranking does not improve promotional allocation efficiency over a uniform random baseline."
                        )

                    st.divider()

                    # ---------------------------------------------------------
                    # 4. Interactive Plotly Visualizations
                    # ---------------------------------------------------------
                    st.subheader("2. Interactive Causal Uplift & Qini Visualizations (Plotly)")

                    sub_qini, sub_uplift, sub_deciles, sub_dist = st.tabs([
                        "📈 Cumulative Qini Gain Curve",
                        "🚀 Cumulative Uplift Curve",
                        "📊 Targeting Deciles & Calibration",
                        "🔬 ITE Distribution & Overlap Diagnostics",
                    ])

                    with sub_qini:
                        fig_qini = create_qini_figure(
                            qini_res,
                            title=f"Cumulative Qini Curve — {bundle.estimator_name} vs. Random Baseline",
                        )
                        st.plotly_chart(fig_qini, use_container_width=True)

                        col_qi1, col_qi2 = st.columns([2, 1])
                        with col_qi1:
                            st.markdown(
                                """
                                **How to interpret the Qini Curve:**
                                - **X-axis:** Cumulative percentage of the evaluation customer base targeted (0% to 100%).
                                - **Y-axis:** Cumulative incremental conversions generated over the control baseline.
                                - **Random Rollout Baseline (Dashed):** Expected conversions if customers are selected at random.
                                - **Shaded Area (Net Qini Score):** Represents the cumulative advantage of causal AI targeting over random allocation.
                                - **Theoretical Oracle (Dotted):** The optimal theoretical upper bound achievable when true individual treatment effects are known.
                                """
                            )
                        with col_qi2:
                            st.info(
                                f"**Convention Applied:** `{qini_res.convention}`\n\n"
                                f"- Total Incremental Conversions at 100%: **{qini_res.total_qini_gain:+.1f}**\n"
                                f"- Random Baseline AURC: **{qini_res.random_auc:.2f}**\n"
                                f"- Causal Model AUQC: **{qini_res.model_auc:.2f}**\n"
                                + (f"- Normalized Qini Ratio: **{qini_res.qini_ratio:.1%}**" if qini_res.qini_ratio else "")
                            )

                    with sub_uplift:
                        fig_uplift = create_uplift_figure(
                            uplift_res,
                            title=f"Cumulative Uplift Curve — {bundle.estimator_name} Incremental Conversion Rate",
                        )
                        st.plotly_chart(fig_uplift, use_container_width=True)

                        st.markdown(
                            f"""
                            **Interpretation & Strategic Targeting Cutoff:**
                            - **Sample Average Treatment Effect (ATE):** **{uplift_res.sample_ate:+.2%}** (Red dashed horizontal line).
                            - **Peak Cumulative Uplift:** Reaches **{uplift_res.max_uplift:+.2%}** at **{uplift_res.best_targeting_fraction:.0%}** population targeted.
                            - Targeting only the top customer cohorts captures higher-than-average conversion lift, confirming that promotional elasticity is heterogeneous.
                            """
                        )

                    with sub_deciles:
                        fig_decile = create_decile_figure(
                            decile_df,
                            title="Observed Incremental Lift vs. Model Predicted CATE Across Customer Deciles",
                        )
                        st.plotly_chart(fig_decile, use_container_width=True)

                        st.markdown("**Decile Level Granular Metrics Table:**")
                        display_dec = decile_df[[
                            "decile", "customers", "treated_count", "control_count",
                            "treatment_rate", "treated_conversion_rate", "control_conversion_rate",
                            "observed_uplift", "incremental_conversions", "mean_predicted_effect"
                        ]].copy()
                        display_dec["treatment_rate"] = display_dec["treatment_rate"].apply(lambda x: f"{x:.1%}")
                        display_dec["treated_conversion_rate"] = display_dec["treated_conversion_rate"].apply(lambda x: f"{x:.1%}")
                        display_dec["control_conversion_rate"] = display_dec["control_conversion_rate"].apply(lambda x: f"{x:.1%}")
                        display_dec["observed_uplift"] = display_dec["observed_uplift"].apply(lambda x: f"{x*100:+.2f}%")
                        display_dec["incremental_conversions"] = display_dec["incremental_conversions"].apply(lambda x: f"{x:+.1f}")
                        display_dec["mean_predicted_effect"] = display_dec["mean_predicted_effect"].apply(lambda x: f"{x*100:+.2f}%")
                        st.dataframe(display_dec, use_container_width=True)

                    with sub_dist:
                        col_d1, col_d2 = st.columns(2)

                        with col_d1:
                            fig_dist = create_effect_distribution_figure(
                                pred_effects=pred_effects,
                                mean_val=bundle.mean_effect,
                                median_val=bundle.median_effect,
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)

                        with col_d2:
                            if bundle.propensities_eval is not None:
                                fig_prop = create_propensity_overlap_figure(
                                    propensities=bundle.propensities_eval,
                                    treatment=t_eval_binary,
                                )
                                st.plotly_chart(fig_prop, use_container_width=True)
                            else:
                                st.info("Propensity overlap diagnostic is generated for binary treatment formulations.")

                    st.divider()

                    # ---------------------------------------------------------
                    # 5. Descriptive Targeting Simulator (Without Budget Optimization)
                    # ---------------------------------------------------------
                    st.subheader("3. Descriptive Targeting Simulator (Without Budget Optimization)")
                    st.caption(
                        "Explore the composition and estimated incremental impact of targeting top customer cohorts based on model rankings. "
                        "Note: These are descriptive causal model estimates. Constrained budget optimization will be implemented separately."
                    )

                    target_pct = st.slider(
                        "Select Population Targeting Fraction (%)",
                        min_value=5,
                        max_value=100,
                        value=30,
                        step=5,
                        help="Select the top percentile of customers prioritized by predicted treatment effect.",
                    )

                    # Compute descriptive statistics for selected cohort
                    k_cutoff = int(np.ceil((target_pct / 100.0) * len(eval_df)))
                    sort_order = np.argsort(-pred_effects)
                    top_indices = sort_order[:k_cutoff]
                    bottom_indices = sort_order[k_cutoff:]

                    top_effects = pred_effects[top_indices]
                    top_eval_sub = eval_df.iloc[top_indices]
                    bottom_eval_sub = eval_df.iloc[bottom_indices] if len(bottom_indices) > 0 else None

                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    col_s1.metric("Customers Targeted", f"{k_cutoff:,}", delta=f"{target_pct}% of evaluation set")
                    col_s2.metric("Mean Predicted Lift", f"{np.mean(top_effects) * 100.0:+.2f}%")
                    expected_incremental_buyers = float(np.sum(top_effects))
                    col_s3.metric("Projected Incremental Buyers", f"+{expected_incremental_buyers:,.1f}")
                    if "price_sensitivity" in top_eval_sub.columns:
                        col_s4.metric("Avg Price Sensitivity (Targeted)", f"{top_eval_sub['price_sensitivity'].mean():.3f}")

                    # Comparison Table between Targeted and Untargeted Cohorts
                    comp_rows = []
                    compare_vars = ["price_sensitivity", "tenure_days", "previous_orders", "days_since_last_order", "avg_order_value", "prior_email_engagement"]
                    for var in compare_vars:
                        if var in top_eval_sub.columns:
                            val_top = float(top_eval_sub[var].mean())
                            val_bottom = float(bottom_eval_sub[var].mean()) if bottom_eval_sub is not None else float(eval_df[var].mean())
                            comp_rows.append({
                                "Customer Attribute": var,
                                "Targeted Cohort Mean (Top Beneficiaries)": f"{val_top:.2f}",
                                "Non-Targeted Cohort Mean": f"{val_bottom:.2f}",
                                "Relative Difference (%)": f"{(val_top - val_bottom) / max(abs(val_bottom), 1e-6) * 100:+.1f}%",
                            })

                    st.markdown("**Targeted vs. Non-Targeted Customer Cohort Comparison:**")
                    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

                    st.info(
                        "💡 **Targeting Insight:** Notice how the DML causal model prioritizes customers with significantly higher "
                        "price sensitivity and churn risk, as these customer segments exhibit the highest counterfactual uplift "
                        "when offered a discount."
                    )



if __name__ == "__main__":
    render_dashboard()
