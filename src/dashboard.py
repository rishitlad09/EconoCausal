import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="EconoCausal | Dynamic Pricing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* ---------- SIDEBAR ---------- */

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] .stMarkdown p {
        color: #9ca3af;
    }

    /* ---------- HEADINGS ---------- */

    h1 {
        font-size: 2.4rem !important;
        font-weight: 750 !important;
        color: #111827 !important;
        letter-spacing: -0.5px;
    }

    h2 {
        font-size: 1.55rem !important;
        font-weight: 700 !important;
        color: #111827 !important;
    }

    h3 {
        font-size: 1.15rem !important;
        font-weight: 650 !important;
        color: #111827 !important;
    }

    /* ---------- METRICS ---------- */

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #111827 !important;
        font-size: 1.65rem !important;
        font-weight: 750 !important;
    }

    /* ---------- DIVIDERS ---------- */

    hr {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 1.5rem 0;
    }

    /* ---------- TABS ---------- */

    button[data-baseweb="tab"] {
        font-weight: 600;
    }

    /* ---------- DATAFRAME ---------- */

    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* ---------- BUTTONS ---------- */

    .stButton button {
        border-radius: 8px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    policy_data = pd.read_csv(
        "data/causal_policy_results.csv"
    )

    summary_data = pd.read_csv(
        "data/causal_modeling_summary.csv"
    )

    return policy_data, summary_data


df, summary_df = load_data()


# ============================================================
# SUMMARY VALUES
# ============================================================

summary_dict = dict(
    zip(
        summary_df["Metric"],
        summary_df["Value"]
    )
)

customers = int(
    summary_dict["Customers"]
)

policy_accuracy = (
    float(summary_dict["Policy accuracy"]) * 100
)

avg_profit_no_discount = float(
    summary_dict["Average profit without discount"]
)

avg_profit_policy = float(
    summary_dict["Average profit with personalized policy"]
)

avg_profit_gain = float(
    summary_dict["Average profit gain per customer"]
)

total_profit_gain = float(
    summary_dict["Total estimated profit gain"]
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## EconoCausal"
    )

    st.caption(
        "Causal Pricing Intelligence"
    )

    st.divider()

    st.markdown(
        "### Dashboard Controls"
    )

    discount_filter = st.multiselect(
        "Recommended Discount",
        options=[0, 10, 20],
        default=[0, 10, 20]
    )

    rows_to_display = st.slider(
        "Customer records",
        min_value=10,
        max_value=100,
        value=25,
        step=5
    )

    customer_search = st.text_input(
        "Search Customer ID"
    )

    st.divider()

    st.markdown(
        "### Model Information"
    )

    st.caption("Method")
    st.write("Double Machine Learning")

    st.caption("Frameworks")
    st.write("EconML + DoWhy")

    st.caption("Base Estimator")
    st.write("Random Forest")

    st.caption("Dataset")
    st.write("100,000 customers")


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns(
    [4, 1]
)

with header_left:

    st.title(
        "EconoCausal"
    )

    st.write(
        "Dynamic Pricing Intelligence powered by "
        "Causal Machine Learning"
    )

    st.caption(
        "Personalized discount recommendations designed "
        "to maximize expected customer profit."
    )

with header_right:

    st.metric(
        "Policy Accuracy",
        f"{policy_accuracy:.2f}%"
    )


st.divider()


# ============================================================
# EXECUTIVE KPI SECTION
# ============================================================

st.subheader(
    "Executive Overview"
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:

    st.metric(
        "Customers Analyzed",
        f"{customers:,}"
    )

with kpi2:

    st.metric(
        "Avg. Profit / Customer",
        f"${avg_profit_policy:,.2f}",
        f"+${avg_profit_gain:,.2f}"
    )

with kpi3:

    st.metric(
        "Total Estimated Gain",
        f"${total_profit_gain:,.0f}"
    )

with kpi4:

    st.metric(
        "Policy Accuracy",
        f"{policy_accuracy:.2f}%"
    )


st.divider()


# ============================================================
# BUSINESS IMPACT
# ============================================================

st.subheader(
    "Business Impact"
)

impact_left, impact_right = st.columns(
    [1.2, 1]
)


# ---------- Profit Comparison ----------

with impact_left:

    profit_comparison = pd.DataFrame(
        {
            "Strategy": [
                "No Discount",
                "Personalized Policy"
            ],
            "Average Profit": [
                avg_profit_no_discount,
                avg_profit_policy
            ]
        }
    )

    fig_profit = px.bar(
        profit_comparison,
        x="Strategy",
        y="Average Profit",
        text="Average Profit",
        title="Average Expected Profit per Customer"
    )

    fig_profit.update_traces(
        texttemplate="$%{text:.2f}",
        textposition="outside"
    )

    fig_profit.update_layout(
        height=420,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis_title="Expected Profit ($)",
        xaxis_title="",
        showlegend=False
    )

    st.plotly_chart(
        fig_profit,
        use_container_width=True
    )


# ---------- Business Metrics ----------

with impact_right:

    st.markdown(
        "### Financial Summary"
    )

    st.metric(
        "Profit Without Discount",
        f"${avg_profit_no_discount:,.2f}"
    )

    st.metric(
        "Profit With Personalized Policy",
        f"${avg_profit_policy:,.2f}"
    )

    st.metric(
        "Incremental Profit / Customer",
        f"${avg_profit_gain:,.2f}"
    )

    st.metric(
        "Estimated Total Incremental Profit",
        f"${total_profit_gain:,.0f}"
    )


st.divider()


# ============================================================
# PRICING POLICY ANALYTICS
# ============================================================

st.subheader(
    "Personalized Pricing Policy"
)

policy_left, policy_right = st.columns(2)


# ---------- Discount Allocation ----------

with policy_left:

    discount_counts = (
        df["recommended_discount"]
        .value_counts()
        .sort_index()
    )

    discount_labels = [
        f"${int(value)} Discount"
        for value in discount_counts.index
    ]

    fig_policy = go.Figure(
        data=[
            go.Pie(
                labels=discount_labels,
                values=discount_counts.values,
                hole=0.55,
                textinfo="percent"
            )
        ]
    )

    fig_policy.update_layout(
        title="Recommended Discount Allocation",
        height=430,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        ),
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fig_policy,
        use_container_width=True
    )


# ---------- Discount Profit ----------

with policy_right:

    avg_profit_by_discount = pd.DataFrame(
        {
            "Discount": [
                "$0",
                "$10",
                "$20"
            ],
            "Expected Profit": [
                df["profit_0_hat"].mean(),
                df["profit_10_hat"].mean(),
                df["profit_20_hat"].mean()
            ]
        }
    )

    fig_discount_profit = px.bar(
        avg_profit_by_discount,
        x="Discount",
        y="Expected Profit",
        text="Expected Profit",
        title="Expected Profit by Discount Level"
    )

    fig_discount_profit.update_traces(
        texttemplate="$%{text:.2f}",
        textposition="outside"
    )

    fig_discount_profit.update_layout(
        height=430,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Discount",
        yaxis_title="Expected Profit ($)",
        showlegend=False
    )

    st.plotly_chart(
        fig_discount_profit,
        use_container_width=True
    )


st.divider()


# ============================================================
# POLICY DISTRIBUTION
# ============================================================

st.subheader(
    "Recommended Customer Segments"
)

count_0 = int(
    (df["recommended_discount"] == 0).sum()
)

count_10 = int(
    (df["recommended_discount"] == 10).sum()
)

count_20 = int(
    (df["recommended_discount"] == 20).sum()
)

segment1, segment2, segment3 = st.columns(3)

with segment1:

    st.metric(
        "No Discount",
        f"{count_0:,}",
        f"{count_0 / customers * 100:.2f}%"
    )

with segment2:

    st.metric(
        "$10 Discount",
        f"{count_10:,}",
        f"{count_10 / customers * 100:.2f}%"
    )

with segment3:

    st.metric(
        "$20 Discount",
        f"{count_20:,}",
        f"{count_20 / customers * 100:.2f}%"
    )


st.divider()


# ============================================================
# CUSTOMER EXPLORER
# ============================================================

st.subheader(
    "Customer-Level Recommendations"
)

filtered_df = df[
    df["recommended_discount"].isin(
        discount_filter
    )
].copy()


if customer_search:

    filtered_df = filtered_df[
        filtered_df["customer_id"]
        .astype(str)
        .str.contains(
            customer_search,
            case=False,
            na=False
        )
    ]


display_columns = [
    "customer_id",
    "recommended_discount",
    "profit_0_hat",
    "profit_10_hat",
    "profit_20_hat",
    "recommended_profit_hat",
    "profit_gain_vs_no_discount"
]

available_columns = [
    column
    for column in display_columns
    if column in filtered_df.columns
]

customer_table = filtered_df[
    available_columns
].head(
    rows_to_display
).copy()


if "recommended_discount" in customer_table.columns:

    customer_table[
        "recommended_discount"
    ] = customer_table[
        "recommended_discount"
    ].apply(
        lambda x: f"${int(x)}"
    )


for column in [
    "profit_0_hat",
    "profit_10_hat",
    "profit_20_hat",
    "recommended_profit_hat",
    "profit_gain_vs_no_discount"
]:

    if column in customer_table.columns:

        customer_table[column] = (
            customer_table[column]
            .round(2)
        )


st.dataframe(
    customer_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MODEL INSIGHTS
# ============================================================

st.divider()

st.subheader(
    "Model Insights"
)

insight1, insight2, insight3 = st.columns(3)

with insight1:

    st.markdown(
        "### Causal Effect"

    )

    st.write(
        "Double Machine Learning estimates the "
        "incremental impact of each discount level "
        "on purchase probability."
    )


with insight2:

    st.markdown(
        "### Customer-Level Policy"
    )

    st.write(
        "Each customer receives a discount based "
        "on their estimated response and expected "
        "profit."
    )


with insight3:

    st.markdown(
        "### Profit Optimization"
    )

    st.write(
        "The recommended discount is selected by "
        "comparing expected profit across $0, $10 "
        "and $20 treatments."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "EconoCausal | Dynamic Pricing via Double Machine Learning"
)

st.caption(
    "Python • EconML • DoWhy • Random Forest • Plotly"
)