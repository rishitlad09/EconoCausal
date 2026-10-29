
import pandas as pd
import networkx as nx


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


if not hasattr(nx.algorithms, "d_separated"):
    from networkx.algorithms.d_separation import is_d_separator
    nx.algorithms.d_separated = is_d_separator

from dowhy import CausalModel

# Load the synthetic campaign dataset
DATA_PATH = "synthetic_campaign_data.csv"

df = pd.read_csv(DATA_PATH)

# Define treatment and outcome
treatment = "treatment"
outcome = "purchased"

# Potential pre-treatment confounders
confounders = [
    "tenure_days",
    "previous_orders",
    "days_since_last_order",
    "avg_order_value",
    "age",
    "customer_segment",
    "region",
    "membership_tier",
    "prior_email_engagement",
    "email_clicks_90d",
    "website_sessions_30d",
    "cart_additions_30d",
    "price_sensitivity",
    "campaign_id",
    "campaign_channel",
    "list_price",
]

# Check that required columns exist
required_columns = [treatment, outcome] + confounders
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# Prepare a working copy
analysis_df = df[required_columns].copy()

# Convert categorical variables into numeric indicator columns
analysis_df = pd.get_dummies(
    analysis_df,
    columns=[
        "customer_segment",
        "region",
        "membership_tier",
        "campaign_id",
        "campaign_channel",
    ],
    drop_first=True,
    dtype=int,
)

# Remove rows with missing values in the analysis variables
analysis_df = analysis_df.dropna()

print("Rows used:", len(analysis_df))
print("Treatment counts:")
print(analysis_df[treatment].value_counts())
print("\nPurchase counts:")
print(analysis_df[outcome].value_counts())

# Define the causal model
model = CausalModel(
    data=analysis_df,
    treatment=treatment,
    outcome=outcome,
    common_causes=[
        col for col in analysis_df.columns
        if col not in [treatment, outcome]
    ],
)

# Identify the causal estimand
estimand = model.identify_effect()
print("\nIdentified estimand:")
print(estimand)

# Estimate the average treatment effect

# Estimate the average treatment effect

# Estimate the average treatment effect
propensity_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)

estimate = model.estimate_effect(
    estimand,
    method_name="backdoor.propensity_score_matching",
    method_params={
        "propensity_score_model": propensity_model
    },
)

print("\nEstimated average treatment effect (ATE):")
print(estimate.value)
