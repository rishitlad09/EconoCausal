
import pandas as pd
import networkx as nx
import numpy as np


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

# Confounders identified from the synthetic data-generation equations
confounders = [
    "previous_orders",
    "days_since_last_order",
    "prior_email_engagement",
    "price_sensitivity",
]

# Check that required columns exist
required_columns = [treatment, outcome] + confounders
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# Prepare a working copy
analysis_df = df[required_columns].copy()

# Convert categorical variables into numeric indicator columns

# Remove rows with missing values in the analysis variables
analysis_df = analysis_df.dropna()

print("Rows used:", len(analysis_df))
print("Treatment counts:")
print(analysis_df[treatment].value_counts())
print("\nPurchase counts:")
print(analysis_df[outcome].value_counts())

# Define the causal model

# Define the causal model
model = CausalModel(
    data=analysis_df,
    treatment=treatment,
    outcome=outcome,
    common_causes=confounders,
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

# Calculate theoretical ATE using the data-generation mechanism

orders_z = (
    df["previous_orders"] - df["previous_orders"].mean()
) / (df["previous_orders"].std() + 1e-8)

recency_z = (
    df["days_since_last_order"] - df["days_since_last_order"].mean()
) / (df["days_since_last_order"].std() + 1e-8)

baseline_log_odds = (
    -0.8
    + 0.25 * orders_z
    - 0.25 * recency_z
    + 0.80 * df["prior_email_engagement"]
    + 0.35 * df["price_sensitivity"]
    + 0.15 * np.log1p(df["website_sessions_30d"])
)

p_without_discount = 1 / (1 + np.exp(-baseline_log_odds))

p_with_10_discount = 1 / (
    1 + np.exp(-(
        baseline_log_odds
        + 0.035 * 10
        + 0.025 * 10 * df["price_sensitivity"]
    ))
)

p_with_20_discount = 1 / (
    1 + np.exp(-(
        baseline_log_odds
        + 0.035 * 20
        + 0.025 * 20 * df["price_sensitivity"]
    ))
)

expected_treated_probability = (
    0.60 * p_with_10_discount
    + 0.40 * p_with_20_discount
)

true_ate = (
    expected_treated_probability - p_without_discount
).mean()

print("\nTheoretical ATE from data-generation mechanism:")
print(round(float(true_ate), 4))

print("\nEstimated ATE from DoWhy:")
print(round(float(estimate.value), 4))

print("\nAbsolute estimation error:")
print(round(abs(float(estimate.value) - float(true_ate)), 4))


# Refutation Test 1: Random Common Cause
print("\n--- Random Common Cause Refutation ---")

refutation1 = model.refute_estimate(
    estimand,
    estimate,
    method_name="random_common_cause",
    num_simulations=10,
)

print(refutation1)



# Refutation Test 2: Placebo Treatment
print("\n--- Placebo Treatment Refutation ---")

try:
    refutation2 = model.refute_estimate(
        estimand,
        estimate,
        method_name="placebo_treatment_refuter",
        placebo_type="permute",
        num_simulations=10,
    )
    print(refutation2)

except Exception as error:
    print("Placebo test could not complete:")
    print(error)
