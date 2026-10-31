import pandas as pd

from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("synthetic_campaign_data.csv")

# Treatment and outcome
T = df["treatment"]
Y = df["purchased"]

# Features used to estimate individual treatment effects
X = df[["price_sensitivity"]]
W = df[
    [
        "previous_orders",
        "days_since_last_order",
        "prior_email_engagement",
        "price_sensitivity",
        "tenure_days",
        "website_sessions_30d",
    ]
]
# Split data
X_train, X_test, W_train, W_test, T_train, T_test, Y_train, Y_test = train_test_split(
    X, W, T, Y, test_size=0.2, random_state=42
)   
# Initialize Double Machine Learning model
model = CausalForestDML(
    model_y=RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=10,
        random_state=42
    ),
    model_t=RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=10,
        random_state=42
    ),
    discrete_treatment=True,
    n_estimators=100,
    min_samples_leaf=10,
    random_state=42
)

# Train model
model.fit(
    Y_train,
    T_train,
    X=X_train,
    W=W_train
)
# Estimate individual treatment effects
ite = model.effect(X_test)
# Calculate theoretical ITE using the dataset generation formula
test_df = df.loc[X_test.index]

orders_z = (
    df["previous_orders"] - df["previous_orders"].mean()
) / (df["previous_orders"].std() + 1e-8)

recency_z = (
    df["days_since_last_order"] - df["days_since_last_order"].mean()
) / (df["days_since_last_order"].std() + 1e-8)

baseline = (
    -0.8
    + 0.25 * orders_z.loc[X_test.index]
    - 0.25 * recency_z.loc[X_test.index]
    + 0.80 * test_df["prior_email_engagement"]
    + 0.35 * test_df["price_sensitivity"]
    + 0.15 * __import__("numpy").log1p(
        test_df["website_sessions_30d"]
    )
)

p0 = 1 / (1 + __import__("numpy").exp(-baseline))

p10 = 1 / (1 + __import__("numpy").exp(
    -(baseline + 0.035 * 10 + 0.025 * 10 * test_df["price_sensitivity"])
))

p20 = 1 / (1 + __import__("numpy").exp(
    -(baseline + 0.035 * 20 + 0.025 * 20 * test_df["price_sensitivity"])
))

true_ite = 0.60 * p10 + 0.40 * p20 - p0

print("\nITE validation:")
print("Theoretical average ITE:", round(float(true_ite.mean()), 4))
print("Estimated average ITE:", round(float(ite.mean()), 4))
print("Mean absolute ITE error:", round(
    float((ite.flatten() - true_ite).abs().mean()), 4
))
print("ITE correlation:", round(
    float(pd.Series(ite.flatten()).corr(true_ite)), 4
))

# Store results
results = pd.DataFrame({
    "customer_id": df.loc[X_test.index, "customer_id"].values,
    "estimated_ite": ite.flatten(),
    "actual_treatment": T_test.values,
    "actual_purchase": Y_test.values
})

results.to_csv("dml_ite_results.csv", index=False)

print("DML model trained successfully!")
print("Customers evaluated:", len(results))
print("Average estimated treatment effect:", round(float(ite.mean()), 4))
print("\nSample ITE predictions:")
print(results.head(10)) 