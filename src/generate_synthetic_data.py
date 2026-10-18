
import numpy as np
import pandas as pd
from pathlib import Path

# 1. Configuration
SEED = 42
N = 10_000
rng = np.random.default_rng(SEED)

# 2. Customer history and profile
df = pd.DataFrame({
    "customer_id": [f"C{i:05d}" for i in range(1, N + 1)],
    "tenure_days": rng.integers(30, 2001, N),
    "previous_orders": rng.poisson(8, N),
    "days_since_last_order": rng.integers(1, 181, N),
    "avg_order_value": np.round(
        rng.lognormal(6.5, 0.45, N), 2
    ),
    "age": rng.integers(18, 71, N),
    "customer_segment": rng.choice(
        ["New", "Regular", "Premium"], N,
        p=[0.25, 0.50, 0.25]
    ),
    "region": rng.choice(
        ["North", "South", "East", "West"], N
    ),
    "membership_tier": rng.choice(
        ["Standard", "Silver", "Gold"], N,
        p=[0.50, 0.30, 0.20]
    ),
    "prior_email_engagement": np.round(
        rng.beta(2, 3, N), 3
    ),
    "email_clicks_90d": rng.poisson(3, N),
    "website_sessions_30d": rng.poisson(6, N),
    "cart_additions_30d": rng.poisson(2, N),
    "price_sensitivity": np.round(
        rng.beta(2, 2, N), 3
    ),
    "campaign_id": rng.choice(
        ["CMP001", "CMP002", "CMP003", "CMP004"], N
    ),
    "campaign_channel": rng.choice(
        ["Email", "SMS", "Push", "Web"], N
    ),
    "list_price": np.round(
        rng.uniform(500, 5000, N), 2
    )
})

# 3. Product economics
df["unit_cost"] = np.round(
    df["list_price"] * rng.uniform(0.40, 0.75, N), 2
)

# 4. Assign discounts based on customer characteristics
# This intentionally creates confounding.
def standardize(series):
    return (series - series.mean()) / (series.std() + 1e-8)

tenure_z = standardize(df["tenure_days"]).to_numpy()
orders_z = standardize(df["previous_orders"]).to_numpy()
recency_z = standardize(df["days_since_last_order"]).to_numpy()

engagement = df["prior_email_engagement"].to_numpy()
sensitivity = df["price_sensitivity"].to_numpy()

offer_score = (
    -0.3
    + 0.25 * tenure_z
    - 0.30 * orders_z
    + 0.45 * recency_z
    + 0.60 * sensitivity
    + 0.30 * (1 - engagement)
)

p_offer = 1 / (1 + np.exp(-offer_score))
offered = rng.binomial(1, p_offer)

discount = np.zeros(N)
discount[offered == 1] = rng.choice(
    [10, 20], size=offered.sum(), p=[0.60, 0.40]
)

df["discount_percent"] = discount
df["treatment"] = (discount > 0).astype(int)

# 5. Generate purchase probability
baseline_log_odds = (
    -0.8
    + 0.25 * orders_z
    - 0.25 * recency_z
    + 0.80 * engagement
    + 0.35 * sensitivity
    + 0.15 * np.log1p(
        df["website_sessions_30d"].to_numpy()
    )
)

discount_effect = (
    0.035 * discount
    + 0.025 * discount * sensitivity
)

p_with_discount = 1 / (
    1 + np.exp(-(baseline_log_odds + discount_effect))
)
p_without_discount = 1 / (
    1 + np.exp(-baseline_log_odds)
)

df["purchased"] = rng.binomial(1, p_with_discount)
df["purchase_probability"] = np.round(p_with_discount, 5)

# Evaluation-only field: exclude from model inputs
df["true_treatment_effect"] = np.round(
    p_with_discount - p_without_discount, 5
)

# 6. Revenue and profit
realized_price = (
    df["list_price"] * (1 - discount / 100)
)

df["revenue"] = np.round(
    df["purchased"] * realized_price, 2
)
df["discount_cost"] = np.round(
    df["purchased"] * df["list_price"] * discount / 100,
    2
)
df["profit"] = np.round(
    df["revenue"] - df["purchased"] * df["unit_cost"], 2
)

# 7. Save CSV
output_path = Path("data") / "synthetic_campaign_data.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

# 8. Validate
print("Dataset generated successfully")
print("Shape:", df.shape)
print("Saved to:", output_path)
print("Missing values:", df.isnull().sum().sum())

print("\nDiscount distribution:")
print(df["discount_percent"].value_counts().sort_index())

print("\nPurchase rate by discount:")
print(df.groupby("discount_percent")["purchased"].mean().round(3))

print("\nFirst five rows:")
print(df.head())
