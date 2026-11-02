import numpy as np
import pandas as pd
from scipy.optimize import linprog


def optimize_budget(
    customers: pd.DataFrame,
    budget: float,
    discount_cost: float = 10.0
):
    """
    Allocate discounts to maximize expected incremental profit
    within a fixed budget.

    Required customer columns:
    - customer_id
    - estimated_ite
    - expected_order_value
    """

    required_columns = [
        "customer_id",
        "estimated_ite",
        "expected_order_value"
    ]

    for column in required_columns:
        if column not in customers.columns:
            raise ValueError(f"Missing required column: {column}")

    if budget < 0 or discount_cost <= 0:
        raise ValueError("Budget must be non-negative and discount cost positive.")

    if customers.empty:
        raise ValueError("Customer dataset is empty.")

    if customers[required_columns].isnull().any().any():
        raise ValueError("Customer data contains missing values.")

    if not np.isfinite(
        customers[["estimated_ite", "expected_order_value"]].to_numpy()
    ).all():
        raise ValueError("Customer data contains invalid numeric values.")

    if (customers["estimated_ite"] < 0).any():
        raise ValueError("Estimated ITE cannot be negative.")

    if (customers["expected_order_value"] < 0).any():
        raise ValueError("Expected order value cannot be negative.")

    # Expected incremental revenue from treatment
    expected_incremental_revenue = (
        customers["estimated_ite"] *
        customers["expected_order_value"]
    )

    # Expected incremental profit after discount cost
    incremental_profit = (
        expected_incremental_revenue - discount_cost
    )

    # Maximize profit by minimizing negative profit
    objective = -incremental_profit.to_numpy()

    # Each customer receives either 0 or 1 discount
    bounds = [(0, 1)] * len(customers)

    # Total discount spending cannot exceed budget
    constraints = [np.full(len(customers), discount_cost)]
    budget_limit = [budget]

    result = linprog(
        c=objective,
        A_ub=constraints,
        b_ub=budget_limit,
        bounds=bounds,
        method="highs"
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    allocation = np.rint(result.x).astype(int)

    output = customers.copy()
    output["expected_incremental_revenue"] = expected_incremental_revenue
    output["expected_incremental_profit"] = incremental_profit
    output["discount_allocated"] = allocation
    output["discount_spend"] = allocation * discount_cost

    summary = {
        "budget": budget,
        "total_spent": float(output["discount_spend"].sum()),
        "customers_selected": int(allocation.sum()),
        "expected_incremental_profit": float(
            (output["expected_incremental_profit"] * allocation).sum()
        )
    }

    return output, summary


if __name__ == "__main__":

    # Sample customer data for testing
    sample_customers = pd.DataFrame({
        "customer_id": [101, 102, 103, 104, 105],
        "estimated_ite": [0.20, 0.15, 0.30, 0.10, 0.25],
        "expected_order_value": [100, 200, 150, 120, 180]
    })

    allocation, summary = optimize_budget(
        customers=sample_customers,
        budget=30,
        discount_cost=10
    )

    print("\nCUSTOMER ALLOCATION")
    print(allocation.to_string(index=False))

    print("\nOPTIMIZATION SUMMARY")
    for key, value in summary.items():
        print(f"{key}: {value}")