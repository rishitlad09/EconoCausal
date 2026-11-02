import pandas as pd
from budget_optimizer import optimize_budget


def test_budget_optimizer():

    customers = pd.DataFrame({
        "customer_id": [101, 102, 103, 104, 105],
        "estimated_ite": [0.20, 0.15, 0.30, 0.10, 0.25],
        "expected_order_value": [100, 200, 150, 120, 180]
    })

    allocation, summary = optimize_budget(
        customers,
        budget=30,
        discount_cost=10
    )

    assert summary["total_spent"] <= 30
    assert summary["customers_selected"] == 3
    assert summary["expected_incremental_profit"] == 90

    print("All budget optimizer tests passed!")


if __name__ == "__main__":
    test_budget_optimizer()