# Week 3: Budget-Constrained Discount Optimization

## 1. Objective
Develop a budget-constrained optimization module to allocate discounts among customers while maximizing expected incremental profit.

## 2. Technologies Used
- Python
- Pandas
- NumPy
- SciPy (`scipy.optimize.linprog`)

## 3. Implementation
Created `backend/budget_optimizer.py` to:
- Accept customer-level estimated treatment effects.
- Calculate expected incremental revenue.
- Calculate expected incremental profit after discount cost.
- Allocate discounts subject to a fixed budget.
- Generate customer-level allocation results and an optimization summary.

## 4. Optimization Method
The initial implementation uses linear programming through SciPy's `linprog` function.

The objective is to maximize:

Expected Incremental Profit = (Estimated ITE × Expected Order Value) − Discount Cost

The budget constraint ensures that total discount spending does not exceed the available budget.

## 5. Testing
Created `backend/test_budget_optimizer.py` to verify the allocation and budget constraint.

Sample test configuration:
- Customers: 5
- Budget: 30
- Discount cost per customer: 10

Results:
- Customers selected: 3
- Total spending: 30
- Expected incremental profit: 90

The test passed successfully.

## 6. Limitations
- The current implementation uses a fixed discount cost.
- The optimizer uses continuous linear programming followed by rounding to binary allocations. This is not guaranteed to be optimal for all integer-allocation problems.
- Customer-level DML estimates have not demonstrated reliable individual treatment-effect ranking in the current baseline.
- The current implementation has only been tested with sample customer data.

## 7. Future Work
- Introduce proper integer optimization.
- Integrate actual DML predictions.
- Support multiple discount levels.
- Evaluate budget utilization and expected profit on realistic datasets.