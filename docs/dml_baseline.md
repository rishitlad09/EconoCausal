# Week 2: Double Machine Learning Baseline

## Objective
Implement Double Machine Learning using EconML to estimate
individual treatment effects of discounts on customer purchases.

## Dataset
- Dataset: synthetic_campaign_data.csv
- Total records: 10,000
- Treatment: Binary discount indicator
- Outcome: Purchased (0 or 1)
- Training split: 80%
- Testing split: 20%

## Model
- Estimator: CausalForestDML
- Library: EconML 0.17.0
- Treatment model: RandomForestClassifier
- Outcome model: RandomForestRegressor
- Effect modifiers: 10 customer features

## Initial Results
- Customers evaluated: 2,000
- Average estimated treatment effect: 0.1657
- Theoretical average treatment effect: 0.1530
- Mean absolute ITE error: 0.0311
- ITE correlation: 0.0147

## Observations
The model successfully trains and generates individual
treatment-effect estimates.

The estimated average effect is reasonably close to the
theoretical average effect.

However, the low ITE correlation indicates that the model
has not yet demonstrated reliable customer-level effect
ranking.

## Limitations and Future Work
- Improve individual treatment-effect estimation.
- Evaluate model performance using uplift and Qini curves.
- Investigate treatment-effect heterogeneity.
- Explore budget-constrained discount allocation.