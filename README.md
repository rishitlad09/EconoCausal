# EconoCausal

**Dynamic Pricing via Double Machine Learning**

EconoCausal is a causal AI project focused on helping businesses make smarter promotional and discounting decisions. Instead of only predicting which customers are likely to purchase, it estimates how a discount may change each customer's probability of purchasing.

## Overview

Traditional predictive models identify patterns and correlations in customer behavior, but they do not necessarily reveal whether a promotion caused a purchase. EconoCausal uses **Double Machine Learning (DML)** to estimate the **Individual Treatment Effect (ITE)** of different discount offers.

The goal is to help marketing teams target customers who are more likely to purchase because of an incentive, while making better use of a limited promotional budget.

## Key Features

- **Causal effect estimation:** Estimate the impact of discounts on customer purchase behavior using Double Machine Learning.
- **Confounder analysis:** Represent assumptions about treatments, outcomes, and confounding variables with causal graphs.
- **Propensity score matching:** Help balance observed customer characteristics when comparing treatment and control groups.
- **Budget-aware discount allocation:** Use optimization to recommend discount amounts under a fixed marketing budget.
- **Causal uplift visualization:** Display uplift and Qini curves to evaluate targeting performance.
- **Model validation:** Use causal refutation tests to examine how robust the estimated effects are.
- **Data drift monitoring:** Flag potential changes in customer behavior that may affect model reliability.

## Example Use Case

Suppose a marketing team has a limited budget for discounts. Rather than giving every customer the same offer—or targeting only customers with a high predicted chance of leaving—EconoCausal estimates how different discounts may influence each customer's likelihood of purchasing.

The system can then use these estimates to recommend where promotional spending may have the greatest impact.

## Technology Stack

- **Language:** Python
- **Causal inference:** EconML, DoWhy
- **Optimization:** SciPy
- **Base estimators:** Random Forest or LightGBM
- **Visualization:** Plotly
- **Interface:** React

## How It Works

1. Load historical customer and campaign data.
2. Identify the treatment (discount), outcome (purchase), and relevant confounding variables.
3. Estimate individual treatment effects using Double Machine Learning.
4. Evaluate the model with uplift analysis and causal refutation tests.
5. Optimize discount allocation subject to the available budget.
6. Present effect estimates and allocation recommendations in a dashboard.

## Project Status

This README describes the intended scope of EconoCausal. Update the setup and usage sections once the implementation, dataset, and run commands are finalized.

## Disclaimer

Causal estimates depend on data quality, modeling assumptions, and the confounders captured in the data. Results are decision-support estimates, not guarantees of customer behavior.
