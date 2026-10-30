# EconoCausal – Causal DAG Documentation

## 1. Objective

To estimate the causal effect of offering discounts on customer purchases using the DoWhy causal inference framework.

## 2. Treatment and Outcome

**Treatment:**
- Variable: `treatment`
- 1 = Discount offered
- 0 = No discount offered

**Outcome:**
- Variable: `purchased`
- 1 = Customer purchased
- 0 = Customer did not purchase

## 3. Confounders

The following four variables directly influence both treatment assignment and purchase probability:

1. `previous_orders`
2. `days_since_last_order`
3. `prior_email_engagement`
4. `price_sensitivity`

Other pre-treatment variables:
- `tenure_days`: Influences treatment assignment.
- `website_sessions_30d`: Influences purchase probability.

## 4. Causal Relationships

- previous_orders → treatment
- previous_orders → purchased
- days_since_last_order → treatment
- days_since_last_order → purchased
- prior_email_engagement → treatment
- prior_email_engagement → purchased
- price_sensitivity → treatment
- price_sensitivity → purchased
- tenure_days → treatment
- website_sessions_30d → purchased
- treatment → purchased

## 5. Causal DAG Diagram

```mermaid
graph TD
    PO[Previous Orders] --> T[Treatment: Discount]
    PO --> Y[Purchased]

    DSL[Days Since Last Order] --> T
    DSL --> Y

    PE[Prior Email Engagement] --> T
    PE --> Y

    PS[Price Sensitivity] --> T
    PS --> Y

    TD[Tenure Days] --> T
    WS[Website Sessions] --> Y

    T --> Y