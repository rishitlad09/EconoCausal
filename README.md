# EconoCausal

**Dynamic Pricing via Double Machine Learning (Economics & Causal AI)**

EconoCausal is a computational causal AI platform designed to optimize enterprise promotional and discount allocations. Rather than naively predicting which customers are likely to purchase (predictive modeling), EconoCausal isolates the **true counterfactual uplift** (the causal treatment effect of a promotional price discount on customer conversion probability) while respecting financial budget constraints.

---

## 1. Project Overview & Current Status

In observational retail data, promotional discounts are systematically confounded: churning customers and highly price-sensitive shoppers are targeted with larger discounts, while affluent loyalists are charged full price. A naive regression or correlation model conflates treatment effects with baseline behavioral differences.

- **Completed Milestones:**
  - **Week 1:** Causal Graphing, Pearl's DAG, DoWhy Backdoor Identification & Scaffolding.
  - **Week 2:** EconML Double Machine Learning (DML), Out-of-Sample ITE Estimation, Radcliffe Qini Curves, Cumulative Uplift Analysis, and Interactive Plotly Visualizations.
- **Next Milestone:**
  - **Week 3 / Milestone 3:** Constrained Enterprise Marketing Budget Optimization (SciPy) under linear and non-linear campaign constraints.

---

## 2. Implemented Capabilities

### A. Week 1 Deliverables
1. **Synthetic Retail Dataset (5,000 Customers):** Grounded in realistic behavioral distributions with observable confounding and zero-hurdle discount policies ([`data/retail_causal_data.csv`](data/retail_causal_data.csv)).
2. **Structural Causal Directed Acyclic Graph (DAG):** 9 nodes and 19 directed edges formalizing physical retail causal relationships in DOT syntax, validated for strict acyclicity with NetworkX.
3. **Microsoft DoWhy Causal Inference Engine:** Backdoor adjustment, linear regression adjustment estimation ($\hat{\beta} \approx 0.5957$), and refutation testing (Random Common Cause $p = 0.94$, Data Subset 80%).
4. **Streamlit UI Scaffolding:** Data ingestion, 10-point schema auditing, parametric budget configuration, and DAG visualization.

### B. Week 2 Deliverables (Uplift & Causal Evaluation)
1. **EconML Double Machine Learning (DML) Pipeline ([`causal_dml.py`](causal_dml.py)):**
   - Implements `LinearDML` (Chernozhukov et al., 2018) and `CausalForestDML` (Athey et al.).
   - Utilizes $K$-fold cross-fitting (`cv=3`) with Random Forest nuisance models to orthogonalize out confounding.
   - Strictly separates training from evaluation (default 70% train / 30% held-out test split) to prevent data leakage and over-optimistic performance claims.
2. **Out-of-Sample ITE / CATE Estimation:**
   - Generates individual treatment effect predictions $\hat{\tau}(X_i)$ strictly for held-out evaluation units.
3. **Radcliffe Qini Curve Engine ([`causal_eval.py`](causal_eval.py)):**
   - Implements the standard Radcliffe (2007) Qini formulation:
     $$Q(k) = Y_t(k) - Y_c(k) \times \frac{N_t^{total}}{N_c^{total}}$$
   - Generates the analytical Random Targeting Baseline:
     $$\text{Baseline}(p) = p \times Q(N)$$
   - Computes Net Qini Score (AUQC - AURC) via numerical trapezoidal integration (`scipy.integrate.trapezoid`).
   - Includes Theoretical Oracle curve when true data-generating process (DGP) parameters are available.
4. **Cumulative Uplift Curve Engine:**
   - Evaluates cumulative incremental conversion rates $\text{Uplift}(k) = \frac{Y_t(k)}{N_t(k)} - \frac{Y_c(k)}{N_c(k)}$ against the sample Average Treatment Effect (ATE).
   - Identifies the empirical peak-benefit targeting cutoff.
5. **Observational Confounder Adjustment (IPW):**
   - Fits standardized, calibrated logistic regression propensity scores $\hat{e}(W_i)$ with weight clipping $[0.02, 0.98]$ to compute Horvitz-Thompson causal gains:
     $$Y_i^* = \frac{T_i Y_i}{\hat{e}(W_i)} - \frac{(1 - T_i) Y_i}{1 - \hat{e}(W_i)}$$
6. **Targeting Decile Calibration:**
   - Discretizes evaluation customers into deciles ranked by predicted benefit; tracks empirical uplift, observed conversion rates, and group sizes across all 10 bins.
7. **Interactive Plotly Visualizations ([`causal_plots.py`](causal_plots.py)):**
   - Publication-quality interactive charts embedded in Streamlit:
     - Cumulative Qini Gain Curve with shaded AUQC net benefit.
     - Cumulative Uplift Curve with peak-benefit threshold.
     - Targeting Decile Analysis (observed empirical lift vs. model predicted CATE).
     - Individual Treatment Effect (ITE) distribution with mean/median markers.
     - Observational Positivity & Overlap diagnostics.
8. **Descriptive Targeting Simulator (Without Budget Optimization):**
   - Interactive slider allowing users to inspect the composition and expected incremental buyers of the top $K\%$ targeted cohorts without allocating enterprise financial budgets.

---

## 3. Repository Directory Structure

```text
EconoCausal/
├── .gitignore                     # Git ignore rules (.venv, data, cache)
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Pinned dependencies
├── app.py                         # Interactive Streamlit dashboard application
├── causal_dml.py                  # EconML DML training & out-of-sample prediction
├── causal_eval.py                 # Qini curves, uplift curves & IPW metrics engine
├── causal_plots.py                # Interactive Plotly figures
├── data/
│   └── retail_causal_data.csv     # 5,000 synthetic retail records
├── notebooks/
│   ├── 01_synthetic_data_generation.ipynb # Data generation & EDA
│   └── 02_causal_dag_dowhy.ipynb          # DAG, DoWhy identification & refutation
└── tests/
    ├── test_week1.py              # Week 1 test suite (5 passing tests)
    └── test_uplift.py             # Week 2 test suite (7 passing tests)
```

---

## 4. Environment Setup & Installation

### Requirements
- **Operating System:** Windows, macOS, or Linux
- **Python Version:** Python 3.11 (tested with Python 3.11.9)

### Step 1: Create Virtual Environment

```powershell
# Windows PowerShell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

*(On macOS / Linux: `python3.11 -m venv .venv && source .venv/bin/activate`)*

### Step 2: Install Required Dependencies

```powershell
python -m pip install -r requirements.txt
```

---

## 5. How to Run

### A. Launch the Streamlit Dashboard

Run from the project root:

```powershell
streamlit run app.py
```

The application will start on `http://localhost:8501`.

**Dashboard Modules:**
1. **Historical Campaign Data:** Upload custom CSV files or load the default 5,000-record dataset. Run 10-point schema verification.
2. **Budget Constraints:** Enter total marketing budget, number of target campaign tiers, and minimum/maximum allocation boundaries.
3. **Causal Architecture:** Inspect the NetworkX acyclic DAG, backdoor adjustment sets, and identification assumptions.
4. **Double ML & Uplift Evaluation (Week 2):**
   - Train `LinearDML` or `CausalForestDML` on training split.
   - Evaluate out-of-sample ITE predictions strictly on held-out test data.
   - Inspect interactive Plotly Qini curves, cumulative uplift curves, decile calibration, and propensity overlap.
   - Run the descriptive targeting simulator across population fractions.

### B. Run the Automated Test Suites

Execute all unit and integration tests across Week 1 and Week 2:

```powershell
pytest tests/ --rootdir=. -v
```

All 12 tests will execute and pass:
- 5 tests in [`tests/test_week1.py`](tests/test_week1.py) (Schema integrity, budget feasibility, DAG acyclicity, DoWhy backdoor & refutation).
- 7 tests in [`tests/test_uplift.py`](tests/test_uplift.py) (Qini score math on controlled toy data, inverted ranking validation, cumulative uplift, decile partitioning, propensity clipping, DML train/eval disjoint splits, Plotly figure creation).

---

## 6. Causal Methodology & Uplift Evaluation Formulas

### A. The Double Machine Learning (DML) Estimand
Standard supervised models predict $\mathbb{E}[Y \mid T, X]$, which is confounded in observational data. DML (Robinson, 1988; Chernozhukov et al., 2018) uses Neyman-orthogonal score functions:
$$\tilde{Y} = Y - \hat{m}(X, W)$$
$$\tilde{T} = T - \hat{e}(X, W)$$
$$\tilde{Y} = \tau(X) \cdot \tilde{T} + \varepsilon$$
By training separate machine learning models ($\hat{m}$ and $\hat{e}$) via $K$-fold cross-fitting, DML removes regularization bias and isolates consistent estimates of $\tau(X)$.

### B. Radcliffe Qini Curve Formula
For a held-out evaluation sample of size $N$ ranked in descending order of predicted CATE ($\hat{\tau}_{(1)} \ge \hat{\tau}_{(2)} \ge \dots$):
$$Q(k) = Y_t(k) - Y_c(k) \times \frac{N_t^{total}}{N_c^{total}}$$
where:
- $N_t(k)$ and $N_c(k)$ are the cumulative numbers of treated and control observations in the top $k$.
- $Y_t(k)$ and $Y_c(k)$ are the cumulative positive outcomes (conversions) in the top $k$.
- $\frac{N_t^{total}}{N_c^{total}}$ normalizes the control group to match the treated group size.

### C. Random Targeting Baseline & Qini Score
The expected gain of random targeting follows a linear diagonal connecting $(0, 0)$ to the total sample Qini gain $(N, Q(N))$:
$$\text{Baseline}(k) = \frac{k}{N} \times Q(N)$$
The **Net Qini Score** is the difference in areas under the curves:
$$\text{Qini Score} = \text{AUQC}(\text{Model}) - \text{AURC}(\text{Random})$$
A positive score indicates that the causal model prioritizes high-lift customers more effectively than random assignment.

### D. Confounder Adjustment in Observational Data (IPW)
When treatment assignment is confounded in observational data, raw empirical differences can be adjusted using Inverse Propensity Weighting (IPW):
$$Y_i^* = \frac{T_i Y_i}{\hat{e}(W_i)} - \frac{(1 - T_i) Y_i}{1 - \hat{e}(W_i)}$$
The cumulative sum $\sum_{i=1}^k Y_i^*$ provides an asymptotically unconfounded estimate of causal gain under Conditional Ignorability, Positivity, and SUTVA.

---

## 7. Scope Boundaries

- **Week 2 Scope:** Completed. Out-of-sample CATE estimation, Qini gain curves, cumulative uplift curves, decile calibration, and interactive Plotly visualization.
- **Explicit Separation:** Business budget optimization (SciPy constrained solver, minimum/maximum tier allocation logic, and campaign portfolio selection) is strictly deferred to **Week 3 / Milestone 3**.
