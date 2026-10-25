# EconoCausal

**Dynamic Pricing via Double Machine Learning (Economics & Causal AI)**

EconoCausal is a causal AI project designed to optimize enterprise promotional and discount allocations. Rather than naively predicting which customers are likely to purchase, EconoCausal isolates the **true counterfactual uplift** (the causal effect of a price discount on customer conversion probability) while respecting financial budget caps.

---

## 1. Project Overview & Current Status

In observational retail data, promotional discounts are systematically confounded: churning customers and highly price-sensitive shoppers are targeted with larger discounts, while affluent loyalists are charged full price. A naive regression conflates treatment effects with baseline behavioral differences.

- **Current Milestone:** **Week 1 — Causal Graphing, DoWhy Identification & Streamlit Scaffolding** (Complete)
- **Next Milestone:** **Week 2 — EconML Double Machine Learning, ITE Estimation, Uplift Curves & Constrained Budget Optimization**

---

## 2. Week 1 Implemented Features

1. **Synthetic Retail Dataset (5,000 Customers):**
   - Grounded in realistic behavioral distributions with observable confounding and zero-hurdle discount policies.
   - Stored in [`data/retail_causal_data.csv`](data/retail_causal_data.csv).
2. **Structural Causal Directed Acyclic Graph (DAG):**
   - 9 nodes and 19 directed edges formalizing physical retail causal relationships in DOT syntax.
   - Programmatically validated for strict acyclicity using NetworkX.
   - Publication-grade visualization with categorical color coding.
3. **Microsoft DoWhy Causal Inference Engine:**
   - Formal initialization of `dowhy.CausalModel` on the full retail dataset.
   - Non-parametric causal identification via Pearl's **Backdoor Criterion**.
   - Average Treatment Effect (ATE) estimation via linear regression adjustment ($\hat{\beta} \approx 0.5957$).
   - Rigorous falsification testing using **Random Common Cause** ($p = 0.94$) and **Data Subset** refutation (stability on 80% subsample).
   - Fully executed and documented in [`notebooks/02_causal_dag_dowhy.ipynb`](notebooks/02_causal_dag_dowhy.ipynb).
4. **Interactive Streamlit Dashboard:**
   - Built with pure Python and Streamlit ([`app.py`](app.py)).
   - Ingestion and parsing of historical campaign/retail CSV datasets with live preview.
   - Automated 10-point schema, missing value, and duplicate row auditing.
   - Enterprise promotional budget constraint configuration with real-time mathematical feasibility validation.
   - Formal causal architecture and identification assumption reference view.
5. **Project-Local Virtual Environment & Automated Testing:**
   - Python 3.11 `.venv` configuration.
   - Fully reproducible [`requirements.txt`](requirements.txt).
   - Automated test suite in [`tests/test_week1.py`](tests/test_week1.py) (5/5 tests passing).

---

## 3. Repository Directory Structure

```text
EconoCausal/
├── .gitignore                     # Git ignore rules (.venv, data, cache)
├── README.md                      # Architecture and operational documentation
├── requirements.txt               # Reproducible pinned dependencies
├── app.py                         # Streamlit dashboard application
├── data/
│   └── retail_causal_data.csv     # 5,000 synthetic retail records
├── notebooks/
│   ├── 01_synthetic_data_generation.ipynb # Data generation & EDA
│   └── 02_causal_dag_dowhy.ipynb          # DAG, DoWhy identification & refutation
└── tests/
    └── test_week1.py              # Automated test suite (Pytest)
```

---

## 4. Environment Setup & Installation

### Requirements
- **Operating System:** Windows, macOS, or Linux
- **Python Version:** Python 3.11 (tested with Python 3.11.9)

### Step 1: Create Project-Local Virtual Environment

From the project root:

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
**Dashboard Capabilities:**
- **Historical Campaign Data:** Upload external CSV datasets or click *"📥 Load Default Synthetic Dataset"* to audit `retail_causal_data.csv`.
- **Budget Constraints:** Enter total marketing budget, number of target campaign tiers, and minimum/maximum budget limits per tier. Mathematical feasibility is validated dynamically.
- **Causal Architecture:** Inspect feature taxonomies and mathematical identification assumptions.

### B. Run the DoWhy Causal Inference Notebook

Launch JupyterLab or VS Code notebook interface:

```powershell
jupyter lab notebooks/02_causal_dag_dowhy.ipynb
```

All 9 cells are pre-executed with outputs saved, demonstrating:
1. Data loading and validation.
2. NetworkX acyclicity verification (`nx.is_directed_acyclic_graph`).
3. Matplotlib causal DAG visual rendering.
4. DoWhy `CausalModel` backdoor identification.
5. Linear regression adjustment estimation.
6. Random common cause & data subset refutation tests.

### C. Run the Test Suite

Execute the unit and integration tests:

```powershell
pytest tests/test_week1.py -v
```

---

## 6. Causal Feature Taxonomy & Dataset Schema

The expected schema comprises 10 variables:

| Column Name | Type | Causal Taxonomy Role | Business Meaning |
| :--- | :--- | :--- | :--- |
| `customer_id` | String | **Identifier** | Non-causal entity primary key |
| `tenure_days` | Numeric | **Confounder ($W$)** | Account age with retailer in days |
| `previous_orders` | Numeric | **Confounder ($W$)** | Cumulative purchase count prior to campaign |
| `days_since_last_order` | Numeric | **Confounder ($W$)** | Inactivity / recency in days (churn risk) |
| `avg_order_value` | Numeric | **Confounder ($W$)** | Historical average basket size in USD |
| `prior_email_engagement` | Numeric | **Confounder ($W$)** | Marketing email engagement score $[0.0, 1.0]$ |
| `price_sensitivity` | Numeric | **Confounder & Modifier** | Price elasticity psychographic score $[0.0, 1.0]$ |
| `discount_percent` | Numeric | **Treatment ($T$)** | Continuous promotional discount $[0.00, 0.30]$ |
| `purchased` | Numeric | **Primary Outcome ($Y$)** | Binary conversion indicator ($0 = \text{No}, 1 = \text{Yes}$) |
| `revenue` | Numeric | **Downstream Metric** | Post-treatment gross revenue (do not adjust!) |

---

## 7. Causal Inference Workflow & Identification

### Identification Assumptions
1. **Conditional Unconfoundedness:** $(Y(t) \perp T) \mid \mathbf{W} \quad \forall t \in [0.0, 0.30]$
2. **Positivity (Overlap):** $0 < P(T \in \mathcal{A} \mid \mathbf{W} = \mathbf{w}) < 1 \quad \forall \mathbf{w}, \mathcal{A}$
3. **SUTVA:** No interference between customer conversions; uniform discount delivery.
4. **DAG Correctness:** All backdoor non-causal paths are blocked by adjusting for the 6 observed pre-treatment confounders.

### Empirical Estimation & Refutation Results
- **Backdoor Adjustment Set:** `['avg_order_value', 'days_since_last_order', 'previous_orders', 'price_sensitivity', 'prior_email_engagement', 'tenure_days']`
- **Estimated ATE Slope:** $\hat{\beta} = 0.5957$ ($p < 0.001$). A 10 percentage point increase in discount yields an estimated $+5.96\%$ lift in average conversion probability.
- **Random Common Cause Refuter:** New effect $= 0.5959$ ($p = 0.94$). Invariant to independent noise confounders.
- **Data Subset Refuter (80%):** New effect $= 0.6526$ ($p = 0.00$). Statistically stable across customer sub-cohorts.

---

## 8. Explicit Scope Boundaries (Deferred to Week 2)

The following components are strictly planned for **Week 2** and are not part of Week 1:
- EconML Double Machine Learning (`LinearDML`, `CausalForestDML`).
- Random Forest and LightGBM nuisance models for treatment and outcome orthogonalization.
- Customer-level Individual Treatment Effect (ITE / CATE) estimation.
- Uplift curves and Qini coefficient benchmark curves.
- Comparison against random rollout targeting baselines.
- Constrained discount optimization algorithms using SciPy under the enterprise marketing budget.
