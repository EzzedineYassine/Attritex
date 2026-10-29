# ChurnSense — AI-Powered Customer Retention & Churn Intelligence

ChurnSense is an end-to-end machine learning system designed to identify subscription customers at risk of churn and equip commercial retention teams with defensible, explainable risk intelligence.

Rather than presenting an isolated trained model in a notebook, ChurnSense implements a production-grade modular monolith: reproducible scikit-learn preprocessing and estimation pipelines, a typed FastAPI inference service, and a responsive React/TypeScript analytics interface.

---

## 1. Problem Statement & Business Objective

Subscription businesses face substantial revenue attrition when accounts disengage silently. Traditional retention approaches suffer from three core deficiencies:

1. **Arbitrary decision thresholds**: Default 50% probability cutoffs misalign with business economics, under-recalling true churners when base rates are skewed.
2. **Black-box predictions**: Business managers cannot act on a bare probability score without understanding which contractual or service attributes drove the risk.
3. **Data leakage and pipeline divergence**: Transformations applied during offline experimentation frequently drift from real-time API inference.

ChurnSense solves these challenges by combining:
- **Calibrated probabilistic classification** with systematic threshold optimization (F1-optimal decision threshold and target-recall screening tiers).
- **Localized counterfactual explainability** showing exact risk point shifts against empirical training medians.
- **Unified pipeline architecture** ensuring identical imputation, scaling, and one-hot encoding across training and serving.

---

## 2. System Architecture

```
[ Customer Data (IBM Telco 7,043 rows) ]
                 │
                 ▼
[ Validation & Normalization Engine (churnsense.data.validation) ]
                 │
                 ├──► Model Training Pipeline (Stratified 5-Fold CV)
                 │    ├── Baseline: Class Prior Dummy
                 │    ├── Baseline: Logistic Regression (L2)
                 │    ├── Comparison: Logistic Regression (Balanced)
                 │    ├── Candidate: HistGradientBoosting
                 │    └── Candidate: Random Forest (Bagged)
                 │            │
                 │            ▼  (1-Standard-Error Parsimony Rule)
                 │    [ Selected Model: Logistic Regression ]
                 │            │
                 │            ▼
                 │    [ Evaluation & Permutation Importance ]
                 │    [ Holdout Benchmark: ROC-AUC 0.8421, PR-AUC 0.6342 ]
                 │            │
                 │            ▼
                 ├──► [ Model Artifacts: model.joblib, metadata.json, evaluation.json ]
                 │
                 ▼
[ FastAPI Inference Engine (churnsense.api) ]
  ├── GET  /api/health
  ├── GET  /api/model
  ├── GET  /api/model/evaluation
  ├── POST /api/predict (Single customer + counterfactuals)
  ├── POST /api/predict/batch (CSV ingestion + histogram + downloadable scored CSV)
  └── GET  /api/predict/batch/template
                 │
                 ▼  (HTTP / JSON / CSV)
[ React 19 + TypeScript + Vite + Tailwind + Recharts ]
  ├── Customer Risk Profiler (Interactive form, presets, gauge, drivers)
  ├── Batch Analysis (Dropzone, tier metrics, risk histogram, data table)
  ├── Model Evaluation Suite (ROC, PR, Threshold, Calibration, Lift, Matrix)
  └── Governance & Provenance Audit (Cryptographic hash, data dictionary)
```

---

## 3. Technology Stack

- **ML Pipeline & Statistics**: Python 3.11, scikit-learn 1.9, pandas 2.2, numpy 1.26, joblib 1.4
- **API & Validation**: FastAPI 0.115, Pydantic v2, Uvicorn, Python-Multipart
- **Frontend Application**: React 19, TypeScript 5.7, Vite 6.1, Tailwind CSS 3.4, Recharts 3.7, Lucide React
- **Packaging & Testing**: uv 0.12, pytest 8.2, ruff 0.17
- **Deployment**: Docker, Docker Compose, Nginx Alpine

---

## 4. Dataset Provenance & Data Dictionary

### Provenance Audit

- **Dataset Name**: IBM Telco Customer Churn
- **Repository**: [https://github.com/IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d)
- **License**: Apache-2.0
- **Total Records**: 7,043 accounts
- **Observed Churn Rate**: 26.54% (1,869 churned, 5,174 retained)
- **SHA-256 Checksum**: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`
- **Data Integrity**: Cryptographically checked on ingestion; duplicate IDs removed; missing total charges for first-month customers imputed via tenure * monthly charges.

### Ethical Feature Exclusion Policy

- **`gender`**: Excluded from model features by policy. Excluding protected demographic attributes prevents demographic stereotyping, algorithmic bias, and compliance violations under fair lending and non-discrimination frameworks.

### Data Dictionary

| Field Name | Type | Group | Allowed Values / Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `customerID` | String | Identifier | Unique string | Account reference key |
| `SeniorCitizen` | Categorical | Demographics | `0` (No), `1` (Yes) | Customer aged 65 or older |
| `Partner` | Categorical | Demographics | `No`, `Yes` | Whether customer has a partner |
| `Dependents` | Categorical | Demographics | `No`, `Yes` | Whether customer has dependents |
| `tenure` | Numeric (int) | Account | `0` to `120` months | Months with the company |
| `PhoneService` | Categorical | Services | `No`, `Yes` | Subscribed to landline phone |
| `MultipleLines` | Categorical | Services | `No`, `Yes`, `No phone service` | Multiple telephone connections |
| `InternetService` | Categorical | Services | `DSL`, `Fiber optic`, `No` | Underlying internet technology |
| `OnlineSecurity` | Categorical | Services | `No`, `Yes`, `No internet service` | Cyber protection add-on |
| `OnlineBackup` | Categorical | Services | `No`, `Yes`, `No internet service` | Cloud storage add-on |
| `DeviceProtection` | Categorical | Services | `No`, `Yes`, `No internet service` | Hardware warranty add-on |
| `TechSupport` | Categorical | Services | `No`, `Yes`, `No internet service` | Dedicated support add-on |
| `StreamingTV` | Categorical | Services | `No`, `Yes`, `No internet service` | Television streaming tier |
| `StreamingMovies` | Categorical | Services | `No`, `Yes`, `No internet service` | Movie streaming tier |
| `Contract` | Categorical | Account | `Month-to-month`, `One year`, `Two year` | Current commitment term |
| `PaperlessBilling` | Categorical | Account | `No`, `Yes` | Electronic statements |
| `PaymentMethod` | Categorical | Account | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` | Payment instrument |
| `MonthlyCharges` | Numeric | Billing | `0.00` to `1,000.00` USD | Current monthly invoice |
| `TotalCharges` | Numeric | Billing | `0.00` to `100,000.00` USD | Cumulative billed amount |
| `Churn` | Target | Outcome | `No` (0), `Yes` (1) | Historical account churn |

---

## 5. Machine Learning Methodology

### Preprocessing & Transformation

- **Leakage Prevention**: All transformations fit exclusively on training folds; test and inference data receive out-of-fold statistics.
- **Numeric Pipeline**: Median imputation followed by standard scaling (`z = (x - μ) / σ`).
- **Categorical Pipeline**: Mode imputation followed by dense one-hot encoding (`handle_unknown="ignore"`).
- **Domain Transformation**: `TotalChargesImputer` automatically derives `TotalCharges = tenure * MonthlyCharges` for first-month customers where cumulative billing is not yet populated.

### Candidate Model Comparison & Selection

All candidates were evaluated using stratified 5-fold cross-validation on 80% training data (5,634 rows):

| Candidate Model | Role | Complexity | 5-Fold CV PR-AUC | CV Std Err | Holdout Test PR-AUC | Holdout ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Prior Baseline** | Reference | 0 | 0.2654 | ±0.0000 | 0.2654 | 0.5000 |
| **Logistic Regression (L2)** | **Baseline** | **1** | **0.6548** | **±0.0089** | **0.6342** | **0.8421** |
| Logistic Regression (Balanced) | Comparison | 1 | 0.6521 | ±0.0091 | 0.6315 | 0.8418 |
| HistGradientBoosting | Candidate | 2 | 0.6512 | ±0.0094 | 0.6308 | 0.8389 |
| Random Forest | Candidate | 3 | 0.6487 | ±0.0102 | 0.6274 | 0.8362 |

#### Selection Rule: 1-Standard-Error Parsimony Principle

Logistic Regression was chosen as the production model because:
1. It achieves cross-validated PR-AUC within one standard error of the highest-scoring candidate.
2. It offers strict probability calibration without gradient overfitting.
3. Its coefficients guarantee monotonic, transparent feature effects suitable for enterprise governance.

### Empirical Holdout Evaluation (1,409 Test Accounts)

- **ROC-AUC**: **0.8421** (95% CI: `[0.8194, 0.8651]`)
- **PR-AUC**: **0.6342** (95% CI: `[0.5841, 0.6865]`)
- **Brier Score Loss**: **0.1380** (indicating well-calibrated probabilities)
- **Decision Threshold**: **0.3262** (derived from F1 optimization on training out-of-fold scores)
- **Holdout Precision**: **53.98%**
- **Holdout Recall**: **72.46%**
- **Holdout F1 Score**: **0.6187** (95% CI: `[0.5732, 0.6624]`)
- **Medium Risk Screening Threshold**: **0.1689** (capturing 90% of all churners)

---

## 6. Installation & Execution Guide

### Prerequisites

- Python 3.11+
- Node.js 18+
- [uv](https://docs.astral.sh/uv/) (modern Python package manager)

### Quick Start (Local)

#### 1. Setup Backend & Train Model

```powershell
# Navigate to backend directory
cd backend

# Install dependencies into virtual environment
uv sync

# Download official IBM Telco dataset (cryptographically checked)
uv run churnsense data download

# Train pipeline, select candidate, evaluate holdout, and save artifacts
uv run churnsense train

# Inspect saved model version and metrics
uv run churnsense info

# Run all 31 automated tests
uv run pytest

# Launch FastAPI prediction server on port 8000
uv run churnsense serve --port 8000
```

#### 2. Setup & Run Frontend

```powershell
# In a separate terminal, navigate to frontend directory
cd frontend

# Install npm packages
npm install

# Run TypeScript build check
npm run build

# Start Vite development server
npm run dev
```

Visit `http://localhost:5173` to explore the interactive interface.

---

### Docker Execution

To run the complete system with Docker Compose:

```bash
docker compose up --build
```

- Web UI: `http://localhost:3000`
- API Docs: `http://localhost:8000/docs`

---

## 7. Automated Test Suite

The test suite covers data validation, preprocessing, artifact integrity, inference, and HTTP API endpoints:

```powershell
cd backend
uv run pytest -v
```

Test coverage includes:
- **`test_preprocessing.py`**: `TotalChargesImputer`, column transformer shapes, pipeline end-to-end training and inference.
- **`test_validation.py`**: Schema adherence, numerical bounds, integer constraints, cross-service consistency rules (e.g. `InternetService="No"` vs add-ons), target normalization.
- **`test_thresholds.py`**: F1-optimal threshold calculation, recall target threshold derivation, risk tier assignments.
- **`test_artifacts.py`**: Atomic artifact storage, schema version compatibility enforcement, missing file handling.
- **`test_inference.py`**: Single customer scoring, counterfactual delta computation, batch scoring, error classification.
- **`test_api.py`**: `/api/health`, `/api/model`, `/api/model/evaluation`, `/api/predict`, `/api/predict/batch` (JSON and CSV streaming formats), `/api/predict/batch/template`, 422 input validation errors, 503 degraded states.

---

## 8. Operational Boundaries & Limitations

1. **Correlation vs Causation**: Risk factor explanations show how model estimates shift when an attribute is swapped with the reference median. They reflect model behavior on observational historical data and do not constitute randomized clinical trial evidence of causation.
2. **Population Drift**: Trained on 7,043 telecommunications accounts. Applying ChurnSense to other domains (e.g., enterprise B2B SaaS, mobile gaming) requires retraining with domain-specific behavioral features.
3. **Macroeconomic Shocks**: Unobserved market shifts (e.g. competitor pricing reductions or regional outages) are not captured in customer contract records and necessitate periodic model re-calibration.
