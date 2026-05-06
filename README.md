# Medicare Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-black?logo=github)

> An end-to-end anomaly detection system that ingests 100K+ Medicare provider records, validates and cleans the data through an automated ETL pipeline, detects fraudulent billing patterns using a two-stage ML ensemble (Isolation Forest + XGBoost), serves predictions via a REST API, and visualizes patterns on an interactive dashboard.

---

## Why This Project Matters Beyond Healthcare

The architecture in this project — automated ETL → validation → anomaly detection → API → dashboard — is domain-agnostic. The same pipeline pattern applies to:

- **Supply chain:** Detecting anomalies in supplier invoices, PO mismatches, or unexpected price variances
- **Finance & operations:** Flagging outliers in expense reports, reimbursements, or vendor payments
- **Inventory:** Identifying unusual variance between recorded and actual stock movements
- **Sales analytics:** Surfacing unusual order patterns or returns at scale

The business problem — *"find the few suspicious records hidden in millions of normal ones"* — is universal across enterprise data.

---

## Problem Statement

Medicare fraud costs the U.S. government over $60 billion annually through:

- Billing for treatments never provided
- Charging for medications never dispensed
- Submitting claims for non-existent or deceased patients
- Upcoding and unbundling of services

Manual auditing cannot scale to the volume of claims. This system automates risk-scoring of providers so investigators can focus their time on the highest-risk cases.

---

## System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  CMS Medicare   │────▶│   ETL Pipeline   │────▶│   PostgreSQL     │
│   Raw Data      │     │  (Validation +   │     │   (Normalized    │
│   (100K rows)   │     │   Cleansing)     │     │    Schema)       │
└─────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                          │
                                                          ▼
                        ┌──────────────────┐     ┌──────────────────┐
                        │    XGBoost +     │◀────│ Feature          │
                        │      SMOTE       │     │ Engineering      │
                        │   (Supervised)   │     │ + Isolation      │
                        └────────┬─────────┘     │ Forest           │
                                 │               │ (Unsupervised)   │
                                 ▼               └──────────────────┘
                        ┌──────────────────┐
                        │  FastAPI REST    │
                        │  (Dockerized)    │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │    Streamlit     │
                        │    Dashboard     │
                        └──────────────────┘
```

---

## Project Structure

```
medical-fraud-detection/
├── data/
│   ├── raw/                       # Raw CMS Medicare CSVs
│   └── processed/                 # Cleaned & feature-engineered data
├── src/
│   ├── pipeline/ingest.py         # ETL pipeline with validation
│   ├── models/fraud_model.py      # Two-stage ML ensemble
│   ├── api/main.py                # FastAPI REST endpoints
│   └── visualization/dashboard.py # Streamlit dashboard
├── tests/                         # Unit tests
├── docker/                        # Dockerfiles + DB schema
├── .github/workflows/ci.yml       # GitHub Actions CI/CD
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Data Validation & Quality Checks

Every record passing through the ETL is validated against:

| Check Type | Example |
|---|---|
| **Completeness** | Provider ID, service code, and charge amount must be non-null |
| **Type consistency** | Numeric fields parsed as numeric; dates conform to ISO 8601 |
| **Range validation** | Charge amounts ≥ 0; service counts within plausible bounds |
| **Format validation** | NPI provider IDs match the 10-digit NPI format |
| **Referential integrity** | Service codes must exist in the reference table before load |
| **Cross-field logic** | `medicare_payment ≤ submitted_charge` |

Records that fail validation are logged and quarantined into a separate table rather than silently dropped, preserving full data lineage for audit.

---

## Model Approach: Why Two Stages

Real-world fraud detection rarely has clean labels. A two-stage ensemble handles this:

1. **Stage 1 — Isolation Forest (Unsupervised):** Identifies statistical outliers without needing labels. Generates a soft anomaly score for every provider.
2. **Stage 2 — XGBoost + SMOTE (Supervised):** Refines the candidates from Stage 1. SMOTE addresses severe class imbalance — fraud is rare (~1–3% of records), and without it the model would optimize toward "predict everything is clean."

This pattern works in any domain where anomalies are rare, evolving, and only partially labeled.

### Performance (held-out test set)

| Metric | Score | What It Means |
|---|---|---|
| Recall | 0.999 | Of all actual fraud cases, the model catches 99.9% |
| Precision | 0.81 | Of all cases flagged as fraud, 81% are truly fraud |
| F1 Score | 0.89 | Balanced score combining Precision and Recall (0 = bad, 1 = perfect) |
| AUC-ROC | 0.9995 | How well the model ranks fraud above non-fraud across all thresholds (0.5 = random, 1.0 = perfect) |

Recall is prioritized: in fraud detection, missing a fraudulent
provider costs far more than flagging a clean one for human review.
The top two features driving predictions are billing volume (44%)
and abnormal charge patterns (43%).

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/AbhijeethReddy-0704/medical-fraud-detection.git
cd medical-fraud-detection
```

### 2. Download the dataset

Download from: <https://www.kaggle.com/datasets/tamilsel/healthcare-providers-data>

Place the CSV at `data/raw/medicare_data.csv`.

### 3. Run with Docker (recommended)

```bash
cp .env.example .env
docker-compose up --build
```

### 4. Run locally (without Docker)

```bash
pip install -r requirements.txt
python src/pipeline/ingest.py data/raw/medicare_data.csv
python src/models/fraud_model.py
uvicorn src.api.main:app --reload
streamlit run src/visualization/dashboard.py
```

---

## Services

| Service | URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI Docs (Swagger) | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |

---

## API Example

**Request:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "provider_id": "1234567890",
    "number_of_services": 500,
    "number_of_medicare_beneficiaries": 100,
    "average_submitted_charge_amount": 1500.0,
    "average_medicare_payment_amount": 800.0,
    "average_medicare_standardized_amount": 750.0
  }'
```

**Response:**

```json
{
  "provider_id": "1234567890",
  "fraud_score": 0.8732,
  "is_fraud_predicted": true,
  "risk_level": "Critical",
  "predicted_at": "2024-01-15T10:30:00"
}
```

---

## Tech Stack

**Languages & Libraries:** Python 3.10, Pandas, Scikit-learn, XGBoost, imbalanced-learn (SMOTE)
**API & Dashboard:** FastAPI, Uvicorn, Streamlit
**Database:** PostgreSQL
**Infrastructure:** Docker, Docker Compose, GitHub Actions (CI/CD)
**Testing:** Pytest

---

## Engineering Practices

- **Unit tests** under `/tests` covering ETL validation logic and model inference
- **CI/CD** via GitHub Actions running tests on every push
- **Containerization** with Docker Compose for one-command local setup
- **Configuration management** via `.env.example` (no secrets committed)
- **Logging** at every pipeline stage for traceability
- **Reproducibility** with pinned dependency versions in `requirements.txt`

---

## Dataset

**Source:** CMS Medicare Provider Utilization and Payment Data (Public Domain)
**Link:** <https://www.kaggle.com/datasets/tamilsel/healthcare-providers-data>
**Size:** 100,000+ provider records

---

## Author

**Abhijeeth Reddy Bhavanthula**
Data Analyst | SQL · Python · ETL · Data Modeling
· [GitHub](https://github.com/AbhijeethReddy-0704)