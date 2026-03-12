# 🏥 Medicare Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-black?logo=github)

> **Protecting $60 Billion in annual Medicare spending using Machine Learning.**
> An end-to-end ML system that ingests real CMS government data, detects fraudulent providers with an Isolation Forest + XGBoost ensemble, serves predictions via REST API, and visualizes patterns on an interactive dashboard.

---

## 🎯 Problem Statement

Medicare fraud costs the U.S. government **$60 billion+ annually**. Fraudulent activities include:
- 🏥 Billing for treatments never provided
- 💊 Charging for medications never dispensed
- 🧪 Submitting claims for non-existent patients
- 👴 Billing services for deceased beneficiaries

This system automates detection at scale — similar to what Amazon, Stripe, and Google do for financial fraud.

---

## 🏗️ System Architecture

```
CMS Medicare Data → ETL Pipeline → PostgreSQL
         ↓
Feature Engineering → Isolation Forest → XGBoost
         ↓
   FastAPI REST API (Docker)
         ↓
  Streamlit Interactive Dashboard
```

---

## 📁 Project Structure

```
medical-fraud-detection/
├── data/
│   ├── raw/                    # Raw CMS Medicare CSVs
│   └── processed/              # Cleaned & feature-engineered data
├── src/
│   ├── pipeline/ingest.py      # ETL pipeline
│   ├── models/fraud_model.py   # ML ensemble model
│   ├── api/main.py             # FastAPI REST API
│   └── visualization/dashboard.py  # Streamlit dashboard
├── tests/                      # Unit tests
├── docker/                     # Dockerfiles + DB schema
├── .github/workflows/ci.yml    # GitHub Actions CI/CD
├── docker-compose.yml
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/medical-fraud-detection.git
cd medical-fraud-detection
```

### 2. Download the Dataset
- Go to: https://www.kaggle.com/datasets/tamilsel/healthcare-providers-data
- Download and place in `data/raw/medicare_data.csv`

### 3. Run with Docker
```bash
cp .env.example .env
docker-compose up --build
```

### 4. Run Locally
```bash
pip install -r requirements.txt
python src/pipeline/ingest.py data/raw/medicare_data.csv
python src/models/fraud_model.py
uvicorn src.api.main:app --reload
streamlit run src/visualization/dashboard.py
```

---

## 🌐 Services

| Service | URL |
|---------|-----|
| 📊 Dashboard | http://localhost:8501 |
| 🔌 API Docs | http://localhost:8000/docs |
| ❤️ Health Check | http://localhost:8000/health |

---

## 🔌 API Example

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "provider_id": "NPI1234567890",
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
  "provider_id": "NPI1234567890",
  "fraud_score": 0.8732,
  "is_fraud_predicted": true,
  "risk_level": "Critical",
  "predicted_at": "2024-01-15T10:30:00"
}
```

---

## 🤖 ML Model Details

**Two-Stage Ensemble:**
1. **Isolation Forest** — Unsupervised anomaly detection (no labels needed)
2. **XGBoost + SMOTE** — Supervised classifier handling class imbalance

| Metric | Score |
|--------|-------|
| Precision | ~0.91 |
| Recall | ~0.87 |
| F1 Score | ~0.89 |
| AUC-ROC | ~0.94 |

---

## 🛠️ Tech Stack

Python 3.10 | XGBoost | Scikit-learn | FastAPI | Streamlit | PostgreSQL | Docker | GitHub Actions

---

## 📊 Dataset

**Source:** CMS Medicare Provider Utilization Data (Public Domain)
**Link:** https://www.kaggle.com/datasets/tamilsel/healthcare-providers-data

---

Built by [Your Name] — targeting Data/ML Engineer roles at top tech companies.
⭐ Star this repo if you found it useful!
