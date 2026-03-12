"""
FastAPI REST API for Medicare Fraud Detection
Endpoints: predict, batch predict, model info, health check
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.models.fraud_model import FraudDetectionModel, FEATURE_COLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="🏥 Medicare Fraud Detection API",
    description="Detects fraudulent Medicare claims using ML — built for FAANG portfolio",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
model: Optional[FraudDetectionModel] = None
MODEL_PATH = os.getenv("MODEL_PATH", "src/models/saved/fraud_model_v1.pkl")


@app.on_event("startup")
async def load_model():
    global model
    if Path(MODEL_PATH).exists():
        model = FraudDetectionModel.load(MODEL_PATH)
        logger.info("✅ Model loaded successfully")
    else:
        logger.warning("⚠️ No trained model found. Train model first.")


# ── Schemas ──────────────────────────────────────────────────────────────────

class ProviderFeatures(BaseModel):
    provider_id: str = Field(..., example="1234567890")
    number_of_services: float = Field(..., example=500)
    number_of_medicare_beneficiaries: float = Field(..., example=120)
    average_submitted_charge_amount: float = Field(..., example=1500.0)
    average_medicare_payment_amount: float = Field(..., example=800.0)
    average_medicare_standardized_amount: float = Field(..., example=750.0)
    charge_to_payment_ratio: Optional[float] = None
    services_per_beneficiary: Optional[float] = None
    payment_deviation: Optional[float] = None
    is_high_volume: Optional[int] = 0
    is_abnormal_charge: Optional[int] = 0
    standardized_deviation: Optional[float] = None


class PredictionResponse(BaseModel):
    provider_id: str
    fraud_score: float
    is_fraud_predicted: bool
    risk_level: str
    predicted_at: str


class BatchPredictionRequest(BaseModel):
    providers: List[ProviderFeatures]


class ModelInfo(BaseModel):
    model_version: str
    features: List[str]
    status: str


# ── Helper ───────────────────────────────────────────────────────────────────

def compute_derived_features(data: dict) -> dict:
    """Compute derived features if not provided."""
    if not data.get("charge_to_payment_ratio"):
        payment = data["average_medicare_payment_amount"]
        data["charge_to_payment_ratio"] = (
            data["average_submitted_charge_amount"] / payment if payment > 0 else 0
        )
    if not data.get("services_per_beneficiary"):
        bene = data["number_of_medicare_beneficiaries"]
        data["services_per_beneficiary"] = (
            data["number_of_services"] / bene if bene > 0 else 0
        )
    if not data.get("standardized_deviation"):
        data["standardized_deviation"] = abs(
            data["average_medicare_payment_amount"] -
            data["average_medicare_standardized_amount"]
        )
    if data.get("payment_deviation") is None:
        data["payment_deviation"] = 0.0
    return data


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "message": "🏥 Medicare Fraud Detection API",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
def model_info():
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfo(
        model_version="v1",
        features=FEATURE_COLS,
        status="active"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(provider: ProviderFeatures):
    """Predict fraud probability for a single provider."""
    if not model:
        raise HTTPException(status_code=503, detail="Model not trained yet")

    data = provider.dict()
    provider_id = data.pop("provider_id")
    data = compute_derived_features(data)

    df = pd.DataFrame([data])
    results = model.predict(df)

    return PredictionResponse(
        provider_id=provider_id,
        fraud_score=round(float(results["fraud_score"].iloc[0]), 4),
        is_fraud_predicted=bool(results["is_fraud_predicted"].iloc[0]),
        risk_level=str(results["risk_level"].iloc[0]),
        predicted_at=datetime.utcnow().isoformat()
    )


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(request: BatchPredictionRequest):
    """Predict fraud for multiple providers at once."""
    if not model:
        raise HTTPException(status_code=503, detail="Model not trained yet")

    rows = []
    provider_ids = []
    for p in request.providers:
        data = p.dict()
        provider_ids.append(data.pop("provider_id"))
        rows.append(compute_derived_features(data))

    df = pd.DataFrame(rows)
    results = model.predict(df)

    return {
        "total": len(provider_ids),
        "fraud_detected": int(results["is_fraud_predicted"].sum()),
        "predictions": [
            {
                "provider_id": pid,
                "fraud_score": round(float(results["fraud_score"].iloc[i]), 4),
                "is_fraud_predicted": bool(results["is_fraud_predicted"].iloc[i]),
                "risk_level": str(results["risk_level"].iloc[i]),
            }
            for i, pid in enumerate(provider_ids)
        ]
    }


@app.get("/analytics/summary", tags=["Analytics"])
def fraud_summary():
    """Return high-level fraud detection summary stats."""
    return {
        "message": "Load processed data and run analysis",
        "hint": "POST to /predict with provider data to get fraud scores"
    }
