-- Medicare Fraud Detection Database Schema

CREATE TABLE IF NOT EXISTS providers (
    provider_id VARCHAR(20) PRIMARY KEY,
    provider_type VARCHAR(100),
    state VARCHAR(5),
    total_claims INTEGER,
    total_beneficiaries INTEGER,
    total_services INTEGER,
    total_charges DECIMAL(15,2),
    avg_charge_per_claim DECIMAL(10,2),
    avg_medicare_payment DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id SERIAL PRIMARY KEY,
    provider_id VARCHAR(20) REFERENCES providers(provider_id),
    beneficiary_id VARCHAR(20),
    claim_date DATE,
    diagnosis_code VARCHAR(10),
    procedure_code VARCHAR(10),
    claim_amount DECIMAL(10,2),
    medicare_paid DECIMAL(10,2),
    is_fraud BOOLEAN DEFAULT FALSE,
    fraud_score DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fraud_predictions (
    prediction_id SERIAL PRIMARY KEY,
    provider_id VARCHAR(20),
    fraud_score DECIMAL(5,4),
    is_fraud_predicted BOOLEAN,
    model_version VARCHAR(20),
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_metrics (
    metric_id SERIAL PRIMARY KEY,
    model_version VARCHAR(20),
    precision_score DECIMAL(5,4),
    recall_score DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    auc_roc DECIMAL(5,4),
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_claims_provider ON claims(provider_id);
CREATE INDEX IF NOT EXISTS idx_claims_date ON claims(claim_date);
CREATE INDEX IF NOT EXISTS idx_predictions_provider ON fraud_predictions(provider_id);
