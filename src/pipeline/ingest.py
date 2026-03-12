"""
Data Ingestion Pipeline
Downloads and ingests real CMS Medicare public data
Dataset: https://www.kaggle.com/datasets/tamilsel/healthcare-providers-data
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")


def get_db_engine():
    """Create PostgreSQL connection."""
    url = (
        f"postgresql://{os.getenv('POSTGRES_USER', 'frauduser')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'fraudpass')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'medicare_fraud')}"
    )
    return create_engine(url)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw CMS Medicare dataset."""
    logger.info(f"Loading raw data from {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize raw Medicare data."""
    logger.info("Starting data cleaning...")

    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("/", "_")

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {before - len(df):,} duplicate rows")

    # Handle numeric columns
    numeric_cols = [
        "number_of_services", "number_of_medicare_beneficiaries",
        "number_of_distinct_medicare_beneficiary_per_day_services",
        "average_medicare_allowed_amount", "average_submitted_charge_amount",
        "average_medicare_payment_amount", "average_medicare_standardized_amount"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Drop rows with critical nulls
    df = df.dropna(subset=["national_provider_identifier"])

    logger.info(f"Clean dataset: {len(df):,} records")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer fraud-detection features from raw claims data."""
    logger.info("Engineering features...")

    # Charge-to-payment ratio (high ratio = potential fraud)
    df["charge_to_payment_ratio"] = np.where(
        df["average_medicare_payment_amount"] > 0,
        df["average_submitted_charge_amount"] / df["average_medicare_payment_amount"],
        0
    )

    # Services per beneficiary (unusually high = fraud signal)
    df["services_per_beneficiary"] = np.where(
        df["number_of_medicare_beneficiaries"] > 0,
        df["number_of_services"] / df["number_of_medicare_beneficiaries"],
        0
    )

    # Payment deviation from specialty average
    specialty_avg = df.groupby("provider_type")["average_medicare_payment_amount"].transform("mean")
    df["payment_deviation"] = df["average_medicare_payment_amount"] - specialty_avg

    # High volume flag (top 5% by services)
    threshold = df["number_of_services"].quantile(0.95)
    df["is_high_volume"] = (df["number_of_services"] > threshold).astype(int)

    # Abnormal charge flag
    charge_mean = df["average_submitted_charge_amount"].mean()
    charge_std = df["average_submitted_charge_amount"].std()
    df["is_abnormal_charge"] = (
        df["average_submitted_charge_amount"] > charge_mean + 3 * charge_std
    ).astype(int)

    # Standardized amount deviation
    df["standardized_deviation"] = abs(
        df["average_medicare_payment_amount"] - df["average_medicare_standardized_amount"]
    )

    logger.info(f"Engineered {len(df.columns)} total features")
    return df


def save_to_postgres(df: pd.DataFrame, engine) -> None:
    """Save processed data to PostgreSQL."""
    logger.info("Saving to PostgreSQL...")

    provider_cols = {
        "national_provider_identifier": "provider_id",
        "provider_type": "provider_type",
        "nppes_provider_state": "state",
        "number_of_services": "total_claims",
        "number_of_medicare_beneficiaries": "total_beneficiaries",
        "number_of_distinct_medicare_beneficiary_per_day_services": "total_services",
        "average_submitted_charge_amount": "total_charges",
        "average_submitted_charge_amount": "avg_charge_per_claim",
        "average_medicare_payment_amount": "avg_medicare_payment",
    }

    available_cols = {k: v for k, v in provider_cols.items() if k in df.columns}
    providers_df = df[list(available_cols.keys())].rename(columns=available_cols)
    providers_df = providers_df.drop_duplicates(subset=["provider_id"])

    providers_df.to_sql("providers", engine, if_exists="replace", index=False, chunksize=1000)
    logger.info(f"Saved {len(providers_df):,} providers to database")


def save_processed_csv(df: pd.DataFrame) -> None:
    """Save processed data locally as CSV."""
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_PATH / "medicare_processed.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")


def run_pipeline(filepath: str) -> pd.DataFrame:
    """Run the full ingestion pipeline."""
    logger.info("=" * 50)
    logger.info("Starting Medicare Data Ingestion Pipeline")
    logger.info("=" * 50)

    df = load_raw_data(filepath)
    df = clean_data(df)
    df = engineer_features(df)
    save_processed_csv(df)

    try:
        engine = get_db_engine()
        save_to_postgres(df, engine)
    except Exception as e:
        logger.warning(f"Database save skipped: {e}. Data saved to CSV instead.")

    logger.info("✅ Pipeline complete!")
    return df


if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/raw/medicare_data.csv"
    run_pipeline(filepath)
