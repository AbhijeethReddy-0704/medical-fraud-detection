"""
Fraud Detection Model
Ensemble of Isolation Forest (anomaly detection) + XGBoost (classification)
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_SAVE_PATH = Path("src/models/saved")

FEATURE_COLS = [
    "number_of_services",
    "number_of_medicare_beneficiaries",
    "average_submitted_charge_amount",
    "average_medicare_payment_amount",
    "average_medicare_standardized_amount",
    "charge_to_payment_ratio",
    "services_per_beneficiary",
    "payment_deviation",
    "is_high_volume",
    "is_abnormal_charge",
    "standardized_deviation",
]


class FraudDetectionModel:
    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200
        )
        self.xgb_classifier = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=10,  # Handle class imbalance
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False
        )
        self.scaler = StandardScaler()
        self.feature_cols = FEATURE_COLS
        self.is_trained = False

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and scale features."""
        available = [c for c in self.feature_cols if c in df.columns]
        missing = set(self.feature_cols) - set(available)
        if missing:
            logger.warning(f"Missing features: {missing}. Filling with 0.")
            for col in missing:
                df[col] = 0
        return df[self.feature_cols].fillna(0).values

    def generate_fraud_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate pseudo fraud labels using Isolation Forest.
        In production, these would be actual labeled fraud cases.
        """
        logger.info("Generating fraud labels with Isolation Forest...")
        X = self.prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)

        # Isolation Forest: -1 = anomaly (fraud), 1 = normal
        if_labels = self.isolation_forest.fit_predict(X_scaled)
        fraud_labels = (if_labels == -1).astype(int)

        fraud_rate = fraud_labels.mean() * 100
        logger.info(f"Detected {fraud_rate:.2f}% fraud rate ({fraud_labels.sum():,} suspicious providers)")
        return fraud_labels

    def train(self, df: pd.DataFrame) -> dict:
        """Train the full fraud detection ensemble."""
        logger.info("=" * 50)
        logger.info("Training Fraud Detection Model")
        logger.info("=" * 50)

        X = self.prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)
        y = self.generate_fraud_labels(df)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # SMOTE to handle class imbalance
        logger.info("Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        logger.info(f"Balanced training set: {y_train_bal.sum():,} fraud / {(y_train_bal==0).sum():,} normal")

        # Train XGBoost
        logger.info("Training XGBoost classifier...")
        self.xgb_classifier.fit(
            X_train_bal, y_train_bal,
            eval_set=[(X_test, y_test)],
            verbose=50
        )

        # Evaluate
        y_pred = self.xgb_classifier.predict(X_test)
        y_proba = self.xgb_classifier.predict_proba(X_test)[:, 1]

        metrics = {
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "auc_roc": round(roc_auc_score(y_test, y_proba), 4),
            "fraud_detected": int(y_pred.sum()),
            "total_tested": len(y_test)
        }

        logger.info("\n" + "=" * 50)
        logger.info("MODEL PERFORMANCE")
        logger.info("=" * 50)
        logger.info(f"Precision : {metrics['precision']:.4f}")
        logger.info(f"Recall    : {metrics['recall']:.4f}")
        logger.info(f"F1 Score  : {metrics['f1']:.4f}")
        logger.info(f"AUC-ROC   : {metrics['auc_roc']:.4f}")
        logger.info(classification_report(y_test, y_pred))

        self.is_trained = True
        return metrics

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict fraud probability for providers."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)

        fraud_proba = self.xgb_classifier.predict_proba(X_scaled)[:, 1]
        fraud_pred = self.xgb_classifier.predict(X_scaled)

        results = df.copy()
        results["fraud_score"] = fraud_proba
        results["is_fraud_predicted"] = fraud_pred
        results["risk_level"] = pd.cut(
            fraud_proba,
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=["Low", "Medium", "High", "Critical"]
        )
        return results

    def save(self, version: str = "v1") -> str:
        """Save model artifacts."""
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_SAVE_PATH / f"fraud_model_{version}.pkl"
        joblib.dump({
            "xgb": self.xgb_classifier,
            "isolation_forest": self.isolation_forest,
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
            "version": version
        }, model_path)
        logger.info(f"✅ Model saved to {model_path}")
        return str(model_path)

    @classmethod
    def load(cls, model_path: str) -> "FraudDetectionModel":
        """Load saved model."""
        artifacts = joblib.load(model_path)
        model = cls()
        model.xgb_classifier = artifacts["xgb"]
        model.isolation_forest = artifacts["isolation_forest"]
        model.scaler = artifacts["scaler"]
        model.feature_cols = artifacts["feature_cols"]
        model.is_trained = True
        logger.info(f"✅ Model loaded from {model_path}")
        return model


def get_feature_importance(model: FraudDetectionModel) -> pd.DataFrame:
    """Get feature importance from XGBoost model."""
    importance = model.xgb_classifier.feature_importances_
    return pd.DataFrame({
        "feature": model.feature_cols,
        "importance": importance
    }).sort_values("importance", ascending=False)


if __name__ == "__main__":
    # Train model on processed data
    df = pd.read_csv("data/processed/medicare_processed.csv")
    logger.info(f"Training on {len(df):,} records")

    model = FraudDetectionModel(contamination=0.05)
    metrics = model.train(df)
    model.save("v1")

    # Show feature importance
    importance_df = get_feature_importance(model)
    logger.info("\nTop Features:")
    logger.info(importance_df.to_string())
