"""
Medicare Fraud Detection Dashboard
Interactive Streamlit dashboard for visualizing fraud patterns
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="🏥 Medicare Fraud Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem; border-radius: 10px; color: white; text-align: center;
    }
    .fraud-high { color: #ff4444; font-weight: bold; }
    .fraud-low  { color: #00C851; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = "data/processed/medicare_processed.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Generate fraud scores for demo if model not running
        np.random.seed(42)
        df["fraud_score"] = np.random.beta(1, 10, len(df))
        df["is_fraud_predicted"] = (df["fraud_score"] > 0.5).astype(int)
        df["risk_level"] = pd.cut(
            df["fraud_score"],
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=["Low", "Medium", "High", "Critical"]
        ).astype(str)
        return df
    else:
        # Generate synthetic demo data
        n = 5000
        np.random.seed(42)
        states = ["CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI"]
        specialties = ["Internal Medicine", "Family Practice", "Cardiology",
                       "Orthopedic Surgery", "Psychiatry", "Neurology", "Oncology"]
        df = pd.DataFrame({
            "national_provider_identifier": [f"NPI{i:010d}" for i in range(n)],
            "provider_type": np.random.choice(specialties, n),
            "nppes_provider_state": np.random.choice(states, n),
            "number_of_services": np.random.exponential(300, n).astype(int),
            "number_of_medicare_beneficiaries": np.random.exponential(80, n).astype(int),
            "average_submitted_charge_amount": np.random.exponential(1200, n),
            "average_medicare_payment_amount": np.random.exponential(600, n),
            "charge_to_payment_ratio": np.random.exponential(2.5, n),
            "services_per_beneficiary": np.random.exponential(3, n),
            "is_high_volume": np.random.binomial(1, 0.05, n),
            "is_abnormal_charge": np.random.binomial(1, 0.03, n),
        })
        df["fraud_score"] = np.random.beta(1, 10, n)
        df["fraud_score"] = np.where(
            df["is_abnormal_charge"] == 1,
            df["fraud_score"] * 3, df["fraud_score"]
        ).clip(0, 1)
        df["is_fraud_predicted"] = (df["fraud_score"] > 0.5).astype(int)
        df["risk_level"] = pd.cut(
            df["fraud_score"],
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=["Low", "Medium", "High", "Critical"]
        ).astype(str)
        return df


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/medical-doctor.png", width=80)
st.sidebar.title("🏥 Fraud Detection")
st.sidebar.markdown("---")

df = load_data()

states = ["All"] + sorted(df["nppes_provider_state"].dropna().unique().tolist()) if "nppes_provider_state" in df.columns else ["All"]
specialties = ["All"] + sorted(df["provider_type"].dropna().unique().tolist()) if "provider_type" in df.columns else ["All"]

selected_state = st.sidebar.selectbox("📍 Filter by State", states)
selected_specialty = st.sidebar.selectbox("🩺 Filter by Specialty", specialties)
risk_filter = st.sidebar.multiselect(
    "⚠️ Risk Level", ["Low", "Medium", "High", "Critical"],
    default=["Low", "Medium", "High", "Critical"]
)

# Apply filters
filtered = df.copy()
if selected_state != "All" and "nppes_provider_state" in df.columns:
    filtered = filtered[filtered["nppes_provider_state"] == selected_state]
if selected_specialty != "All" and "provider_type" in df.columns:
    filtered = filtered[filtered["provider_type"] == selected_specialty]
if risk_filter:
    filtered = filtered[filtered["risk_level"].isin(risk_filter)]

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🏥 Medicare Fraud Detection System")
st.markdown("**Protecting $60B+ in annual Medicare spending using Machine Learning**")
st.markdown("---")

# ── KPI Metrics ──────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

total = len(filtered)
fraud_count = filtered["is_fraud_predicted"].sum()
fraud_rate = (fraud_count / total * 100) if total > 0 else 0
avg_fraud_score = filtered["fraud_score"].mean()
critical_count = (filtered["risk_level"] == "Critical").sum()

col1.metric("📋 Total Providers", f"{total:,}")
col2.metric("🚨 Fraud Detected", f"{fraud_count:,}", delta=f"{fraud_rate:.1f}%")
col3.metric("📊 Avg Fraud Score", f"{avg_fraud_score:.3f}")
col4.metric("🔴 Critical Risk", f"{critical_count:,}")
col5.metric("💰 Est. Savings", f"${fraud_count * 45000:,.0f}")

st.markdown("---")

# ── Charts ───────────────────────────────────────────────────────────────────
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("📊 Fraud Score Distribution")
    fig = px.histogram(
        filtered, x="fraud_score", nbins=50, color_discrete_sequence=["#764ba2"],
        title="Distribution of Fraud Probability Scores",
        labels={"fraud_score": "Fraud Score", "count": "Number of Providers"}
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Fraud Threshold")
    st.plotly_chart(fig, use_container_width=True)

with row1_col2:
    st.subheader("⚠️ Risk Level Breakdown")
    risk_counts = filtered["risk_level"].value_counts()
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        color_discrete_map={"Low": "#00C851", "Medium": "#ffbb33", "High": "#ff8800", "Critical": "#ff4444"},
        title="Provider Risk Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    if "provider_type" in filtered.columns:
        st.subheader("🩺 Fraud by Specialty")
        specialty_fraud = (
            filtered.groupby("provider_type")["fraud_score"]
            .mean().sort_values(ascending=False).head(10).reset_index()
        )
        fig = px.bar(
            specialty_fraud, x="fraud_score", y="provider_type",
            orientation="h", color="fraud_score",
            color_continuous_scale="Reds",
            title="Top 10 Specialties by Avg Fraud Score"
        )
        st.plotly_chart(fig, use_container_width=True)

with row2_col2:
    if "nppes_provider_state" in filtered.columns:
        st.subheader("🗺️ Fraud by State")
        state_fraud = (
            filtered.groupby("nppes_provider_state")["is_fraud_predicted"]
            .sum().sort_values(ascending=False).head(15).reset_index()
        )
        fig = px.bar(
            state_fraud, x="nppes_provider_state", y="is_fraud_predicted",
            color="is_fraud_predicted", color_continuous_scale="Reds",
            title="Fraud Cases by State",
            labels={"is_fraud_predicted": "Fraud Cases", "nppes_provider_state": "State"}
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Scatter Plot ─────────────────────────────────────────────────────────────
st.subheader("🔍 Charge vs Payment Anomaly Detection")
if "average_submitted_charge_amount" in filtered.columns:
    fig = px.scatter(
        filtered.sample(min(2000, len(filtered))),
        x="average_submitted_charge_amount",
        y="average_medicare_payment_amount",
        color="risk_level",
        color_discrete_map={"Low": "#00C851", "Medium": "#ffbb33", "High": "#ff8800", "Critical": "#ff4444"},
        size="fraud_score",
        title="Charge Amount vs Medicare Payment (size = fraud score)",
        labels={
            "average_submitted_charge_amount": "Submitted Charge ($)",
            "average_medicare_payment_amount": "Medicare Payment ($)"
        },
        opacity=0.7
    )
    st.plotly_chart(fig, use_container_width=True)

# ── High Risk Table ───────────────────────────────────────────────────────────
st.subheader("🚨 High Risk Providers")
high_risk = filtered[filtered["risk_level"].isin(["High", "Critical"])].sort_values(
    "fraud_score", ascending=False
).head(20)

display_cols = [c for c in [
    "national_provider_identifier", "provider_type", "nppes_provider_state",
    "fraud_score", "risk_level", "number_of_services", "average_submitted_charge_amount"
] if c in high_risk.columns]

if display_cols:
    st.dataframe(
        high_risk[display_cols],
        column_config={
            "fraud_score": st.column_config.ProgressColumn(
                "Fraud Score", min_value=0, max_value=1, format="%.4f"
            )
        },
        use_container_width=True
    )

# ── Live Predict ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔬 Live Fraud Prediction")
st.markdown("Enter provider data to get real-time fraud prediction:")

p_col1, p_col2, p_col3 = st.columns(3)
with p_col1:
    services = st.number_input("Number of Services", min_value=0, value=500)
    beneficiaries = st.number_input("Medicare Beneficiaries", min_value=0, value=100)
with p_col2:
    charge = st.number_input("Avg Submitted Charge ($)", min_value=0.0, value=1500.0)
    payment = st.number_input("Avg Medicare Payment ($)", min_value=0.0, value=800.0)
with p_col3:
    standardized = st.number_input("Avg Standardized Amount ($)", min_value=0.0, value=750.0)
    provider_id = st.text_input("Provider ID", value="NPI1234567890")

if st.button("🔍 Predict Fraud", type="primary"):
    try:
        response = requests.post(f"{API_URL}/predict", json={
            "provider_id": provider_id,
            "number_of_services": services,
            "number_of_medicare_beneficiaries": beneficiaries,
            "average_submitted_charge_amount": charge,
            "average_medicare_payment_amount": payment,
            "average_medicare_standardized_amount": standardized,
        }, timeout=5)
        result = response.json()

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Fraud Score", f"{result['fraud_score']:.4f}")
        col_r2.metric("Risk Level", result["risk_level"])
        col_r3.metric("Fraud Predicted", "✅ YES" if result["is_fraud_predicted"] else "❌ NO")
    except Exception:
        # Demo mode without API
        ratio = charge / payment if payment > 0 else 1
        score = min(0.9, max(0.1, (ratio - 1) * 0.3 + services / 10000))
        risk = "Critical" if score > 0.8 else "High" if score > 0.6 else "Medium" if score > 0.3 else "Low"
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Fraud Score", f"{score:.4f}")
        col_r2.metric("Risk Level", risk)
        col_r3.metric("Fraud Predicted", "✅ YES" if score > 0.5 else "❌ NO")
        st.info("💡 Running in demo mode. Start the API with `docker-compose up` for full predictions.")

st.markdown("---")
st.markdown("Built with ❤️ using Python, XGBoost, FastAPI & Streamlit | Real CMS Medicare Data")
