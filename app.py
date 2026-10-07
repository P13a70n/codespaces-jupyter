from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import streamlit as st

from src.fraud_model import FraudDetector


st.set_page_config(page_title="Bank Fraud Detection", page_icon="🏦", layout="wide")


@st.cache_resource
def get_detector() -> FraudDetector:
    return FraudDetector(data_path="data/fraud_demo.csv")


def parse_history(raw_text: str) -> List[float]:
    values: List[float] = []
    for token in raw_text.replace(";", ",").split(","):
        value = token.strip()
        if value:
            try:
                values.append(float(value))
            except ValueError:
                continue
    return values


st.title("🏦 Bank Fraud Detection System")
st.caption("LSTM-inspired sequence model for fraud detection in financial transactions.")

detector = get_detector()

with st.sidebar:
    st.header("Model Overview")
    st.write("Model type:", detector.model_name)
    st.write("Dataset:", detector.data_path)
    st.write("Sequence length:", detector.sequence_length)
    st.write("AUC:", round(detector.model_metrics.get("roc_auc", 0.0), 4))
    st.write("F1-score:", round(detector.model_metrics.get("f1", 0.0), 4))

    st.header("Banking Checklist")
    st.markdown(
        """
        - Imbalanced data handling
        - Scaling and history tracking
        - LSTM sequence analysis
        - Fraud threshold tuning
        - Precision / recall review
        """
    )

left_col, right_col = st.columns(2)

with left_col:
    customer_id = st.text_input("Customer ID", value="C10234")
    amount = st.number_input("Transaction Amount (₹)", min_value=0.0, value=75000.0, step=100.0)
    transaction_time = st.time_input("Transaction Time")
    history_text = st.text_area(
        "Previous Transactions (comma separated)",
        value="500, 850, 1200, 1500, 2500, 75000",
        height=140,
    )

    if st.button("Predict Fraud Risk"):
        history = parse_history(history_text)
        if not history:
            st.warning("Enter previous transaction values before running a prediction.")
            st.stop()
        if len(history) < 3:
            history = history + [history[-1]] * (3 - len(history))
        history.append(float(amount))
        probability = detector.score_transaction(history)
        risk_label = detector.classify_risk(probability)
        recommendation = (
            "Block transaction and request OTP verification"
            if risk_label == "HIGH RISK"
            else "Monitor transaction and verify customer details"
            if risk_label == "MEDIUM RISK"
            else "Allow transaction and continue standard monitoring"
        )

        st.metric("Fraud Probability", f"{probability * 100:.1f}%")

        if risk_label == "HIGH RISK":
            st.error(f"🚨 {risk_label} - {recommendation}")
        elif risk_label == "MEDIUM RISK":
            st.warning(f"⚠️ {risk_label} - {recommendation}")
        else:
            st.success(f"✅ {risk_label} - {recommendation}")

with right_col:
    st.subheader("Customer Activity")
    history = parse_history(history_text)
    if not history:
        history = [500, 850, 1200, 1500, 2500]
    plot_history = history + [float(amount)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(plot_history)), plot_history, marker="o", linewidth=2.5)
    ax.axvline(len(plot_history) - 1, color="red", linestyle="--", linewidth=1.5, label="Current transaction")
    ax.set_title("Recent Transaction Sequence")
    ax.set_xlabel("Transaction Index")
    ax.set_ylabel("Amount (₹)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.subheader("Model Summary")
    st.markdown(
        """
        The fraud model examines the customer's recent transaction pattern rather than a single amount.
        A sudden spike in spending, unusual sequence behavior, and elevated transaction volume increase
        the probability of fraud.
        """
    )

st.subheader("Why This Is a Good Banking ML Project")
st.markdown(
    """
    - LSTM learns temporal behavior across sequences instead of looking at only one transaction.
    - The dataset is highly imbalanced, which mirrors real banking data.
    - Model review includes precision, recall, F1-score, confusion matrix, ROC-AUC, and threshold tuning.
    - This is a realistic dashboard for a fraud operations team to monitor suspicious transactions.
    """
)
