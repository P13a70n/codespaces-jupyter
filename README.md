# Bank Fraud Detection Dashboard

This project demonstrates a banking fraud detection use case using a Streamlit frontend and Python-based ML workflow.

## What it includes

- Streamlit dashboard for transaction monitoring
- Sequence-based fraud scoring inspired by LSTM behavior
- Synthetic demo dataset for fast local testing
- Optional support for the Kaggle Credit Card Fraud Detection dataset
- Precision, recall, F1-score, and ROC-AUC reporting

## Project structure

- app.py: Streamlit frontend
- src/fraud_model.py: training and fraud scoring logic
- data/fraud_demo.csv: generated demo dataset
- tests/test_fraud_pipeline.py: smoke tests

## Run locally

1. Install dependencies:
   pip install -r requirements.txt
2. Launch the dashboard:
   streamlit run app.py
3. Open the local URL shown in the terminal.

## Kaggle dataset integration

To use the real Kaggle dataset, place the file in the data folder as:

- data/creditcard.csv

The project checks for this dataset automatically before generating the demo version.

## Model notes

- The demo uses a sequence-aware scoring approach inspired by LSTM architecture.
- If TensorFlow is available in the environment, the project can train an LSTM model path.
- If TensorFlow is not installed, it falls back to a logistic-regression model for reliable local execution.
