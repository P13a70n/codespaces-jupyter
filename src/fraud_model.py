from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    HAS_SKLEARN = True
except Exception:  # pragma: no cover
    HAS_SKLEARN = False

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM

    HAS_TENSORFLOW = True
except Exception:  # pragma: no cover
    HAS_TENSORFLOW = False


class FraudDetector:
    def __init__(self, data_path: str | Path = "data/fraud_demo.csv", sequence_length: int = 5):
        self.data_path = Path(data_path)
        if self.data_path.name == "fraud_demo.csv":
            kaggle_path = Path("data/creditcard.csv")
            if kaggle_path.exists():
                self.data_path = kaggle_path

        self.sequence_length = sequence_length
        self.model = None
        self.model_name = "Logistic Regression"
        self.model_metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.0,
        }
        self.ensure_demo_dataset()
        self.load_or_train()

    def ensure_demo_dataset(self) -> pd.DataFrame:
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        if self.data_path.exists():
            return pd.read_csv(self.data_path)

        demo_path = Path("data/fraud_demo.csv")
        demo_path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        rows: List[dict] = []

        for customer_id in range(1, 250):
            txn_count = int(rng.integers(12, 28))
            history: List[float] = []
            for tx_index in range(txn_count):
                base = float(rng.integers(150, 3500))
                if tx_index > 3 and rng.random() < 0.22:
                    base *= 1.8 + rng.random() * 6
                if tx_index > 5 and rng.random() < 0.08:
                    base *= 2.5 + rng.random() * 9

                history.append(base)
                if len(history) > self.sequence_length:
                    history = history[-self.sequence_length:]

                avg_history = float(np.mean(history)) if history else 0.0
                std_history = float(np.std(history)) if history else 0.0
                amount = float(base)

                anomaly_score = (
                    (amount / max(avg_history + 1, 1))
                    + (amount / max(std_history + 1, 1))
                    + float(max(0, len(history) - 4))
                )
                is_fraud = int(
                    (tx_index > 4 and amount > 7000 and anomaly_score > 5.4)
                    or (tx_index > 6 and amount > 15000 and rng.random() < 0.4)
                    or (rng.random() < 0.03 and amount > 2000)
                )

                rows.append(
                    {
                        "customer_id": customer_id,
                        "transaction_id": f"T{customer_id}-{tx_index + 1}",
                        "amount": round(amount, 2),
                        "time_hour": int(rng.integers(0, 24)),
                        "history_mean": round(avg_history, 2),
                        "history_std": round(std_history, 2),
                        "transaction_count": tx_index + 1,
                        "history_window": ",".join(f"{value:.2f}" for value in history),
                        "is_fraud": is_fraud,
                    }
                )

        df = pd.DataFrame(rows)
        df.to_csv(demo_path, index=False)
        self.data_path = demo_path
        return df

    @staticmethod
    def parse_history(raw_text: str) -> List[float]:
        values: List[float] = []
        for token in str(raw_text).replace(";", ",").split(","):
            clean = token.strip()
            if clean:
                try:
                    values.append(float(clean))
                except ValueError:
                    continue
        return values

    @staticmethod
    def prepare_sequence(history: Iterable[float], window_size: int = 5) -> np.ndarray:
        values = np.asarray(list(history), dtype=float)
        if values.size == 0:
            return np.zeros(window_size, dtype=float)
        if values.size < window_size:
            pad = np.pad(values, (0, window_size - values.size), mode="edge")
            values = pad
        elif values.size > window_size:
            values = values[-window_size:]
        return values.astype(float)

    def build_feature_vector(self, history: List[float], current_amount: float | None = None) -> np.ndarray:
        seq = self.prepare_sequence(history, self.sequence_length)
        amount = float(current_amount if current_amount is not None else history[-1]) if history else 0.0
        mean = float(np.mean(seq))
        std = float(np.std(seq))
        return np.array([
            amount,
            mean,
            std,
            len(history),
        ], dtype=float)

    def score_transaction(self, history: List[float]) -> float:
        if not history:
            return 0.0
        if self.model is None:
            self.load_or_train()

        if self.model_name == "LSTM":
            sequence = self.prepare_sequence(history, self.sequence_length).reshape(1, -1, 1)
            prob = float(self.model.predict(sequence, verbose=0)[0][0])
        else:
            features = self.build_feature_vector(history, history[-1])
            prob = float(self.model.predict_proba(features.reshape(1, -1))[0][1])

        return float(np.clip(prob, 0.0, 1.0))

    def classify_risk(self, probability: float) -> str:
        if probability >= 0.75:
            return "HIGH RISK"
        if probability >= 0.45:
            return "MEDIUM RISK"
        return "LOW RISK"

    def load_or_train(self):
        if self.model is not None:
            return self.model

        df = pd.read_csv(self.data_path)

        if "Class" in df.columns:
            X = df[[col for col in df.columns if col not in {"Time", "Class"}]].astype(float).to_numpy()
            y = df["Class"].astype(int).to_numpy()
        else:
            feature_columns = ["amount", "history_mean", "history_std", "transaction_count"]
            X = df[feature_columns].astype(float).to_numpy()
            y = df["is_fraud"].astype(int).to_numpy()

        indices = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        if HAS_TENSORFLOW:
            self.model_name = "LSTM"
            seq_train = []
            seq_test = []
            for row_idx in train_idx:
                row = df.iloc[row_idx]
                if "history_window" in df.columns:
                    history = self.parse_history(str(row["history_window"]))
                else:
                    history = [float(row["Amount"]) for _ in range(self.sequence_length)]
                seq_train.append(self.prepare_sequence(history, self.sequence_length))

            for row_idx in test_idx:
                row = df.iloc[row_idx]
                if "history_window" in df.columns:
                    history = self.parse_history(str(row["history_window"]))
                else:
                    history = [float(row["Amount"]) for _ in range(self.sequence_length)]
                seq_test.append(self.prepare_sequence(history, self.sequence_length))

            seq_train = np.asarray(seq_train, dtype=float).reshape(-1, self.sequence_length, 1)
            seq_test = np.asarray(seq_test, dtype=float).reshape(-1, self.sequence_length, 1)

            model = Sequential(
                [
                    tf.keras.layers.Input(shape=(self.sequence_length, 1)),
                    LSTM(64, return_sequences=True),
                    Dropout(0.3),
                    LSTM(32),
                    Dense(16, activation="relu"),
                    Dense(1, activation="sigmoid"),
                ]
            )
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            model.fit(seq_train, y_train, epochs=8, batch_size=32, validation_split=0.2, verbose=0)
            self.model = model
            prob_test = model.predict(seq_test, verbose=0).ravel()
            y_pred = (prob_test >= 0.5).astype(int)
            self.model_metrics = {
                "precision": float(np.round(precision_score(y_test, y_pred, zero_division=0), 4)),
                "recall": float(np.round(recall_score(y_test, y_pred, zero_division=0), 4)),
                "f1": float(np.round(f1_score(y_test, y_pred), 4)),
                "roc_auc": float(np.round(roc_auc_score(y_test, prob_test), 4)),
            }
            return self.model

        if HAS_SKLEARN:
            self.model_name = "Logistic Regression"
            clf = LogisticRegression(class_weight="balanced", max_iter=2000)
            clf.fit(X_train, y_train)
            self.model = clf
            test_probs = self.model.predict_proba(X_test)[:, 1]
            y_pred = (test_probs >= 0.5).astype(int)
            self.model_metrics = {
                "precision": float(np.round(precision_score(y_test, y_pred, zero_division=0), 4)),
                "recall": float(np.round(recall_score(y_test, y_pred, zero_division=0), 4)),
                "f1": float(np.round(f1_score(y_test, y_pred), 4)),
                "roc_auc": float(np.round(roc_auc_score(y_test, test_probs), 4)),
            }
            return self.model

        self.model_name = "Rule-Based"
        self.model = None
        return self.model

    def dashboard_metrics(self) -> dict:
        if self.model is None:
            return {
                "status": "No model loaded",
                "risk": "LOW RISK",
                "probability": 0.0,
            }
        return {
            "status": self.model_name,
            "risk": "READY",
            "probability": 0.0,
            "precision": self.model_metrics["precision"],
            "recall": self.model_metrics["recall"],
            "f1": self.model_metrics["f1"],
            "roc_auc": self.model_metrics["roc_auc"],
        }
