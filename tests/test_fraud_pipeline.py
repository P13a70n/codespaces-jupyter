import numpy as np

from src.fraud_model import FraudDetector


def test_prepare_sequences_creates_expected_shape():
    detector = FraudDetector()
    history = [100, 150, 200, 300, 500]
    sequence = detector.prepare_sequence(history, window_size=5)
    assert len(sequence) == 5
    assert np.allclose(sequence, np.array([100, 150, 200, 300, 500], dtype=float))


def test_probability_is_clamped_between_zero_and_one():
    detector = FraudDetector()
    prob = detector.score_transaction([100, 200, 300, 5000, 7000])
    assert 0.0 <= prob <= 1.0


def test_threshold_uses_risk_level():
    detector = FraudDetector()
    assert detector.classify_risk(0.92) == "HIGH RISK"
    assert detector.classify_risk(0.55) == "MEDIUM RISK"
    assert detector.classify_risk(0.15) == "LOW RISK"
