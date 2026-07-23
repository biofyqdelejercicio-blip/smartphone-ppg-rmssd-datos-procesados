import numpy as np

from ppg_rmssd.core import combined_gate, interval_features, metrics, nn_mask, rmssd_consecutive


def test_rmssd_preserves_adjacency():
    value, pairs = rmssd_consecutive(np.array([0.8, 0.9, 0.7, 0.8]), np.array([True, True, False, True]))
    assert pairs == 1
    assert np.isnan(value)


def test_nn_mask_rejects_large_local_outlier():
    mask = nn_mask(np.array([0.80, 0.82, 0.81, 1.50, 0.79, 0.80, 0.81]))
    assert mask.tolist() == [True, True, True, False, True, True, True]


def test_combined_gate_accepts_absolute_rmssd_tolerance():
    a = {"predicted_rmssd_ms": 20.0, "predicted_hr_bpm": 60.0, "quality_gate": True, "sqi": 0.70}
    b = {"predicted_rmssd_ms": 24.0, "predicted_hr_bpm": 61.0, "quality_gate": True, "sqi": 0.72}
    assert combined_gate(a, b, 0.65)


def test_metrics_identity():
    result = metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert result["mae"] == 0.0
    assert np.isclose(result["ccc"], 1.0)


def test_interval_feature_count_matches_frozen_model():
    times = np.array([0.0, 0.8, 1.6, 2.4, 3.2])
    amplitude = np.ones(times.size)
    features = interval_features(times, amplitude, 60.0)
    assert features.shape == (4, 14)


def test_combined_gate_rejects_low_sqi():
    a = {"predicted_rmssd_ms": 20.0, "predicted_hr_bpm": 60.0, "quality_gate": True, "sqi": 0.64}
    b = {"predicted_rmssd_ms": 21.0, "predicted_hr_bpm": 60.5, "quality_gate": True, "sqi": 0.72}
    assert not combined_gate(a, b, 0.65)
