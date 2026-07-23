"""Signal processing and frozen-model inference used in the manuscript."""

from __future__ import annotations

import lzma
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch

ANALYSIS_RATES = (30.0, 60.0)
INTERVAL_PROBABILITY = 0.20
MIN_ACCEPTED_FRACTION = 0.70
MIN_SUCCESSIVE_PAIRS = 30


def bandpass(x: np.ndarray, fs: float, lo: float = 0.55, hi: float = 3.5) -> np.ndarray:
    x = np.asarray(x, float)
    finite = np.flatnonzero(np.isfinite(x))
    if finite.size < 2:
        raise ValueError("Signal must contain at least two finite samples")
    x = np.interp(np.arange(len(x)), finite, x[finite])
    b, a = butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def parabolic_offset(y: np.ndarray, idx: int) -> float:
    if idx <= 0 or idx >= len(y) - 1:
        return 0.0
    a, b, c = y[idx - 1], y[idx], y[idx + 1]
    denominator = a - 2 * b + c
    if abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (a - c) / denominator, -0.75, 0.75))


def lowres_candidate_peaks(y: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Detect peaks after selecting the polarity closest to the spectral rate."""
    f, psd = welch(y, fs, nperseg=min(len(y), max(128, int(20 * fs))))
    band = (f >= 0.55) & (f <= 3.5)
    if not band.any():
        raise ValueError("Signal is too short for spectral peak detection")
    spectral_hr = float(f[band][np.argmax(psd[band])] * 60.0)
    min_distance = max(2, int(0.55 * (60.0 / spectral_hr) * fs))
    options = []
    for sign in (1.0, -1.0):
        oriented = sign * y
        peaks, _ = find_peaks(
            oriented,
            distance=min_distance,
            prominence=max(np.std(oriented) * 0.12, 1e-9),
        )
        peak_hr = 60.0 / np.median(np.diff(peaks) / fs) if len(peaks) > 2 else np.nan
        score = abs(peak_hr - spectral_hr) if np.isfinite(peak_hr) else 1e9
        options.append((score, peaks, oriented))
    _, peaks, oriented = min(options, key=lambda item: item[0])
    return peaks, oriented


def interval_features(times: np.ndarray, amplitude: np.ndarray, fs: float) -> np.ndarray:
    rr = np.diff(times)
    amplitude = amplitude / max(float(np.median(np.abs(amplitude))), 1e-9)
    median_rr = float(np.median(rr)) if len(rr) else 1.0
    rows = []
    for i, value in enumerate(rr):
        neighbourhood = [rr[int(np.clip(i + k, 0, len(rr) - 1))] for k in (-2, -1, 0, 1, 2)]
        local = float(np.median(neighbourhood))
        rows.append([
            *neighbourhood,
            value / max(median_rr, 1e-6),
            value / max(local, 1e-6),
            abs(value - local),
            abs(neighbourhood[1] - neighbourhood[3]),
            amplitude[i],
            amplitude[i + 1],
            abs(amplitude[i + 1] - amplitude[i]),
            median_rr,
            fs,
        ])
    return np.asarray(rows, float)


def extract_intervals(x: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    y = bandpass(x, fs)
    peaks, oriented = lowres_candidate_peaks(y, fs)
    pulse_times = peaks / fs + np.asarray([parabolic_offset(oriented, int(p)) / fs for p in peaks])
    return {
        "X": interval_features(pulse_times, oriented[peaks], fs),
        "raw_rr": np.diff(pulse_times),
    }


def nn_mask(rr: np.ndarray) -> np.ndarray:
    """Fixed local ECG NN rule, preserving original indices and adjacency."""
    rr = np.asarray(rr, float)
    output = np.zeros(len(rr), bool)
    if len(rr) < 3:
        return output
    plausible = rr[(rr >= 0.30) & (rr <= 2.0)]
    if not len(plausible):
        return output
    global_median = float(np.median(plausible))
    for i, value in enumerate(rr):
        local = rr[max(0, i - 3):min(len(rr), i + 4)]
        local = local[(local >= 0.30) & (local <= 2.0)]
        median = float(np.median(local)) if len(local) else global_median
        tolerance = max(0.20 * median, 0.035)
        output[i] = 0.30 <= value <= 2.0 and abs(value - median) <= tolerance
    return output


def rmssd_consecutive(rr: np.ndarray, accepted: np.ndarray) -> tuple[float, int]:
    rr_ms = np.asarray(rr, float) * 1000.0
    accepted = np.asarray(accepted, bool)
    pair = accepted[:-1] & accepted[1:] & np.isfinite(rr_ms[:-1]) & np.isfinite(rr_ms[1:])
    differences = np.diff(rr_ms)[pair]
    value = float(np.sqrt(np.mean(differences**2))) if len(differences) >= 2 else np.nan
    return value, int(len(differences))


def sqi_features(x: np.ndarray, fs: float) -> dict[str, float]:
    y = bandpass(x, fs)
    peaks, oriented = lowres_candidate_peaks(y, fs)
    f, power = welch(y, fs, nperseg=min(len(y), int(30 * fs)))
    band = (f >= 0.55) & (f <= 3.5)
    frequencies, band_power = f[band], power[band]
    dominant = float(frequencies[np.argmax(band_power)])
    concentration = float(band_power[np.abs(frequencies - dominant) <= 0.15].sum() / max(band_power.sum(), 1e-12))
    peak_hr = 60.0 / np.median(np.diff(peaks) / fs) if len(peaks) >= 3 else np.nan
    agreement = float(np.exp(-abs(peak_hr - 60 * dominant) / 10)) if np.isfinite(peak_hr) else 0.0
    expected = 60 * dominant
    coverage = float(np.clip(min(len(peaks) / max(expected, 1), expected / max(len(peaks), 1)), 0, 1))
    amplitude = np.abs(oriented[peaks])
    median = float(np.median(amplitude)) if len(amplitude) else 0.0
    mad = float(1.4826 * np.median(np.abs(amplitude - median))) if len(amplitude) else np.inf
    # This component is based on peak-amplitude consistency. The legacy key
    # name is retained because it is part of the frozen result schema.
    prominence = float(np.clip(1 - mad / max(median, 1e-12), 0, 1))
    spectral_quality = float(np.clip((concentration - 0.15) / 0.45, 0, 1))
    sqi = 0.25 * (spectral_quality + agreement + coverage + prominence)
    return {
        "sqi": sqi,
        "sqi_spectral": spectral_quality,
        "sqi_hr_agreement": agreement,
        "sqi_coverage": coverage,
        "sqi_prominence": prominence,
        "sqi_concentration_raw": concentration,
    }


def load_models(path: str | Path) -> dict[float, tuple[object, object]]:
    """Load the losslessly compressed frozen 30/60 Hz models."""
    path = Path(path)
    if path.suffix == ".xz":
        with lzma.open(path, "rb") as stream:
            models = joblib.load(stream)
    else:
        models = joblib.load(path)
    if set(float(k) for k in models) != set(ANALYSIS_RATES):
        raise ValueError("Expected frozen models for 30 and 60 Hz")
    return {float(k): value for k, value in models.items()}


def infer(example: dict[str, np.ndarray], classifier: object, regressor: object) -> dict[str, float | bool | int]:
    if not len(example["X"]):
        return {"predicted_rmssd_ms": np.nan, "predicted_hr_bpm": np.nan,
                "accepted_fraction": 0.0, "successive_pairs_n": 0, "quality_gate": False}
    probability = classifier.predict_proba(example["X"])[:, 1]
    corrected = example["raw_rr"] + regressor.predict(example["X"])
    accepted = probability >= INTERVAL_PROBABILITY
    predicted_rmssd, pairs = rmssd_consecutive(corrected, accepted)
    fraction = float(np.mean(accepted))
    quality_gate = fraction >= MIN_ACCEPTED_FRACTION and pairs >= MIN_SUCCESSIVE_PAIRS
    return {
        "predicted_rmssd_ms": predicted_rmssd if quality_gate else np.nan,
        "predicted_hr_bpm": 60.0 / float(np.median(corrected[accepted])) if quality_gate else np.nan,
        "accepted_fraction": fraction,
        "successive_pairs_n": pairs,
        "quality_gate": quality_gate,
    }


def combined_gate(row_30: dict, row_60: dict, sqi_threshold: float = 0.65) -> bool:
    rmssd_delta = abs(row_60["predicted_rmssd_ms"] - row_30["predicted_rmssd_ms"])
    rmssd_scale = (abs(row_60["predicted_rmssd_ms"]) + abs(row_30["predicted_rmssd_ms"])) / 2
    consistent = (abs(row_60["predicted_hr_bpm"] - row_30["predicted_hr_bpm"]) <= 2.0
                  and (rmssd_delta <= 5.0 or rmssd_delta / rmssd_scale <= 0.10))
    return bool(row_30["quality_gate"] and row_60["quality_gate"]
                and row_30["sqi"] >= sqi_threshold and row_60["sqi"] >= sqi_threshold
                and consistent)


def metrics(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    reference, predicted = np.asarray(reference, float), np.asarray(predicted, float)
    valid = np.isfinite(reference) & np.isfinite(predicted)
    x, y = reference[valid], predicted[valid]
    if not len(x):
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "pearson": np.nan, "ccc": np.nan}
    difference = y - x
    vx, vy = np.var(x), np.var(y)
    ccc = 2 * np.cov(x, y, ddof=0)[0, 1] / (vx + vy + (np.mean(x) - np.mean(y)) ** 2) if len(x) > 1 else np.nan
    return {"n": int(len(x)), "mae": float(np.mean(abs(difference))),
            "rmse": float(np.sqrt(np.mean(difference**2))), "bias": float(np.mean(difference)),
            "pearson": float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else np.nan, "ccc": float(ccc)}
