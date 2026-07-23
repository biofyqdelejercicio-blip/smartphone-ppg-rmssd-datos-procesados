"""Reproduce the explicitly out-of-domain UTSA office stress test."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ppg_rmssd.core import combined_gate, extract_intervals, infer, load_models, metrics, nn_mask, rmssd_consecutive, sqi_features

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "frozen_candidate_v1_models.joblib.xz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="UTSA dataset directory containing subject*_office files")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/utsa_office"))
    args = parser.parse_args()
    models = load_models(args.model)
    rows = []
    files = sorted(args.dataset.glob("subject*_office_signals.csv"))
    if not files:
        raise FileNotFoundError(f"No subject*_office_signals.csv files found in {args.dataset}")
    for signal_path in files:
        stem = signal_path.stem.removesuffix("_signals")
        subject, scenario = stem.split("_", 1)
        signal = pd.read_csv(signal_path, usecols=["time", "ir", "red"])
        ecg = pd.read_csv(args.dataset / f"{stem}_ecg.csv", usecols=["time", "rr"])
        time = signal.time.to_numpy(float)
        beat_time = ecg.time.to_numpy(float)
        beat_rr = ecg.rr.to_numpy(float) / 1000
        end = min(time[-1], beat_time[-1])
        for window, start in enumerate(np.linspace(0, end - 60, 5)):
            beat = (beat_time >= start) & (beat_time < start + 60)
            rr = beat_rr[beat]
            reference_nn = nn_mask(rr)
            reference_rmssd, _ = rmssd_consecutive(rr, reference_nn)
            reference_hr = 60 / np.median(rr[reference_nn])
            for channel in ("ir", "red"):
                rate_rows = {}
                source = signal[channel].to_numpy(float)
                for fs in (30.0, 60.0):
                    ppg = np.interp(start + np.arange(0, 60, 1 / fs), time, source)
                    result = infer(extract_intervals(ppg, fs), *models[fs])
                    result.update(sqi_features(ppg, fs))
                    rate_rows[fs] = result
                row = {"analysis_fs_hz": 60.0, "fold": -1, "subject": subject, "scenario": scenario,
                       "channel": channel, "window": window, "evaluation": "office_stress",
                       "reference_rmssd_ms": reference_rmssd, "predicted_rmssd_ms": rate_rows[60.0]["predicted_rmssd_ms"],
                       "reference_hr_bpm": reference_hr, "predicted_hr_bpm": rate_rows[60.0]["predicted_hr_bpm"],
                       **{k: v for k, v in rate_rows[60.0].items() if k not in ("predicted_rmssd_ms", "predicted_hr_bpm")}}
                row.update({"rmssd30": rate_rows[30.0]["predicted_rmssd_ms"], "rmssd60": rate_rows[60.0]["predicted_rmssd_ms"],
                            "hr30": rate_rows[30.0]["predicted_hr_bpm"], "hr60": rate_rows[60.0]["predicted_hr_bpm"],
                            "base30": rate_rows[30.0]["quality_gate"], "base60": rate_rows[60.0]["quality_gate"],
                            "sqi30": rate_rows[30.0]["sqi"], "sqi60": rate_rows[60.0]["sqi"]})
                row["cross_fs_consistent"] = combined_gate(rate_rows[30.0], rate_rows[60.0], 0.0)
                row["gate_sqi065_consistency"] = combined_gate(rate_rows[30.0], rate_rows[60.0], 0.65)
                row["gate_sqi075_consistency"] = combined_gate(rate_rows[30.0], rate_rows[60.0], 0.75)
                rows.append(row)
    detail = pd.DataFrame(rows)
    summary_rows = []
    for gate in ("gate_sqi065_consistency", "gate_sqi075_consistency"):
        for channel, frame in detail.groupby("channel"):
            accepted = frame[frame[gate]]
            row = {"evaluation": "office_stress", "scenario": "office", "channel": channel, "gate": gate,
                   "windows_n": len(frame), "accepted_n": len(accepted), "coverage": len(accepted) / len(frame)}
            row.update({f"rmssd_{k}": v for k, v in metrics(accepted.reference_rmssd_ms, accepted.predicted_rmssd_ms).items()})
            row.update({f"hr_{k}": v for k, v in metrics(accepted.reference_hr_bpm, accepted.predicted_hr_bpm).items()})
            summary_rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "utsa_office_detail.csv", index=False, float_format="%.9f")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "utsa_office_summary.csv", index=False, float_format="%.9f")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

