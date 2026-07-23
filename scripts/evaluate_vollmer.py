"""Reproduce the secondary external replication on Vollmer resting records."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from ppg_rmssd.core import extract_intervals, infer, load_models, nn_mask, rmssd_consecutive, sqi_features
from _evaluation import combine_rates, summarize

SOURCE_FS = 256.0
PPG_CHANNEL = "SOT/Pleth"
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "frozen_candidate_v1_models.joblib.xz"


def fixed_rest_window(record: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    annotation = wfdb.rdann(str(record), "aux")
    notes = [str(note).strip() for note in annotation.aux_note]
    rest = next(i for i, note in enumerate(notes) if note.endswith("/Rest"))
    walking = next(i for i, note in enumerate(notes) if note.endswith("/Walking"))
    phase_start, phase_stop = int(annotation.sample[rest]), int(annotation.sample[walking])
    window_samples = int(60 * SOURCE_FS)
    if phase_stop - phase_start < window_samples:
        raise ValueError("Rest phase is shorter than 60 seconds")
    first = phase_start + (phase_stop - phase_start - window_samples) // 2
    signal, fields = wfdb.rdsamp(str(record), sampfrom=first, sampto=first + window_samples, channels=[11])
    if fields["sig_name"] != [PPG_CHANNEL]:
        raise ValueError(f"Expected channel {PPG_CHANNEL!r}, found {fields['sig_name']!r}")
    beats = wfdb.rdann(str(record), "atr").sample.astype(int)
    beats = beats[(beats >= first) & (beats < first + window_samples)] - first
    return signal[:, 0], beats, first, phase_stop - phase_start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Directory containing x??? WFDB records")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vollmer"))
    args = parser.parse_args()
    models = load_models(args.model)
    rows, audit = [], []
    records = sorted(args.dataset.glob("x???.hea"))
    if not records:
        raise FileNotFoundError(f"No x???.hea records found in {args.dataset}")
    for header_path in records:
        record, subject = header_path.with_suffix(""), header_path.stem
        try:
            source, beats, first, phase_samples = fixed_rest_window(record)
        except Exception as error:
            audit.append({"subject": subject, "status": "excluded", "reason": str(error)})
            continue
        rr = np.diff(beats) / SOURCE_FS
        reference_nn = nn_mask(rr)
        reference_rmssd, reference_pairs = rmssd_consecutive(rr, reference_nn)
        reference_hr = 60 / np.median(rr[reference_nn])
        audit.append({"subject": subject, "status": "included", "rest_phase_s": phase_samples / SOURCE_FS,
                      "window_start_sample": first, "r_peaks": len(beats),
                      "nn_intervals": int(reference_nn.sum()), "reference_pairs": reference_pairs})
        for fs in (30.0, 60.0):
            ppg = np.interp(np.arange(int(60 * fs)) / fs, np.arange(len(source)) / SOURCE_FS, source)
            row = {"subject": subject, "record": f"{subject}_standing_rest", "channel": "fingerclip_ppg",
                   "source_fs_hz": SOURCE_FS, "analysis_fs_hz": fs,
                   "reference_rmssd_ms": reference_rmssd, "reference_hr_bpm": reference_hr}
            row.update(infer(extract_intervals(ppg, fs), *models[fs]))
            row.update(sqi_features(ppg, fs))
            rows.append(row)
    detail = combine_rates(pd.DataFrame(rows), ["subject", "record", "channel"])
    summary = summarize(detail)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit).to_csv(args.output_dir / "vollmer_window_audit.csv", index=False, float_format="%.9f")
    detail.to_csv(args.output_dir / "vollmer_detail.csv", index=False, float_format="%.9f")
    summary.to_csv(args.output_dir / "vollmer_summary.csv", index=False, float_format="%.9f")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

