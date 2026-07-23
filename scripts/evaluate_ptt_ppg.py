"""Reproduce the primary external validation on PTT-PPG seated records."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from ppg_rmssd.core import extract_intervals, infer, load_models, nn_mask, rmssd_consecutive, sqi_features
from _evaluation import combine_rates, summarize

CHANNELS = {"pleth_1": "distal_pleth_1", "pleth_2": "distal_pleth_2", "pleth_3": "distal_pleth_3"}
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "frozen_candidate_v1_models.joblib.xz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Directory containing s*_sit WFDB records")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ptt_ppg"))
    args = parser.parse_args()
    models = load_models(args.model)
    rows = []
    records = sorted(args.dataset.glob("s*_sit.hea"), key=lambda p: int(p.stem.split("_")[0][1:]))
    if not records:
        raise FileNotFoundError(f"No s*_sit.hea records found in {args.dataset}")
    for header_path in records:
        record = header_path.stem
        header = wfdb.rdheader(str(args.dataset / record))
        source_fs = float(header.fs)
        start = max(0, int((header.sig_len - 60 * source_fs) / 2))
        stop = start + int(60 * source_fs)
        annotation = wfdb.rdann(str(args.dataset / record), "atr")
        r_times = (annotation.sample[(annotation.sample >= start) & (annotation.sample < stop)] - start) / source_fs
        rr = np.diff(r_times)
        reference_nn = nn_mask(rr)
        reference_rmssd, _ = rmssd_consecutive(rr, reference_nn)
        reference_hr = 60 / np.median(rr[reference_nn])
        signal = wfdb.rdrecord(str(args.dataset / record), sampfrom=start, sampto=stop,
                               channel_names=list(CHANNELS)).p_signal
        for index, (channel, label) in enumerate(CHANNELS.items()):
            source = signal[:, index]
            for fs in (30.0, 60.0):
                ppg = np.interp(np.arange(0, 60, 1 / fs), np.arange(len(source)) / source_fs, source)
                row = {"subject": record.split("_")[0], "record": record, "channel": channel,
                       "channel_label": label, "analysis_fs_hz": fs, "source_fs_hz": source_fs,
                       "reference_rmssd_ms": reference_rmssd, "reference_hr_bpm": reference_hr}
                row.update(infer(extract_intervals(ppg, fs), *models[fs]))
                row.update(sqi_features(ppg, fs))
                rows.append(row)
    detail = combine_rates(pd.DataFrame(rows), ["subject", "record", "channel", "channel_label"])
    summary = summarize(detail, "channel")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "ptt_ppg_detail.csv", index=False, float_format="%.9f")
    summary.to_csv(args.output_dir / "ptt_ppg_summary.csv", index=False, float_format="%.9f")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

