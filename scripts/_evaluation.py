"""Shared table assembly for external-evaluation scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ppg_rmssd.core import combined_gate, metrics


def combine_rates(detail: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    values = ["predicted_rmssd_ms", "predicted_hr_bpm", "quality_gate", "sqi"]
    wide = detail.pivot(index=keys, columns="analysis_fs_hz", values=values).reset_index()
    wide.columns = ["_".join(str(x) for x in col if str(x)) if isinstance(col, tuple) else col for col in wide.columns]
    output = detail[detail.analysis_fs_hz.eq(60.0)].merge(wide, on=keys, how="left")
    for threshold in (0.65, 0.75):
        gate = f"frozen_gate_{int(threshold * 100):03d}"
        output[gate] = output.apply(
            lambda row: combined_gate(
                {"predicted_rmssd_ms": row["predicted_rmssd_ms_30.0"],
                 "predicted_hr_bpm": row["predicted_hr_bpm_30.0"],
                 "quality_gate": row["quality_gate_30.0"], "sqi": row["sqi_30.0"]},
                {"predicted_rmssd_ms": row["predicted_rmssd_ms_60.0"],
                 "predicted_hr_bpm": row["predicted_hr_bpm_60.0"],
                 "quality_gate": row["quality_gate_60.0"], "sqi": row["sqi_60.0"]},
                threshold,
            ), axis=1,
        )
    return output


def summarize(detail: pd.DataFrame, group_column: str | None = None) -> pd.DataFrame:
    rows = []
    groups = [(None, detail)] if group_column is None else detail.groupby(group_column)
    for gate in ("frozen_gate_065", "frozen_gate_075"):
        for group, frame in groups:
            accepted = frame[frame[gate]]
            row = {"gate": gate, "subjects_n": len(frame), "accepted_n": len(accepted),
                   "coverage": len(accepted) / len(frame) if len(frame) else np.nan}
            if group_column is not None:
                row[group_column] = group
            row.update({f"rmssd_{k}": v for k, v in metrics(
                accepted.reference_rmssd_ms.to_numpy(), accepted.predicted_rmssd_ms.to_numpy()).items()})
            row.update({f"hr_{k}": v for k, v in metrics(
                accepted.reference_hr_bpm.to_numpy(), accepted.predicted_hr_bpm.to_numpy()).items()})
            rows.append(row)
    return pd.DataFrame(rows)

