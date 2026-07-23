"""Verify artifact integrity, model structure, and manuscript-result invariants."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppg_rmssd.core import load_models  # noqa: E402


def verify_hashes() -> None:
    for line in (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        data = (ROOT / relative).read_bytes()
        if Path(relative).suffix not in {".png", ".xz"}:
            data = data.replace(b"\r\n", b"\n")
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise AssertionError(f"Hash mismatch: {relative}")


def verify_models() -> None:
    models = load_models(ROOT / "models" / "frozen_candidate_v1_models.joblib.xz")
    assert set(models) == {30.0, 60.0}
    for classifier, regressor in models.values():
        assert classifier.n_features_in_ == 14
        assert regressor.n_features_in_ == 14


def verify_reported_results() -> None:
    ptt = pd.read_csv(ROOT / "results/primary_external/ptt_ppg_summary.csv")
    primary = ptt[ptt.gate.eq("frozen_gate_065")].set_index("channel")
    expected = {"pleth_1": (15, 4.455054483, 0.829291425),
                "pleth_2": (20, 5.054588981, 0.813383206),
                "pleth_3": (20, 4.095911460, 0.854971814)}
    for channel, (accepted, mae, ccc) in expected.items():
        assert int(primary.loc[channel, "accepted_n"]) == accepted
        assert np.isclose(primary.loc[channel, "rmssd_mae"], mae, atol=1e-9)
        assert np.isclose(primary.loc[channel, "rmssd_ccc"], ccc, atol=1e-9)

    vollmer = pd.read_csv(ROOT / "results/secondary_external/vollmer_summary.csv").iloc[0]
    assert int(vollmer.accepted_n) == 12
    assert np.isclose(vollmer.rmssd_mae, 3.111464360, atol=1e-9)
    assert np.isclose(vollmer.rmssd_ccc, 0.943008856, atol=1e-9)

    utsa = pd.read_csv(ROOT / "results/out_of_domain/utsa_summary.csv")
    office = utsa[(utsa.evaluation == "office_stress") & (utsa.gate == "gate_sqi065_consistency")].set_index("channel")
    assert int(office.loc["ir", "accepted_n"]) == 14
    assert int(office.loc["red", "accepted_n"]) == 9
    assert np.isclose(office.loc["ir", "rmssd_mae"], 28.682126316, atol=1e-9)
    assert np.isclose(office.loc["red", "rmssd_mae"], 42.452632864, atol=1e-9)

    audit = pd.read_csv(ROOT / "results/posthoc/nn_label_audit.csv")
    assert audit.changed_labels.tolist() == [17, 17]
    assert audit.eligible_labels.tolist() == [11883, 11910]

    intervals = pd.read_csv(ROOT / "results/primary_external/ptt_bootstrap_ci.csv").set_index("channel")
    expected_intervals = {
        "pleth_1": (15, 1.99, 8.17, 0.694, 0.963),
        "pleth_2": (20, 3.24, 7.26, 0.445, 0.936),
        "pleth_3": (20, 2.25, 6.77, 0.646, 0.971),
    }
    for channel, expected_values in expected_intervals.items():
        observed = intervals.loc[channel]
        assert int(observed.accepted_n) == expected_values[0]
        assert int(observed.resamples) == 10000
        assert np.allclose(
            observed[["mae_ci_low_ms", "mae_ci_high_ms", "ccc_ci_low", "ccc_ci_high"]].to_numpy(float),
            expected_values[1:],
            atol=1e-12,
        )


if __name__ == "__main__":
    verify_hashes()
    verify_models()
    verify_reported_results()
    print("Repository verification passed.")
