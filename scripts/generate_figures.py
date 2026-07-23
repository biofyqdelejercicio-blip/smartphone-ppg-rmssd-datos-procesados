"""Generate the manuscript figures and graphical abstract from archived results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

COLORS = {
    "pleth_1": "#B9342B",
    "pleth_2": "#3E5366",
    "pleth_3": "#21875A",
    "vollmer": "#315F8A",
}


def _style() -> None:
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
        font_family = "Arial"
    except ValueError:
        font_family = "DejaVu Sans"
    mpl.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.fontsize": 7.0,
            "savefig.facecolor": "white",
        }
    )


def _accepted(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["frozen_gate_065"].astype(bool)].copy()


def _identity_limits(reference: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([reference, predicted])
    margin = max(2.0, 0.08 * (values.max() - values.min()))
    return float(values.min() - margin), float(values.max() + margin)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _save(fig: plt.Figure, stem: str, submission_dir: Path | None) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=350, bbox_inches="tight", pad_inches=0.04)
    if submission_dir is not None:
        submission_dir.mkdir(parents=True, exist_ok=True)
        number = {"figure_1_ptt_scatter": 1, "figure_2_ptt_bland_altman": 2, "figure_3_vollmer": 3}[stem]
        target = submission_dir / f"Fig{number}.tiff"
        fig.savefig(
            target,
            dpi=600,
            format="tiff",
            pil_kwargs={"compression": "tiff_lzw"},
        )
        with Image.open(target) as image:
            rgb = image.convert("RGB")
            rgb.save(target, dpi=(600, 600), compression="tiff_lzw")
    plt.close(fig)


def figure_1(df: pd.DataFrame, submission_dir: Path | None) -> None:
    labels = [
        ("pleth_1", "Red (pleth_1)"),
        ("pleth_2", "Infrared (pleth_2)"),
        ("pleth_3", "Green (pleth_3)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(174 / 25.4, 59 / 25.4), constrained_layout=True)
    for index, (channel, title) in enumerate(labels):
        ax = axes[index]
        part = _accepted(df.loc[df["channel"] == channel])
        ref = part["reference_rmssd_ms"].to_numpy(float)
        pred = part["predicted_rmssd_ms"].to_numpy(float)
        lo, hi = _identity_limits(ref, pred)
        ax.scatter(ref, pred, s=22, color=COLORS[channel], edgecolor="white", linewidth=0.45, zorder=3)
        ax.plot([lo, hi], [lo, hi], color="#202124", linewidth=0.9, zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, pad=5)
        ax.set_xlabel("ECG RMSSD (ms)")
        if index == 0:
            ax.set_ylabel("PPG RMSSD (ms)")
        ax.grid(True, color="#D9DEE3", linewidth=0.45, alpha=0.8)
        _panel_label(ax, chr(ord("a") + index))
    _save(fig, "figure_1_ptt_scatter", submission_dir)


def figure_2(df: pd.DataFrame, submission_dir: Path | None) -> None:
    labels = [
        ("pleth_1", "Red (pleth_1)"),
        ("pleth_2", "Infrared (pleth_2)"),
        ("pleth_3", "Green (pleth_3)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(174 / 25.4, 59 / 25.4), constrained_layout=True)
    for index, (channel, title) in enumerate(labels):
        ax = axes[index]
        part = _accepted(df.loc[df["channel"] == channel])
        ref = part["reference_rmssd_ms"].to_numpy(float)
        pred = part["predicted_rmssd_ms"].to_numpy(float)
        mean = (ref + pred) / 2.0
        difference = pred - ref
        bias = difference.mean()
        sd = difference.std(ddof=1)
        loa_low, loa_high = bias - 1.96 * sd, bias + 1.96 * sd
        ax.scatter(mean, difference, s=22, color=COLORS[channel], edgecolor="white", linewidth=0.45, zorder=3)
        ax.axhline(bias, color="#202124", linewidth=0.9)
        ax.axhline(loa_low, color="#202124", linewidth=0.8, linestyle="--")
        ax.axhline(loa_high, color="#202124", linewidth=0.8, linestyle="--")
        ax.set_title(title, pad=5)
        ax.set_xlabel("Mean RMSSD (ms)")
        if index == 0:
            ax.set_ylabel("PPG − ECG RMSSD (ms)")
        ax.grid(True, color="#D9DEE3", linewidth=0.45, alpha=0.8)
        _panel_label(ax, chr(ord("a") + index))
    _save(fig, "figure_2_ptt_bland_altman", submission_dir)


def figure_3(df: pd.DataFrame, submission_dir: Path | None) -> None:
    part = _accepted(df)
    ref = part["reference_rmssd_ms"].to_numpy(float)
    pred = part["predicted_rmssd_ms"].to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(174 / 25.4, 70 / 25.4), constrained_layout=True)

    lo, hi = _identity_limits(ref, pred)
    axes[0].scatter(ref, pred, s=28, color=COLORS["vollmer"], edgecolor="white", linewidth=0.5, zorder=3)
    axes[0].plot([lo, hi], [lo, hi], color="#202124", linewidth=0.9)
    axes[0].set_xlim(lo, hi)
    axes[0].set_ylim(lo, hi)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title("Agreement", pad=5)
    axes[0].set_xlabel("ECG RMSSD (ms)")
    axes[0].set_ylabel("PPG RMSSD (ms)")
    axes[0].grid(True, color="#D9DEE3", linewidth=0.45, alpha=0.8)
    _panel_label(axes[0], "a")

    mean = (ref + pred) / 2.0
    difference = pred - ref
    bias = difference.mean()
    sd = difference.std(ddof=1)
    axes[1].scatter(mean, difference, s=28, color=COLORS["vollmer"], edgecolor="white", linewidth=0.5, zorder=3)
    axes[1].axhline(bias, color="#202124", linewidth=0.9)
    axes[1].axhline(bias - 1.96 * sd, color="#202124", linewidth=0.8, linestyle="--")
    axes[1].axhline(bias + 1.96 * sd, color="#202124", linewidth=0.8, linestyle="--")
    axes[1].set_title("Bland–Altman", pad=5)
    axes[1].set_xlabel("Mean RMSSD (ms)")
    axes[1].set_ylabel("PPG − ECG RMSSD (ms)")
    axes[1].grid(True, color="#D9DEE3", linewidth=0.45, alpha=0.8)
    _panel_label(axes[1], "b")
    _save(fig, "figure_3_vollmer", submission_dir)


def graphical_abstract(submission_dir: Path) -> None:
    source = FIGURES / "graphical_abstract_source.png"
    if not source.exists():
        raise FileNotFoundError(f"Missing graphical abstract source: {source}")
    submission_dir.mkdir(parents=True, exist_ok=True)
    target = submission_dir / "Graphical_Abstract_Hernandez-Garcia.tiff"
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        fitted = ImageOps.fit(
            rgb,
            (778, 889),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        fitted.save(target, dpi=(600, 600), compression="tiff_lzw")
        fitted.save(
            FIGURES / "graphical_abstract_preview.png",
            dpi=(600, 600),
            optimize=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", type=Path)
    args = parser.parse_args()
    _style()
    ptt = pd.read_csv(RESULTS / "primary_external" / "ptt_ppg_detail.csv")
    vollmer = pd.read_csv(RESULTS / "secondary_external" / "vollmer_detail.csv")
    figure_1(ptt, args.submission_dir)
    figure_2(ptt, args.submission_dir)
    figure_3(vollmer, args.submission_dir)
    if args.submission_dir is not None:
        graphical_abstract(args.submission_dir)


if __name__ == "__main__":
    main()
