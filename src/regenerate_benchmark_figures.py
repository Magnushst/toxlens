#!/usr/bin/env python3
"""Regenerate four benchmark figures without rerunning any experiments.

Required inputs in --results-dir:
  - training_metrics.csv
  - sensitivity_sweep.csv

Optional exact-data overrides:
  - latent_drive.csv          columns: time_step, drive
  - raster_epoch_100.csv      columns: time_step, encoder_unit
  - breakeven_core_times.csv  columns: batch_size, core_ms

The original benchmark did not serialize the latent waveform, final raster
events, or the clean batch-256 timing. Exact reference values recovered from
the archived SVG figures are embedded below as fallbacks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TIME_STEPS = 100
ENCODER_WIDTH_DEFAULT = 4096
FORMATS = ("png", "pdf", "svg", "tiff")

ARCHITECTURES = (
    "Cloud Quantum",
    "PCIe Local",
    "MCM Chiplet",
    "Advanced CPO",
    "Monolithic TSV",
)
LATENCY_BENCHMARKS_MS = (50.0, 5.0, 0.5, 0.05, 0.0005)

COLOUR_PRIMARY = "#1f77b4"
COLOUR_DANGER = "#d62728"
COLOUR_NEUTRAL = "#7f7f7f"

# Clean isolated B=256 forward-pass timing used by the archived
# breakeven_scaling figure. This value was not written to a results CSV.
REFERENCE_BATCH_256_CORE_MS = 48.31963937916634

# Exact 100-point PQC latent drive used by the archived fig1_latent_drive.
REFERENCE_LATENT_DRIVE = np.array(
    [
        0.823418670, 0.823139482, 0.822860350, 0.822102004, 0.821334175,
        0.820266360, 0.819186919, 0.818008953, 0.816825797, 0.815689196,
        0.814557840, 0.813518931, 0.812491410, 0.811511805, 0.810539714,
        0.809529535, 0.808513931, 0.807392528, 0.806251460, 0.804995884,
        0.803715759, 0.802376441, 0.801023294, 0.799705498, 0.798399202,
        0.797227435, 0.796100014, 0.795188717, 0.794355803, 0.793804340,
        0.793365531, 0.793277015, 0.793343120, 0.793852677, 0.794577770,
        0.795867673, 0.797459536, 0.799744659, 0.802437235, 0.805918209,
        0.809907845, 0.814704411, 0.820071688, 0.826159291, 0.832804611,
        0.839977969, 0.847598186, 0.855477860, 0.863597263, 0.871681265,
        0.879731467, 0.887479653, 0.894903949, 0.901830128, 0.908177664,
        0.913918428, 0.918897678, 0.923238565, 0.926714590, 0.929569829,
        0.931516220, 0.932877112, 0.933310022, 0.933189261, 0.932115367,
        0.930512537, 0.927916041, 0.924820234, 0.920694282, 0.916122489,
        0.910530735, 0.904584758, 0.897724439, 0.890639894, 0.882873763,
        0.875034561, 0.866869024, 0.858775129, 0.850786917, 0.842980255,
        0.835712363, 0.828684084, 0.822547307, 0.816655626, 0.811867287,
        0.807290424, 0.803867214, 0.800602966, 0.798404744, 0.796312740,
        0.795116835, 0.793985359, 0.793562642, 0.793175985, 0.793340139,
        0.793522768, 0.794141044, 0.794766834, 0.795737319, 0.796708693,
    ],
    dtype=float,
)

# Exact (time step, encoder unit) events shown in raster_epoch_100.
REFERENCE_RASTER_EVENTS = np.array(
    [
        (14, 37), (15, 11), (16, 5), (16, 11), (16, 34), (16, 39),
        (16, 47), (16, 67), (17, 11), (17, 34), (17, 49), (17, 51),
        (17, 52), (17, 99), (18, 4), (18, 49), (18, 95), (19, 4),
        (19, 10), (19, 13), (19, 60), (19, 80), (20, 4), (20, 10),
        (20, 13), (20, 42), (20, 60), (20, 71), (20, 83), (21, 10),
        (21, 50), (21, 60), (21, 83), (22, 10), (22, 42), (22, 50),
        (22, 58), (22, 60), (22, 83), (23, 10), (24, 42), (24, 50),
        (24, 52), (25, 42), (25, 52), (25, 58), (26, 51), (26, 52),
        (26, 58), (26, 62), (27, 52), (27, 62), (28, 39), (29, 73),
        (30, 28), (30, 86), (31, 28), (31, 33), (31, 54), (31, 73),
        (32, 33), (34, 9), (34, 37), (34, 78), (34, 96), (35, 37),
        (35, 78), (35, 89), (35, 93), (35, 94), (36, 26), (36, 65),
        (36, 78), (36, 93), (36, 94), (37, 5), (37, 11), (37, 31),
        (37, 93), (38, 4), (38, 5), (38, 78), (38, 94), (39, 16),
        (40, 16), (40, 31), (40, 80), (41, 13), (41, 16), (41, 33),
        (41, 58), (41, 60), (41, 80), (41, 83), (42, 5), (42, 33),
        (42, 58), (42, 80), (43, 10), (43, 13), (43, 80), (44, 4),
        (44, 13), (44, 58), (44, 62), (45, 5), (45, 16), (46, 13),
        (46, 33), (47, 13), (47, 17), (47, 26), (47, 33), (47, 43),
        (47, 64), (48, 28), (50, 9), (50, 54), (51, 89), (60, 21),
        (67, 10), (67, 17), (79, 66), (80, 3), (84, 55), (84, 87),
        (88, 63),
    ],
    dtype=int,
)


def configure_matplotlib(dpi: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "axes.linewidth": 1.5,
            "lines.linewidth": 2.4,
            "figure.dpi": 200,
            "savefig.dpi": dpi,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        }
    )


def require_columns(frame: pd.DataFrame, columns: set[str], filename: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{filename} is missing required columns: {missing_text}")


def load_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_path = results_dir / "training_metrics.csv"
    sweep_path = results_dir / "sensitivity_sweep.csv"
    for path in (training_path, sweep_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required results file not found: {path}")

    training = pd.read_csv(training_path)
    sweep = pd.read_csv(sweep_path)
    require_columns(training, {"epoch", "epoch_seconds"}, training_path.name)
    require_columns(
        sweep,
        {"encoder_width", "batch_size", "mean_inf_per_sample_ms"},
        sweep_path.name,
    )
    return training, sweep


def load_latent_drive(results_dir: Path) -> tuple[np.ndarray, str]:
    path = results_dir / "latent_drive.csv"
    if not path.is_file():
        return REFERENCE_LATENT_DRIVE.copy(), "embedded archived latent drive"

    frame = pd.read_csv(path)
    drive_column = next(
        (name for name in ("drive", "drive_intensity", "latent_drive") if name in frame),
        None,
    )
    if drive_column is None:
        raise ValueError(
            f"{path.name} must contain drive, drive_intensity, or latent_drive"
        )
    if "time_step" in frame:
        frame = frame.sort_values("time_step")
    drive = frame[drive_column].to_numpy(dtype=float)
    if drive.size != TIME_STEPS:
        raise ValueError(f"{path.name} must contain exactly {TIME_STEPS} rows")
    return drive, str(path)


def load_raster_events(results_dir: Path) -> tuple[np.ndarray, str]:
    path = results_dir / "raster_epoch_100.csv"
    if not path.is_file():
        return REFERENCE_RASTER_EVENTS.copy(), "embedded archived raster events"

    frame = pd.read_csv(path)
    require_columns(frame, {"time_step", "encoder_unit"}, path.name)
    events = frame[["time_step", "encoder_unit"]].to_numpy(dtype=int)
    if np.any(events < 0) or np.any(events[:, 0] >= TIME_STEPS):
        raise ValueError(f"{path.name} contains out-of-range event coordinates")
    return events, str(path)


def load_breakeven_core_times(
    results_dir: Path, sweep: pd.DataFrame
) -> tuple[dict[int, float], str]:
    override_path = results_dir / "breakeven_core_times.csv"
    if override_path.is_file():
        frame = pd.read_csv(override_path)
        require_columns(frame, {"batch_size", "core_ms"}, override_path.name)
        core_times = {
            int(row.batch_size): float(row.core_ms) / 1000.0
            for row in frame.itertuples(index=False)
        }
        if not core_times:
            raise ValueError(f"{override_path.name} contains no timing rows")
        return core_times, str(override_path)

    default_width = sweep.loc[
        sweep["encoder_width"].eq(ENCODER_WIDTH_DEFAULT)
    ].copy()
    if default_width.empty:
        raise ValueError(
            f"sensitivity_sweep.csv has no encoder_width={ENCODER_WIDTH_DEFAULT} rows"
        )

    core_times: dict[int, float] = {}
    for batch_size in (1, 16, 64):
        row = default_width.loc[default_width["batch_size"].eq(batch_size)]
        if len(row) != 1:
            raise ValueError(
                "sensitivity_sweep.csv must contain one row for "
                f"encoder_width={ENCODER_WIDTH_DEFAULT}, batch_size={batch_size}"
            )
        per_sample_ms = float(row.iloc[0]["mean_inf_per_sample_ms"])
        core_times[batch_size] = per_sample_ms * batch_size / 1000.0

    core_times[256] = REFERENCE_BATCH_256_CORE_MS / 1000.0
    source = (
        "sensitivity_sweep.csv plus embedded clean batch-256 reference timing"
    )
    return core_times, source


def save_figure(fig: plt.Figure, output_dir: Path, name: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in FORMATS:
        fig.savefig(
            output_dir / f"{name}.{extension}",
            dpi=dpi,
            format=extension,
            bbox_inches="tight",
            pad_inches=0.18,
        )
    plt.close(fig)


def plot_latent_drive(drive: np.ndarray, output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.plot(drive, color=COLOUR_PRIMARY, linewidth=3.0)
    ax.fill_between(
        np.arange(drive.size), drive, color=COLOUR_PRIMARY, alpha=0.18
    )
    ax.set_title(
        r"PQC-Modulated Macroscopic Population Drive ($\lambda(t)$)",
        fontweight="bold",
    )
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Drive Intensity")
    save_figure(fig, output_dir, "fig1_latent_drive", dpi)


def plot_raster(
    events: np.ndarray, epoch: int, output_dir: Path, dpi: int
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.scatter(
        events[:, 0],
        events[:, 1],
        s=11,
        c=COLOUR_PRIMARY,
        marker="|",
        alpha=0.9,
    )
    ax.set_title(
        f"Encoder Raster Activity Snapshot (Epoch {epoch})", fontweight="bold"
    )
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Encoder Unit ID")
    ax.set_xlim(0, TIME_STEPS)
    ax.set_ylim(0, 100)
    save_figure(fig, output_dir, f"raster_epoch_{epoch}", dpi)


def plot_latency_overhead(
    core_seconds: float, output_dir: Path, dpi: int
) -> list[float]:
    overheads = []
    for latency_ms in LATENCY_BENCHMARKS_MS:
        latency_seconds = latency_ms / 1000.0
        overheads.append(
            latency_seconds / (core_seconds + latency_seconds) * 100.0
        )

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.plot(
        LATENCY_BENCHMARKS_MS,
        overheads,
        marker="o",
        markersize=10,
        color=COLOUR_PRIMARY,
        linewidth=2.8,
        zorder=3,
    )
    label_layout = {
        "Cloud Quantum": ((0.91, 0.92), "right"),
        "PCIe Local": ((0.78, 0.24), "left"),
        "MCM Chiplet": ((0.58, 0.16), "left"),
        "Advanced CPO": ((0.34, 0.10), "left"),
        "Monolithic TSV": ((0.09, 0.18), "left"),
    }
    for index, name in enumerate(ARCHITECTURES):
        text_position, alignment = label_layout[name]
        ax.annotate(
            name,
            (LATENCY_BENCHMARKS_MS[index], overheads[index]),
            fontsize=13,
            color=COLOUR_NEUTRAL,
            xytext=text_position,
            textcoords="axes fraction",
            ha=alignment,
            va="center",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": COLOUR_NEUTRAL,
                "linewidth": 0.8,
                "alpha": 0.8,
            },
            annotation_clip=False,
        )
    ax.set_xscale("log")
    ax.margins(x=0.08, y=0.18)
    ax.set_title("QPU I/O Latency vs. System Overhead", fontweight="bold")
    ax.set_xlabel("Hardware Integration Latency (ms) [Log Scale]")
    ax.set_ylabel("QPU Synchronisation Penalty (%)")
    save_figure(fig, output_dir, "latency_overhead", dpi)
    return overheads


def plot_breakeven(
    core_times: dict[int, float],
    output_dir: Path,
    dpi: int,
    threshold_pct: float = 1.0,
) -> None:
    batches = sorted(core_times)
    fraction = threshold_pct / 100.0
    breakeven_ms = [
        fraction / (1.0 - fraction) * core_times[batch] * 1000.0
        for batch in batches
    ]

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.plot(
        batches,
        breakeven_ms,
        marker="s",
        markersize=8,
        linewidth=2.6,
        color=COLOUR_DANGER,
        label=f"Break-even ({threshold_pct}%)",
    )
    for architecture, latency_ms in zip(
        ARCHITECTURES, LATENCY_BENCHMARKS_MS
    ):
        ax.axhline(
            latency_ms,
            linestyle=":",
            linewidth=1.0,
            color=COLOUR_NEUTRAL,
            alpha=0.7,
        )
        ax.annotate(
            architecture,
            xy=(0.02, latency_ms),
            xycoords=("axes fraction", "data"),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=13,
            ha="left",
            va="bottom",
            color=COLOUR_NEUTRAL,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
            },
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Inference Batch Size [Log Scale]")
    ax.set_ylabel(
        r"Maximum Tolerable QPU Round-Trip $\tau_{QPU}^*$ (ms) [Log Scale]"
    )
    ax.set_title(
        f"Hardware Selection Frontier ({threshold_pct}% Overhead Budget)",
        fontweight="bold",
    )
    ax.legend(loc="upper right", framealpha=0.95)
    save_figure(fig, output_dir, "breakeven_scaling", dpi)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=script_dir / "results_csv",
        help="Directory containing the saved benchmark CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results_figures_regenerated",
        help="Directory for regenerated PNG, PDF, SVG, and TIFF files.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Raster output resolution. The original figures used 600 DPI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    configure_matplotlib(args.dpi)

    training, sweep = load_results(results_dir)
    latent_drive, latent_source = load_latent_drive(results_dir)
    raster_events, raster_source = load_raster_events(results_dir)
    core_times, breakeven_source = load_breakeven_core_times(
        results_dir, sweep
    )

    epoch = int(training["epoch"].max())
    if epoch != 100:
        raise ValueError(
            "training_metrics.csv must include epoch 100 to create "
            "raster_epoch_100"
        )

    # This is the exact effective T_core represented in the archived
    # latency_overhead figure.
    latency_core_seconds = float(training["epoch_seconds"].median())

    plot_breakeven(core_times, output_dir, args.dpi)
    plot_latent_drive(latent_drive, output_dir, args.dpi)
    overheads = plot_latency_overhead(
        latency_core_seconds, output_dir, args.dpi
    )
    plot_raster(raster_events, epoch, output_dir, args.dpi)

    print(f"Loaded results from: {results_dir}")
    print(f"Saved 16 files to:   {output_dir}")
    print(f"Latent source:       {latent_source}")
    print(f"Raster source:       {raster_source}")
    print(f"Break-even source:   {breakeven_source}")
    print(f"Latency T_core:      {latency_core_seconds:.9f} s")
    print(
        "Latency overheads:   "
        + ", ".join(f"{value:.9g}%" for value in overheads)
    )


if __name__ == "__main__":
    main()
