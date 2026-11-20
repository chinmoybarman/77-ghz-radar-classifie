"""Inspect and plot one measurement/segment from the verified KTH 77 GHz archive."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.kth77_dataset import load_dataset, target_signal, extract_segment_features


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--measurement", type=int, default=0, help="zero-based measurement row")
    parser.add_argument("--segment", type=int, default=0, help="zero-based segment column")
    parser.add_argument("--output", type=Path, default=Path("results/kth77_view.png"))
    args = parser.parse_args(argv)

    measurements, fingerprint = load_dataset(args.input)
    if not 0 <= args.measurement < len(measurements):
        parser.error(f"measurement must be between 0 and {len(measurements)-1}")
    measurement = measurements[args.measurement]
    if not 0 <= args.segment < measurement.samples.shape[1]:
        parser.error(f"segment must be between 0 and {measurement.samples.shape[1]-1}")
    if measurement.edge[args.segment]:
        parser.error("that segment is marked edge/truncated; choose another segment")

    signal = target_signal(measurement, args.segment)
    extracted = extract_segment_features(signal)
    print("Verified archive:", fingerprint["path"])
    print("MD5:", fingerprint["md5"])
    print("Measurement:", measurement.identity)
    print("Source label:", measurement.source_label)
    print("Project class:", measurement.label)
    print("Raw complex matrix shape:", measurement.samples.shape)
    print("Segments in this measurement:", measurement.samples.shape[1])
    print("Selected segment:", args.segment)
    print("Timestamp (s):", measurement.times_s[args.segment])
    print("Range (m):", measurement.ranges_m[args.segment])
    print("Author split code:", measurement.author_split[args.segment])
    print("Edge/truncated:", bool(measurement.edge[args.segment]))
    print("Selected signal shape:", signal.shape)
    print("Feature shape:", extracted["features"].shape)
    print("Spectrogram shape:", extracted["spectrogram"].shape)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    time_ms = np.arange(len(extracted["signal"])) / 17_000 * 1000
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(time_ms, extracted["signal"].real, label="Real / I", lw=1)
    axes[0].plot(time_ms, extracted["signal"].imag, label="Imaginary / Q", lw=1, alpha=.8)
    axes[0].set(xlabel="Slow-time within selected segment (ms)", ylabel="Normalized amplitude",
                title=f"{measurement.identity} · {measurement.source_label} · segment {args.segment}")
    axes[0].legend()
    image = axes[1].pcolormesh(extracted["time_s"] * 1000, extracted["frequency_hz"] / 1000,
        extracted["spectrogram"], shading="auto", cmap="magma", vmin=0, vmax=1)
    axes[1].set(xlabel="STFT window center (ms)", ylabel="Doppler frequency (kHz)",
                title="32 × 32 normalized micro-Doppler image")
    fig.colorbar(image, ax=axes[1], label="Normalized power")
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print("Saved plot:", args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

