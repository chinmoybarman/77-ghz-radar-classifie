"""Single-recording inference CLI and optional local Streamlit dashboard.

Example: python src/predict_offline.py --input recording.npy --model models/rf_bgt60tr13c.joblib
Dashboard: python src/predict_offline.py --streamlit
Only load model artifacts you trust: joblib is a Python executable serialization.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.evaluate_pipeline import (PROJECT_ROOT, SYNTHETIC_WATERMARK, ScanTrackFilter,
                                   center_burst_indices, config_signature, write_json)
from src.models_ml import load_model, model_probabilities


def _check_compatibility(metadata, config, feature_names):
    if metadata.get("domain", "bgt60tr13c") != "bgt60tr13c":
        raise ValueError("This is an external-dataset model. Use the KTH 77 GHz workflow; BGT60TR13C prediction requires its own model.")
    if metadata.get("feature_dimension") != 46 or list(metadata.get("feature_names", [])) != list(feature_names):
        raise ValueError("Model feature schema differs from this DSP implementation; retrain the model")
    signature = config_signature(config)
    trained = metadata.get("training_configs", [])
    if not trained:
        raise ValueError("Model does not contain training acquisition settings; retrain with this pipeline")
    compatible = any(all(np.isclose(signature[k], candidate.get(k, float("nan")), rtol=1e-6, atol=1e-12)
                         for k in signature) for candidate in trained)
    if not compatible:
        raise ValueError("Recording acquisition settings differ from model training settings; train a compatible model")


def predict_recording(input_path: str | Path, model_path: str | Path,
                      config_path: str | Path | None = None,
                      output_dir: str | Path = PROJECT_ROOT / "results" / "predictions") -> dict:
    from src.inspect_raw_recording import load_recording
    from src.radar_dsp import extract_burst_features, FEATURE_NAMES

    input_path, model_path, output_dir = Path(input_path), Path(model_path), Path(output_dir)
    recording = load_recording(input_path, config_path=config_path)
    bundle = load_model(model_path)
    _check_compatibility(bundle["metadata"], recording.config, FEATURE_NAMES)
    results = [extract_burst_features(frame, recording.config, include_ftmdse=True) for frame in recording.frames]
    features = np.stack([r["features"] for r in results])
    spectrograms = np.stack([r["spectrogram"] for r in results])
    probabilities = model_probabilities(bundle, features, spectrograms)
    if not np.isfinite(probabilities).all():
        raise ValueError("Model produced nonfinite scores")
    classes = np.asarray(bundle["classes"])
    labels = classes[probabilities.argmax(axis=1)]
    scores = probabilities.max(axis=1)
    source_synthetic = bool(recording.metadata.get("is_synthetic", False))
    trained_synthetic = bool(bundle["metadata"].get("is_synthetic", False))
    synthetic_context = source_synthetic or trained_synthetic
    tracker = ScanTrackFilter(classes)
    rows = []
    for index, (label, score, result) in enumerate(zip(labels, scores, results)):
        time_s = recording.metadata["frame_timestamps_s"][index]
        tracked = tracker.update(str(label), float(score), time_s=time_s,
            max_gap_s=1.5 * recording.config.frame_repetition_time_s)
        row = {"frame_index": index, "time_s": time_s,
            "predicted_class": str(label), "confidence_score": float(score),
            "track_label": tracked or "warmup", "estimated_range_m": float(result["dsp"]["target_range_m"]),
            "target_bin": int(result["dsp"]["target_bin"]), "input_is_synthetic": source_synthetic,
            "model_trained_with_synthetic": trained_synthetic,
            "pruning_flags": json.dumps(result["dsp"]["pruning"], default=lambda x: x.item())}
        row.update({f"score_{name}": float(p) for name, p in zip(classes, probabilities[index])})
        rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.name.replace(".", "_")
    csv_path = output_dir / f"{stem}_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    center = int(center_burst_indices(len(recording.frames), training=False)[0])
    plot_path = output_dir / f"{stem}_range_velocity_spectrogram.png"
    _plot_prediction(results[center], recording.config, plot_path,
                     f"Center frame {center}: {labels[center]} (score {scores[center]:.3f})",
                     synthetic_context, source_synthetic, trained_synthetic)
    np.save(output_dir / f"{stem}_stft.npy", spectrograms)
    np.save(output_dir / f"{stem}_ftmdse.npy", np.stack([r["ftmdse_spectrogram"] for r in results]))
    summary = {"input": str(input_path.resolve()), "model": str(model_path.resolve()),
        "predicted_class": str(labels[center]), "confidence_score": float(scores[center]),
        "summary_policy": "center burst prediction, zero-based n//2",
        "center_frame_index": center, "frame_count": len(rows),
        "last_track_label": rows[-1]["track_label"], "input_is_synthetic": source_synthetic,
        "model_trained_with_synthetic": trained_synthetic,
        "notice": SYNTHETIC_WATERMARK if source_synthetic else
            "Model trained with synthetic data; real-hardware performance is unvalidated" if trained_synthetic else
            "Classifier scores are uncalibrated and require independent field validation",
        "csv": str(csv_path.resolve()), "plot": str(plot_path.resolve()),
        "config": config_signature(recording.config),
        "confidence_meaning": "uncalibrated classifier score; not a physical detection probability",
        "timestamps_source": recording.metadata["timestamps_source"],
        "track_gap_policy": "reset when timestamp gap exceeds 1.5 configured frame periods",
        "quality": recording.quality, "rows": rows}
    write_json(output_dir / f"{stem}_prediction.json", summary)
    return summary


def _plot_prediction(result, cfg, path, title, synthetic, input_synthetic, trained_synthetic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dsp = result["dsp"]
    ranges = np.asarray(dsp["range_axis_m"])
    doppler = np.asarray(dsp["doppler_axis_hz"])
    velocity = doppler * cfg.wavelength_m / 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im = axes[0].pcolormesh(ranges, velocity, dsp["rv_db"], shading="auto", cmap="viridis")
    axes[0].axvline(dsp["target_range_m"], color="white", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Range (m)", ylabel="Radial velocity (m/s; mixer sign uncalibrated)",
                title="3-RX range–velocity power")
    fig.colorbar(im, ax=axes[0], label="Power (dB)")
    im2 = axes[1].pcolormesh(result["spectrogram_time_s"]*1000,
        result["spectrogram_frequency_hz"], result["spectrogram"], shading="auto", cmap="magma")
    axes[1].set(xlabel="STFT window center within frame (ms)", ylabel="Doppler (Hz)",
                title="3-RX slow-time STFT (resized 32 × 32)")
    fig.colorbar(im2, ax=axes[1], label="Normalized spectrogram")
    fig.suptitle(title)
    if synthetic:
        # Keep the requested benchmark watermark on all synthetic pretraining outputs,
        # and specify separately whether the recording itself was synthetic.
        fig.text(.5, .04, SYNTHETIC_WATERMARK, ha="center", fontsize=8, color="darkred")
        fig.text(.5, .012, f"Input synthetic: {input_synthetic}; model trained with synthetic: {trained_synthetic}",
                 ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .08 if synthetic else 0, 1, .94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def dashboard():
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt to use the Streamlit dashboard") from exc
    st.set_page_config(page_title="BGT60TR13C Offline Radar", layout="wide")
    st.title("60 GHz radar recording classifier")
    st.caption("Inspect one finite recording using a locally trained model. Scores are not calibrated detection probabilities.")
    source = st.text_input("Recording path (.npy, .raw.bin, .raw.txt)")
    model = st.text_input("Trusted model path", str(PROJECT_ROOT / "models" / "rf_bgt60tr13c.joblib"))
    config = st.text_input("Config path (leave empty for companion metadata)")
    destination = st.text_input("Output folder", str(PROJECT_ROOT / "results" / "predictions"))
    if st.button("Analyze recording", type="primary"):
        if not source:
            st.error("Enter a recording path.")
            return
        try:
            with st.spinner("Processing radar frames"):
                result = predict_recording(source, model, config or None, destination)
            st.info(result["notice"])
            st.metric("Center-frame class", result["predicted_class"])
            st.metric("Confidence score", f"{result['confidence_score']:.3f}")
            st.image(result["plot"])
            st.dataframe(result["rows"], width="stretch")
            st.download_button("Download predictions CSV", Path(result["csv"]).read_bytes(),
                               file_name=Path(result["csv"]).name, mime="text/csv")
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            st.error(str(exc))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "rf_bgt60tr13c.joblib")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "predictions")
    parser.add_argument("--streamlit", action="store_true", help="Launch the optional local dashboard")
    parser.add_argument("--dashboard", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.dashboard:
        dashboard()
        return 0
    if args.streamlit:
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()),
                                "--", "--dashboard"])
    if args.input is None:
        parser.error("--input is required unless --streamlit is selected")
    try:
        result = predict_recording(args.input, args.model, args.config, args.output_dir)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(2, f"Prediction failed: {exc}\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "quality")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

