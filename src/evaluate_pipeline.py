"""Offline session-grouped training/evaluation; synthetic results are never field results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from src.models_ml import (TERRMClassifier, cnn_probabilities, make_random_forest,
                           save_random_forest, train_cnn)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_WATERMARK = "[SYNTHETIC 60 GHz BENCHMARK - NOT REAL HARDWARE DATA]"


def center_burst_indices(num_bursts: int, training: bool = True) -> np.ndarray:
    """Middle floor(70%*n) training bursts; zero-based n//2 for single-burst test."""
    if num_bursts < 1:
        raise ValueError("At least one burst is required")
    if not training:
        return np.asarray([num_bursts // 2], dtype=int)
    count = max(1, int(np.floor(0.7 * num_bursts)))
    start = (num_bursts - count) // 2
    return np.arange(start, start + count)


class ScanTrackFilter:
    """Oh Algorithm 1 label history, with abstention for the first three scans.

    Deterministic ties follow the supplied class order. Confidence supplied by
    this project is an uncalibrated classifier score in [0,1]. A file is one
    continuous single-target track; state must not cross unrelated recordings.
    """
    def __init__(self, classes):
        self.classes = [str(c) for c in classes]
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError("classes must be nonempty and unique")
        self.reset()

    def reset(self):
        self.scores = dict.fromkeys(self.classes, 0.0)
        self.history = []
        self.current = None
        self.last_time_s = None

    def update(self, label: str, confidence: float, *, time_s: float | None = None,
               max_gap_s: float | None = None) -> str | None:
        label = str(label)
        if label not in self.scores or not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Unknown label or invalid confidence")
        if time_s is not None:
            if not np.isfinite(time_s) or max_gap_s is None or not np.isfinite(max_gap_s) or max_gap_s <= 0:
                raise ValueError("Timed tracking needs a finite timestamp and positive max_gap_s")
            if self.last_time_s is not None:
                delta = time_s - self.last_time_s
                if delta <= 0:
                    raise ValueError("Track timestamps must be strictly increasing")
                if delta > max_gap_s:
                    self.reset()
            self.last_time_s = float(time_s)
        self.scores[label] += float(confidence)
        self.history.append(label)
        if len(self.history) < 4:
            return None
        if len(self.history) == 4:
            self.current = max(self.classes, key=self.history.count)
        elif label != self.current:
            self.current = max(self.classes, key=self.scores.get)
        return self.current


def apply_scan_to_scan_filter(labels, confidences, classes=None):
    classes = sorted(set(map(str, labels))) if classes is None else classes
    tracker = ScanTrackFilter(classes)
    if len(labels) != len(confidences):
        raise ValueError("labels and confidences must align")
    return [tracker.update(label, float(confidence)) for label, confidence in zip(labels, confidences)]


def assert_no_session_leakage(partitions: dict) -> None:
    sets = {name: {row["session_id"] for row in rows} for name, rows in partitions.items()}
    names = list(sets)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            if sets[name] & sets[other]:
                raise ValueError(f"Session leakage between {name} and {other}")


def _json_value(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_value, allow_nan=False), encoding="utf-8")


def config_signature(cfg) -> dict:
    names = ["start_frequency_Hz", "end_frequency_Hz", "sample_rate_Hz",
             "num_samples_per_chirp", "num_chirps_per_frame", "chirp_repetition_time_s",
             "frame_repetition_time_s", "rx_mask", "slope_Hz_s"]
    return {name: getattr(cfg, name) for name in names}


def _extract(rows, partition: str, output_root: Path):
    from src.inspect_raw_recording import load_recording
    from src.radar_dsp import extract_burst_features, FEATURE_NAMES

    features, images, labels, records, configs = [], [], [], [], []
    feature_dir = output_root / "data" / "processed" / "features"
    image_dir = output_root / "data" / "processed" / "spectrograms"
    feature_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    feature_rows = []
    for row in rows:
        recording = load_recording(row["filepath"], config_path=row.get("config_path") or None)
        actual_synthetic = bool(recording.metadata.get("is_synthetic", False))
        if actual_synthetic != bool(row["is_synthetic"]):
            raise ValueError(f"Registry/sidecar synthetic provenance disagrees: {row['filepath']}")
        signature = config_signature(recording.config)
        if signature not in configs:
            configs.append(signature)
        selected = set(center_burst_indices(len(recording.frames)).tolist()) if partition == "train" else set(range(len(recording.frames)))
        sample_features, sample_images, sample_ft = [], [], []
        for frame_index, burst in enumerate(recording.frames):
            result = extract_burst_features(burst, recording.config, include_ftmdse=True)
            vector = np.asarray(result["features"], dtype=np.float64)
            spec = np.asarray(result["spectrogram"], dtype=np.float32)
            if vector.shape != (46,) or not np.isfinite(vector).all() or spec.shape != (32, 32):
                raise ValueError("DSP returned invalid feature/spectrogram shape or nonfinite features")
            sample_features.append(vector)
            sample_images.append(spec)
            sample_ft.append(result["ftmdse_spectrogram"])
            metadata = {"recording_id": row["recording_id"], "session_id": row["session_id"],
                "frame_index": frame_index, "label": row["label"], "is_synthetic": actual_synthetic,
                "time_s": recording.metadata["frame_timestamps_s"][frame_index],
                "track_max_gap_s": 1.5 * recording.config.frame_repetition_time_s,
                "partition": partition, "selected": frame_index in selected,
                "center_burst": frame_index == len(recording.frames) // 2,
                "target_range_m": float(result["dsp"]["target_range_m"]),
                "pruning": result["dsp"]["pruning"]}
            feature_rows.append({**{k: v for k, v in metadata.items() if k != "pruning"},
                                 **dict(zip(FEATURE_NAMES, vector.tolist()))})
            if frame_index in selected:
                features.append(vector)
                images.append(spec)
                labels.append(row["label"])
                records.append(metadata)
        digest = hashlib.sha256(str(Path(row["filepath"]).resolve()).encode()).hexdigest()[:12]
        name = f"{partition}_{digest}"
        np.save(feature_dir / f"{name}.npy", np.asarray(sample_features))
        np.save(image_dir / f"{name}_stft.npy", np.asarray(sample_images))
        np.save(image_dir / f"{name}_ftmdse.npy", np.asarray(sample_ft))
        write_json(feature_dir / f"{name}.json", {"recording": row, "config": signature,
            "feature_names": FEATURE_NAMES, "is_synthetic": actual_synthetic})
        write_json(image_dir / f"{name}.json", {"recording_id": row["recording_id"],
            "is_synthetic": actual_synthetic, "stft": "3-RX noncoherent slow-time STFT",
            "ftmdse": "Experimental interpolated IF; missing idle-time samples are not measured"})
    if feature_rows:
        with (feature_dir / f"{partition}_features.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0]))
            writer.writeheader()
            writer.writerows(feature_rows)
    if not features:
        raise ValueError(f"No samples in {partition} partition")
    return {"x": np.asarray(features), "images": np.asarray(images), "y": np.asarray(labels),
            "records": records, "configs": configs}


def _metrics(truth, predicted, classes):
    if not len(truth):
        return {"n": 0, "accuracy": None, "balanced_accuracy": None, "macro_f1": None, "confusion_matrix": None}
    return {"n": len(truth), "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=classes, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=classes).tolist(),
        "classification_report": classification_report(truth, predicted, labels=classes,
            output_dict=True, zero_division=0)}


def evaluate_predictions(data: dict, probabilities: np.ndarray, classes, csv_path: Path) -> dict:
    classes = np.asarray(classes)
    if probabilities.shape != (len(data["y"]), len(classes)) or not np.isfinite(probabilities).all():
        raise ValueError("Prediction shape or finite check failed")
    predicted = classes[np.argmax(probabilities, axis=1)]
    confidences = probabilities.max(axis=1)
    trackers, previous_frames = {}, {}
    filtered, output_rows, center = [], [], []
    for index, (row, label, confidence) in enumerate(zip(data["records"], predicted, confidences)):
        key = (row["session_id"], row["recording_id"])
        # Separate files and missing frames reset state; no held-out/training history is mixed.
        if key not in trackers or row["frame_index"] != previous_frames[key] + 1:
            trackers[key] = ScanTrackFilter(classes)
        previous_frames[key] = row["frame_index"]
        tracked = trackers[key].update(str(label), float(confidence),
            time_s=row.get("time_s"), max_gap_s=row.get("track_max_gap_s"))
        filtered.append(tracked)
        if row["center_burst"]:
            center.append(index)
        output_rows.append({"session_id": row["session_id"], "recording_id": row["recording_id"],
            "frame_index": row["frame_index"], "time_s": row.get("time_s"), "truth": row["label"], "predicted": str(label),
            "confidence_score": float(confidence), "track_label": tracked or "warmup",
            "is_synthetic": row["is_synthetic"], "pruning": json.dumps(row["pruning"], default=_json_value)})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    eligible = np.asarray([i for i, label in enumerate(filtered) if label is not None], dtype=int)
    filtered_values = np.asarray([filtered[i] for i in eligible])
    return {"single_scan": _metrics(data["y"], predicted, classes),
        "center_burst_per_recording": _metrics(data["y"][center], predicted[center], classes),
        "track_filtered": _metrics(data["y"][eligible], filtered_values, classes),
        "single_scan_on_track_eligible_frames": _metrics(data["y"][eligible], predicted[eligible], classes),
        "track_warmup_abstentions": len(filtered) - len(eligible),
        "track_coverage": float(len(eligible) / len(filtered)),
        "pruning_policy": "flags only; no forced mapping from paper non-UAV to four task classes"}


def _plot_matrix(matrix, classes, path, title, synthetic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes,
           yticklabels=classes, xlabel="Predicted", ylabel="True", title=title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, label="Frames")
    if synthetic:
        fig.text(0.5, 0.02, SYNTHETIC_WATERMARK, ha="center", color="darkred", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_evaluation(registry: Path, output_root: Path, allow_synthetic=False,
                   seed=42, model_names=("rf", "terrm", "cnn"), epochs=12) -> dict:
    from src.dataset_manager import read_registry, split_sessions
    from src.radar_dsp import FEATURE_NAMES

    rows = read_registry(registry, allow_synthetic=allow_synthetic)
    if not rows:
        raise ValueError("The registry is empty")
    partitions = split_sessions(rows, seed=seed, train_fraction=0.6, val_fraction=0.2)
    assert_no_session_leakage(partitions)
    data = {name: _extract(partitions[name], name, output_root) for name in ("train", "val", "test")}
    classes = np.unique(data["train"]["y"])
    for name in ("val", "test"):
        if set(data[name]["y"]) != set(classes):
            raise ValueError(f"{name} must contain every training class; collect more independent sessions")
    synthetic = any(row["is_synthetic"] for row in rows)
    synthetic_only = all(row["is_synthetic"] for row in rows)
    run_type = "synthetic_only" if synthetic_only else "mixed_synthetic_real" if synthetic else "real_recordings"
    results_dir = output_root / "results" / ("synthetic_benchmark" if synthetic else "evaluation")
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"schema_version": 1, "is_synthetic": synthetic, "run_type": run_type,
        "classes": classes.tolist(), "feature_names": list(FEATURE_NAMES), "feature_dimension": 46,
        "input_shape": [32, 32], "training_configs": data["train"]["configs"],
        "train_session_ids": sorted({r["session_id"] for r in partitions["train"]}),
        "validation_session_ids": sorted({r["session_id"] for r in partitions["val"]}),
        "test_session_ids": sorted({r["session_id"] for r in partitions["test"]}),
        "seed": seed, "created_utc": datetime.now(timezone.utc).isoformat(),
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
        "confidence": "uncalibrated classifier score; not probability of physical UAV presence",
        "track_gap_policy": "reset after >1.5 configured frame periods in supplied or nominal timestamps",
        "limitations": ["Synthetic scores do not establish hardware detection performance.",
            "Independent sessions are required; adjacent frames are not independent experiments.",
            "Each file is assumed to contain one continuous single-target track.",
            "Frames are scan surrogates, distinct from Oh2019 mechanically scanned radar tracks.",
            "Pruning is diagnostic only; the four labels differ from the paper's categories."]}
    write_json(results_dir / "split_manifest.json", {"partitions": partitions, "metadata": metadata,
        "sample_counts": {k: len(v["y"]) for k, v in data.items()}})
    summary = {"is_synthetic": synthetic, "run_type": run_type,
        "watermark": SYNTHETIC_WATERMARK if synthetic else "Recorded data; provenance supplied by registry",
        "classes": classes.tolist(), "seed": seed, "models": {}, "metadata": metadata}
    for name in model_names:
        model_metadata = dict(metadata, model_type=name)
        print(f"Training {name} on {len(data['train']['y'])} bursts; evaluating held-out sessions", flush=True)
        if name == "rf":
            model = make_random_forest(seed).fit(data["train"]["x"], data["train"]["y"])
            checkpoint = models_dir / "rf_bgt60tr13c.joblib"
            save_random_forest(model, checkpoint, model_metadata)
            val_probability = model.predict_proba(data["val"]["x"])
            test_probability = model.predict_proba(data["test"]["x"])
            model_classes = model.classes_
            importance = model.named_steps["classifier"].feature_importances_
            with (results_dir / "rf_feature_importance.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["feature", "importance", "is_synthetic"])
                writer.writerows((FEATURE_NAMES[i], float(importance[i]), synthetic) for i in np.argsort(-importance))
            _plot_importance(importance, FEATURE_NAMES, results_dir, synthetic)
        elif name == "terrm":
            model = TERRMClassifier(order=2).fit(data["train"]["x"], data["train"]["y"])
            model_metadata["terrm"] = {"order": 2, "regularization": 1e-4, "tau": 0.5, "eta": 0.5,
                "equation": "Oh2019 Eq3 class-balanced weighted solve",
                "basis": "Toh2004 reduced multivariate, 141 columns for 46 features",
                "departure": "0/1 targets; avoids identical targets implied by printed tau=1,eta=0"}
            checkpoint = models_dir / "terrm_bgt60tr13c.npz"
            model.save(checkpoint, model_metadata)
            val_probability = model.predict_proba(data["val"]["x"])
            test_probability = model.predict_proba(data["test"]["x"])
            model_classes = model.classes_
        elif name == "cnn":
            checkpoint = models_dir / "cnn_bgt60tr13c.pt"
            model, history, model_classes = train_cnn(data["train"]["images"], data["train"]["y"],
                data["val"]["images"], data["val"]["y"], checkpoint, model_metadata,
                epochs=epochs, seed=seed)
            val_probability = cnn_probabilities(model, data["val"]["images"])
            test_probability = cnn_probabilities(model, data["test"]["images"])
            write_json(results_dir / "cnn_history.json", {"is_synthetic": synthetic, "history": history})
            _plot_history(history, results_dir, synthetic)
        else:
            raise ValueError(f"Unknown model {name}")
        write_json(checkpoint.with_suffix(checkpoint.suffix + ".json"), model_metadata)
        val_results = evaluate_predictions(data["val"], val_probability, model_classes,
                                           results_dir / f"{name}_validation_predictions.csv")
        test_results = evaluate_predictions(data["test"], test_probability, model_classes,
                                            results_dir / f"{name}_test_predictions.csv")
        summary["models"][name] = {"checkpoint": str(checkpoint.resolve()),
                                    "validation": val_results, "test": test_results}
        for variant in ("single_scan", "track_filtered"):
            matrix = test_results[variant]["confusion_matrix"]
            if matrix is not None:
                _plot_matrix(matrix, model_classes, results_dir / f"{name}_{variant}_confusion.png",
                             f"{name.upper()} held-out sessions: {variant.replace('_', ' ')}", synthetic)
        print(f"{name}: test single-scan accuracy={test_results['single_scan']['accuracy']:.3f}; "
              f"track accuracy={test_results['track_filtered']['accuracy']}", flush=True)
    write_json(results_dir / "metrics.json", summary)
    (results_dir / "README.txt").write_text(
        summary["watermark"] + "\nSplit by session before processing. All preprocessing fits training only.\n"
        "Track accuracy excludes the first three frames of each recording; coverage is in metrics.json.\n"
        "CNN checkpoint is chosen using validation loss; test samples never choose checkpoints.\n"
        "Report these as software verification results, not hardware performance claims.\n", encoding="utf-8")
    return summary


def _plot_importance(values, names, directory, synthetic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    indices = np.argsort(values)[-15:]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([names[i] for i in indices], values[indices])
    ax.set(xlabel="RF impurity importance (training)", title="Top 15 feature importances")
    if synthetic:
        fig.text(0.5, .015, SYNTHETIC_WATERMARK, ha="center", fontsize=8, color="darkred")
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(directory / "rf_feature_importance.png", dpi=150)
    plt.close(fig)


def _plot_history(history, directory, synthetic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], label="Training")
    ax.plot([r["epoch"] for r in history], [r["val_loss"] for r in history], label="Validation")
    ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="CNN validation-selected training")
    ax.legend()
    if synthetic:
        fig.text(.5, .015, SYNTHETIC_WATERMARK, ha="center", fontsize=8, color="darkred")
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(directory / "cnn_loss.png", dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--registry", type=Path)
    source.add_argument("--run-synthetic-benchmark", action="store_true")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "bgt60tr13c_config.json")
    parser.add_argument("--models", nargs="+", choices=("rf", "terrm", "cnn"), default=["rf", "terrm", "cnn"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sessions-per-class", type=int, default=5)
    parser.add_argument("--frames-per-recording", type=int, default=6)
    args = parser.parse_args(argv)
    try:
        if args.run_synthetic_benchmark:
            from src.config import RadarConfig
            from src.dataset_manager import generate_synthetic_pretrain_dataset
            print(SYNTHETIC_WATERMARK, flush=True)
            registry = generate_synthetic_pretrain_dataset(args.output_root, cfg=RadarConfig.from_file(args.config),
                sessions_per_class=args.sessions_per_class, frames_per_recording=args.frames_per_recording, seed=args.seed)
        else:
            registry = args.registry
        run_evaluation(Path(registry), args.output_root.resolve(), args.allow_synthetic or args.run_synthetic_benchmark,
                       args.seed, args.models, args.epochs)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")
    print(f"Saved metrics and model artifacts under {args.output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

