"""Sample input and inference contracts for the local KTH 77 GHz dashboard."""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
import zipfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.kth77_dataset import (CARRIER_HZ, CLASSES, CROP, FEATURE_NAMES,
    FEATURE_VERSION, PRF_HZ, load_dataset, target_signal, extract_segment_features)
from src.models_ml import model_probabilities

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "kth77_first"
REFINEMENT_ROOT = EXPERIMENT_ROOT.parent / "kth77_refined"
FUSION_ROOT = EXPERIMENT_ROOT.parent / "kth77_fusion"
REFINED_ROOT = FUSION_ROOT if (FUSION_ROOT / "metrics.json").is_file() else REFINEMENT_ROOT
DOMAIN = "kth77_range_compressed"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024


def central_signal(values):
    """One complex segment: 150/256 sweeps or five concatenated range cells."""
    values = np.asarray(values)
    if values.dtype.kind != "c" or not np.isfinite(values).all():
        raise ValueError("Use finite complex radar samples (real + imaginary); real-only arrays are unsupported.")
    if values.shape == (150,):
        result = values
    elif values.shape == (256,):
        result = values[CROP]
    elif values.shape == (5, 256):
        result = values[2, CROP]
    elif values.shape == (1280,):
        result = values[512:768][CROP]
    else:
        raise ValueError("Expected one complex segment shaped (150,), (256,), (5,256), or (1280,).")
    return np.asarray(result, dtype=np.complex128).copy()


def _validate_metadata(metadata):
    if not isinstance(metadata, dict) or metadata.get("domain") != DOMAIN:
        raise ValueError("This sample must declare the KTH 77 GHz range-compressed signal format.")
    for key, expected in (("prf_hz", PRF_HZ), ("carrier_hz", CARRIER_HZ)):
        value = metadata.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or not np.isclose(value, expected, rtol=1e-6, atol=0):
            raise ValueError(f"Sample {key} differs from the trained model ({expected:g}).")
    if metadata.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Sample preprocessing version differs from the trained model.")


def sample_metadata():
    return {"domain": DOMAIN, "prf_hz": PRF_HZ, "carrier_hz": CARRIER_HZ,
            "feature_version": FEATURE_VERSION}


def read_upload(content, filename):
    """Read numeric sample uploads only; never enable pickle for browser files."""
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Choose a sample smaller than 8 MB; browse the large source archive through Recorded samples.")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".npy", ".npz"}:
        raise ValueError("Choose a .npy sample or an exported .npz sample.")
    if suffix == ".npz":
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                if sum(info.file_size for info in archive.infolist()) > MAX_UPLOAD_BYTES:
                    raise ValueError("Expanded sample exceeds the 8 MB limit.")
        except zipfile.BadZipFile as exc:
            raise ValueError("The sample is not a valid NumPy archive.") from exc
    try:
        loaded = np.load(BytesIO(content), allow_pickle=False)
        if suffix == ".npy":
            if not isinstance(loaded, np.ndarray):
                if hasattr(loaded, "close"):
                    loaded.close()
                raise ValueError("Expected a numeric .npy sample.")
            return central_signal(loaded), {**sample_metadata(), "settings_assumed": True}
        if not isinstance(loaded, np.lib.npyio.NpzFile):
            raise ValueError("Expected an exported .npz sample.")
        with loaded:
            if not {"signal", "metadata_json"} <= set(loaded.files):
                raise ValueError("An exported .npz must contain signal and metadata_json.")
            metadata = json.loads(str(loaded["metadata_json"].item()))
            _validate_metadata(metadata)
            record = metadata.get("source_record")
            if record is not None and (not isinstance(record, dict)
                    or record.get("label") not in CLASSES):
                raise ValueError("Invalid reference label metadata in the uploaded sample.")
            return central_signal(loaded["signal"]), metadata
    except ValueError as exc:
        if "allow_pickle" in str(exc) or "pickled" in str(exc):
            raise ValueError("This file contains Python objects. Use a numeric complex sample exported by this page.") from exc
        raise
    except (OSError, EOFError, TypeError, KeyError, zipfile.BadZipFile) as exc:
        raise ValueError("Cannot read this sample. Use a numeric sample exported by this page.") from exc


def export_sample(values, record=None):
    metadata = sample_metadata()
    if record is not None:
        metadata["source_record"] = record
    buffer = BytesIO()
    np.savez_compressed(buffer, signal=central_signal(values), metadata_json=json.dumps(metadata))
    return buffer.getvalue()


def check_model(bundle):
    meta = bundle["metadata"]
    _validate_metadata(meta)
    if (meta.get("feature_dimension") != 42 or list(meta.get("feature_names", [])) != FEATURE_NAMES
            or set(map(str, bundle["classes"])) != set(CLASSES)):
        raise ValueError("The selected model has an incompatible feature or class schema.")
    if meta.get("is_synthetic") is not False:
        raise ValueError("This dashboard expects the saved measured-data models.")


def resolve_model_path(saved_path):
    path = Path(saved_path.replace("\\", "/"))
    # Older manifests store an absolute path to a model inside experiments.
    if path.parts[-4:-3] == ("experiments",):
        path = Path(*path.parts[-4:])
    return PROJECT_ROOT / path


def model_selection():
    """Switch atomically to a completed refinement; keep the baseline as fallback."""
    if (REFINED_ROOT / "metrics.json").exists():
        manifest = json.loads((REFINED_ROOT / "selected_models.json").read_text(encoding="utf-8"))
        for model in manifest["models"].values():
            model["path"] = str(resolve_model_path(model["path"]))
        return manifest
    return {"models": {name: {"path": str(EXPERIMENT_ROOT / "models" / filename)}
        for name, filename in (("Random Forest", "rf_kth77.joblib"), ("CNN", "cnn_kth77.pt"))},
        "fusion_rf_weight": .5, "version": "baseline"}


def combined_prediction(predictions, rf_weight=.5):
    """Fuse scores without using a reference label; support a single selected model."""
    if len(predictions) == 1:
        name, result = next(iter(predictions.items()))
        return {**result, "method": name}
    if set(predictions) != {"Random Forest", "CNN"} or not 0 <= rf_weight <= 1:
        raise ValueError("Choose RF/CNN models and a fusion weight between zero and one.")
    scores = {label: rf_weight * predictions["Random Forest"]["scores"][label] +
        (1-rf_weight) * predictions["CNN"]["scores"][label] for label in CLASSES}
    method = "CNN" if rf_weight == 0 else "Random Forest" if rf_weight == 1 else "Combined RF + CNN"
    return {"prediction": max(scores, key=scores.get), "scores": scores, "method": method}


def demo_positions(records, probabilities, view):
    """Presentation filter only. Labels select examples, never alter predictions."""
    labels = np.array([record["label"] for record in records])
    correct = np.ones(len(records), dtype=bool)
    for scores in probabilities.values():
        if np.asarray(scores).shape != (len(records), len(CLASSES)):
            raise ValueError("Sample bank and model scores are not aligned.")
        correct &= np.array(CLASSES)[np.argmax(scores, axis=1)] == labels
    if view == "Correct demo examples":
        return np.flatnonzero(correct)
    if view == "Model mistakes / disagreements":
        return np.flatnonzero(~correct)
    return np.arange(len(records))


def analyze_signal(values, bundles):
    values = central_signal(values)
    energy = np.mean(np.abs(values) ** 2)
    variation = np.mean(np.abs(values - values.mean()) ** 2)
    if not np.isfinite(energy) or variation <= np.finfo(float).eps * max(energy, np.finfo(float).tiny):
        raise ValueError("This sample has no varying complex signal to classify.")
    extracted = extract_segment_features(values)
    predictions = {}
    for name, bundle in bundles.items():
        check_model(bundle)
        scores = model_probabilities(bundle, extracted["features"][None], extracted["spectrogram"][None])[0]
        classes = np.asarray(bundle["classes"]).astype(str)
        if (scores.shape != (3,) or not np.isfinite(scores).all() or np.any(scores < 0)
                or not np.isclose(scores.sum(), 1, atol=1e-5)):
            raise ValueError("The model returned invalid class scores.")
        predictions[name] = {"prediction": str(classes[scores.argmax()]),
                             "scores": {str(c): float(s) for c, s in zip(classes, scores)}}
    return extracted, predictions


def load_demo_bank(root=EXPERIMENT_ROOT):
    root = Path(root)
    with np.load(root / "dashboard_samples.npz", allow_pickle=False) as bank:
        signals = bank["signals"].copy()
        metadata = json.loads(str(bank["metadata_json"].item()))
    _validate_metadata(metadata)
    records = metadata["records"]
    if (signals.shape != (len(records), 150) or signals.dtype.kind != "c"
            or not np.isfinite(signals).all() or not records
            or any(r.get("partition") != "test" or r.get("label") not in CLASSES for r in records)):
        raise ValueError("Invalid held-out sample bank; rebuild it from the verified archive.")
    return signals, records


def prepare_demo_bank(source, root=EXPERIMENT_ROOT):
    """Extract only the already evaluated test segments from the verified archive."""
    root = Path(root)
    destination = root / "dashboard_samples.npz"
    if destination.exists():
        raise ValueError("Sample bank already exists; keeping the existing file.")
    document = json.loads((root / "processed" / "test_segments.json").read_text(encoding="utf-8"))
    if document["feature_version"] != FEATURE_VERSION:
        raise ValueError("Test segments use a different feature schema.")
    measurements, fingerprints = load_dataset(source)
    by_identity = {m.identity: m for m in measurements}
    records, signals = document["records"], []
    for record in records:
        measurement = by_identity[record["measurement_id"]]
        index = record["segment_index"]
        if (record["partition"] != "test" or measurement.edge[index]
                or record["label"] != measurement.label):
            raise ValueError("Test sample metadata does not match the verified archive.")
        signals.append(target_signal(measurement, index))
    metadata = {**sample_metadata(), "fingerprints": fingerprints, "records": records}
    np.savez_compressed(destination, signals=np.stack(signals), metadata_json=json.dumps(metadata))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a small dashboard sample bank from the official archive")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    args = parser.parse_args()
    print(prepare_demo_bank(args.input, args.experiment_root))
