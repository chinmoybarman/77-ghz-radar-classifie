"""Cached local inference and read-only recording resources for the dashboard."""
from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import streamlit as st

from src.dashboard_data import EXPERIMENT_ROOT, analyze_signal, check_model, load_demo_bank
from src.kth77_dataset import CLASSES, FEATURE_NAMES, FEATURE_VERSION, PRF_HZ
from src.models_ml import load_model, model_probabilities
from src.radar_dsp import normalize_complex_burst, standardized_spectrogram

SPECTRAL_DIR = EXPERIMENT_ROOT.parent / "kth77_spectral"


def file_version(path):
    path = Path(path).resolve()
    info = path.stat()
    return str(path), info.st_mtime_ns, info.st_size


def model_versions(model_paths):
    return tuple((name, file_version(path)) for name, path in model_paths)


@st.cache_resource(show_spinner=False, max_entries=8)
def _model(version):
    bundle = load_model(version[0])
    check_model(bundle)
    if bundle["kind"] == "cnn":
        import torch
        torch.set_num_threads(2)
        bundle["model"].eval().requires_grad_(False)
    elif bundle["kind"] == "rf":
        bundle["model"].set_params(classifier__n_jobs=1)
    return bundle


def dashboard_model(path):
    return _model(file_version(path))


@st.cache_resource(show_spinner=False, max_entries=2)
def _samples(version):
    signals, records = load_demo_bank(Path(version[0]).parent)
    signals.setflags(write=False)
    return signals, tuple(MappingProxyType(row) for row in records)


def samples():
    signals, records = _samples(file_version(EXPERIMENT_ROOT / "dashboard_samples.npz"))
    # Metadata is cheap to copy and callers can safely add export fields.
    return signals, [dict(row) for row in records]


def processed_versions():
    return tuple(file_version(EXPERIMENT_ROOT / path) for path in (
        "dashboard_samples.npz", "processed/test.npz", "processed/test_segments.json"))


@st.cache_resource(show_spinner=False, max_entries=2)
def _processed(versions):
    signals, records = _samples(versions[0])
    document = json.loads(Path(versions[2][0]).read_text(encoding="utf-8"))
    if document.get("feature_version") != FEATURE_VERSION or document["records"] != list(records):
        raise ValueError("Recorded features and sample metadata do not match. Rebuild the sample bank.")
    with np.load(versions[1][0], allow_pickle=False) as saved:
        features, images = saved["features"], saved["spectrograms"]
        if (features.shape != (len(records), len(FEATURE_NAMES)) or images.shape != (len(records), 32, 32)
                or not np.array_equal(saved["feature_names"], FEATURE_NAMES)
                or not np.array_equal(saved["labels"], [row["label"] for row in records])
                or not np.isfinite(features).all() or not np.isfinite(images).all()):
            raise ValueError("Recorded features have an incompatible shape, labels or values.")
    features.setflags(write=False)
    images.setflags(write=False)
    return signals, features, images


@st.cache_data(show_spinner=False, max_entries=128)
def _classify_upload(values, versions):
    return analyze_signal(values, {name: _model(version) for name, version in versions})


def classify(values, model_paths):
    return _classify_upload(values, model_versions(model_paths))


def _prediction(scores, classes):
    if (scores.shape != (len(CLASSES),) or not np.isfinite(scores).all()
            or np.any(scores < 0) or not np.isclose(scores.sum(), 1, atol=1e-5)):
        raise ValueError("The model returned invalid class scores.")
    return {"prediction": str(classes[scores.argmax()]),
            "scores": {str(label): float(score) for label, score in zip(classes, scores)}}


@st.cache_data(show_spinner=False, max_entries=256)
def _classify_recorded(position, versions, resources):
    signals, features, images = _processed(resources)
    if not isinstance(position, (int, np.integer)) or not 0 <= position < len(signals):
        raise ValueError("Choose a valid recorded sample.")
    values = signals[position]
    normalized = normalize_complex_burst(values - values.mean())
    _, frequency, times = standardized_spectrogram(normalized, PRF_HZ, nperseg=64, return_axes=True)
    extracted = {"signal": normalized, "features": features[position],
                 "spectrogram": images[position], "frequency_hz": frequency, "time_s": times}
    predictions = {}
    # Small writable copies keep the shared feature bank immutable through Torch conversion.
    feature_row = features[position:position+1].copy()
    image_row = images[position:position+1].copy()
    for name, version in versions:
        bundle = _model(version)
        scores = model_probabilities(bundle, feature_row, image_row)[0]
        predictions[name] = _prediction(scores, np.asarray(bundle["classes"]))
    return extracted, predictions


def classify_recorded(position, model_paths):
    return _classify_recorded(position, model_versions(model_paths), processed_versions())


@st.cache_data(show_spinner=False, max_entries=8)
def _bank_scores(versions, resources):
    _, features, images = _processed(resources)
    result = {}
    for name, version in versions:
        bundle = _model(version)
        order = [list(bundle["classes"]).index(label) for label in CLASSES]
        scores = model_probabilities(bundle, features, images.copy())[:, order]
        if (scores.shape != (len(features), len(CLASSES)) or not np.isfinite(scores).all()
                or np.any(scores < 0) or not np.allclose(scores.sum(axis=1), 1, atol=1e-5)):
            raise ValueError("The model returned invalid recorded class scores.")
        result[name] = scores
    return result


def bank_scores(model_paths):
    return _bank_scores(model_versions(model_paths), processed_versions())


@st.cache_resource(show_spinner=False, max_entries=2)
def _spectral_model(version):
    import torch
    from src.kth77_spectral import load_spectral
    torch.set_num_threads(2)
    return load_spectral(version[0]).eval().requires_grad_(False)


def spectral_model():
    return _spectral_model(file_version(SPECTRAL_DIR / "spectral.pt"))


@st.cache_resource(show_spinner=False, max_entries=2)
def _replay(version, sample_version):
    from src.kth77_spectral import replay_bank
    anchors, records = _samples(sample_version)
    arrays = replay_bank(version[0], records, anchors)
    for array in arrays:
        array.setflags(write=False)
    return arrays


def replay():
    return _replay(file_version(SPECTRAL_DIR / "replay.npz"),
                   file_version(EXPERIMENT_ROOT / "dashboard_samples.npz"))


@st.cache_data(show_spinner=False, max_entries=128)
def _spectral_prediction(echoes, version):
    from src.kth77_spectral import spectral_scores
    scores = spectral_scores(_spectral_model(version), echoes).mean(axis=0, dtype=np.float64)
    _prediction(scores, np.asarray(CLASSES))
    return scores


def spectral_prediction(echoes):
    return _spectral_prediction(echoes, file_version(SPECTRAL_DIR / "spectral.pt"))


@st.cache_data(show_spinner=False, max_entries=8)
def _spectral_bank_scores(window, version, sample_version, replay_version):
    from src.kth77_spectral import mean_scores, spectral_scores
    model = _spectral_model(version)
    if window == 1:
        return spectral_scores(model, _samples(sample_version)[0])
    signals, counts, _, _ = _replay(replay_version, sample_version)
    return mean_scores(spectral_scores(model, signals), counts, window)


def spectral_bank_scores(window):
    return _spectral_bank_scores(window, file_version(SPECTRAL_DIR / "spectral.pt"),
        file_version(EXPERIMENT_ROOT / "dashboard_samples.npz"),
        file_version(SPECTRAL_DIR / "replay.npz") if window > 1 else None)


def collection_scores(model_paths, choice):
    scores = bank_scores(model_paths)
    if choice and choice["spectral_weight"] > 0:
        scores["Spectral CNN"] = spectral_bank_scores(choice["window"])
    return scores
