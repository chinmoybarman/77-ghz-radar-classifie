"""Check the KTH spectral input, causal window boundaries, and dashboard replay."""
import csv
import json
from io import BytesIO

import numpy as np
import pytest

pytest.importorskip("torch")
from src.dashboard_data import export_sample, load_demo_bank, read_upload_with_history
from src.kth77_dataset import CLASSES
from src.kth77_spectral import (MODEL_DIR, blend_scores, history_indices, load_spectral,
                                mean_scores, predict_file, replay_bank, spectral_scores, spectral_views)


def test_relative_power_ignores_gain_and_global_phase():
    rng = np.random.default_rng(0)
    signals = rng.normal(size=(3, 150)) + 1j*rng.normal(size=(3, 150))
    features = spectral_views(signals)
    assert features.shape == (3, 5, 150)
    np.testing.assert_allclose(features, spectral_views(signals * (2-4j)), atol=1e-6)
    with pytest.raises(ValueError):
        spectral_views(signals.real)


def test_history_stops_at_edge_gap_and_current_scan():
    times = np.array([0, .1, .2, .3, .6, .7, .8])
    edge = np.array([0, 1, 0, 0, 0, 0, 0], dtype=bool)
    for end, expected in [(0, [0]), (3, [2, 3]), (4, [4]), (5, [4, 5])]:
        np.testing.assert_array_equal(history_indices(end, times, edge, 9), expected)
    np.testing.assert_array_equal(history_indices(6, times, edge, 2), [5, 6])


def test_window_average_never_borrows_from_another_recording():
    scores = np.eye(3)[[0, 0, 0, 1, 2]]
    lengths = np.array([3, 2])
    np.testing.assert_allclose(mean_scores(scores, lengths, 1), [[1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(mean_scores(scores, lengths, 9), [[1, 0, 0], [0, .5, .5]])


def test_uploaded_sequence_preserves_times_and_checks_anchor_and_order():
    rng = np.random.default_rng(52)
    x = rng.normal(size=(3, 150)) + 1j*rng.normal(size=(3, 150))
    times = np.array([11., 11.1, 11.2])
    content = export_sample(x[-1], history=(x, times))
    anchor, _, (echoes, loaded_times) = read_upload_with_history(content, "sequence.npz")
    np.testing.assert_array_equal(anchor, x[-1])
    np.testing.assert_array_equal(echoes, x)
    np.testing.assert_array_equal(loaded_times, times)
    for bad_anchor, bad_times in [(x[0], times), (x[-1], times[::-1])]:
        with pytest.raises(ValueError):
            export_sample(bad_anchor, history=(x, bad_times))


@pytest.mark.skipif(not (MODEL_DIR / "metrics.json").exists(), reason="trained spectral run unavailable")
def test_cpu_replay_matches_both_evaluations():
    from src.dashboard_data import FUSION_ROOT
    anchors, records = load_demo_bank()
    echoes, counts, _, _ = replay_bank(MODEL_DIR / "replay.npz", records, anchors)
    choices = json.loads((MODEL_DIR / "selection.json").read_text())
    baseline = np.load(FUSION_ROOT / "combined_test_scores.npy", allow_pickle=False)
    model = load_spectral(MODEL_DIR / "spectral.pt")
    for name, choice in choices.items():
        if name == "single":
            spectral = spectral_scores(model, anchors)
        else:
            spectral = mean_scores(spectral_scores(model, echoes), counts, choice["window"])
        mixed = blend_scores(baseline, spectral, choice["spectral_weight"])
        with (MODEL_DIR / f"{name}_test.csv").open(newline="") as handle:
            refs = list(csv.DictReader(handle))
        np.testing.assert_array_equal(np.array(CLASSES)[mixed.argmax(1)], [r["prediction"] for r in refs])
        np.testing.assert_allclose(mixed.max(1), [float(r["confidence_score"]) for r in refs], atol=2e-5)


@pytest.mark.skipif(not (MODEL_DIR / "metrics.json").exists(), reason="trained spectral run unavailable")
def test_uploaded_sequence_uses_the_same_decision_as_recorded_replay(monkeypatch, tmp_path):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    choices = json.loads((MODEL_DIR / "selection.json").read_text())
    if choices["sequence"]["window"] == 1:
        pytest.skip("This run selected single-scan inference.")
    anchors, records = load_demo_bank()
    echoes, counts, ends, times = replay_bank(MODEL_DIR / "replay.npz", records, anchors)
    i = int(np.argmax(counts))
    section = slice(ends[i]-counts[i], ends[i])
    content = export_sample(anchors[i], history=(echoes[section], times[section]))
    path = tmp_path / "sequence.npz"
    path.write_bytes(content)
    direct = predict_file(path, recent=True)
    assert direct["scans_used"] == counts[i]
    buf = BytesIO(content)
    buf.name = "sequence.npz"
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: buf)
    app = AppTest.from_file(str(MODEL_DIR.parents[1] / "src" / "radar_dashboard.py")).run(timeout=30)
    next(r for r in app.radio if r.label == "Sample source").set_value("Upload a sample").run(timeout=30)
    next(r for r in app.radio if r.label == "Classification mode").set_value("Recent scans").run(timeout=30)
    assert not app.exception and not app.error
    with (MODEL_DIR / "sequence_test.csv").open(newline="") as f:
        assert app.metric[0].value.lower() == direct["prediction"] == list(csv.DictReader(f))[i]["prediction"]
    assert any(f"Used {counts[i]} of up to " in c.value for c in app.caption)
