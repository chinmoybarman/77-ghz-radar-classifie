"""Upload boundaries and real-model consistency for the local sample dashboard."""
from io import BytesIO
import json

import numpy as np
import pytest

from src.dashboard_data import (EXPERIMENT_ROOT, analyze_signal, central_signal,
    check_model, export_sample, load_demo_bank, read_upload, sample_metadata,
    demo_positions, combined_prediction, model_selection, REFINED_ROOT)
from src.kth77_dataset import FEATURE_NAMES
from src.models_ml import load_model


def tone(size=150):
    return np.exp(2j * np.pi * 1000 * np.arange(size) / 17000)


def packed_sample(values, metadata=None):
    buffer = BytesIO()
    np.savez_compressed(buffer, signal=values,
        metadata_json=json.dumps(sample_metadata() if metadata is None else metadata))
    return buffer.getvalue()


def test_sample_export_upload_round_trip_preserves_identity_and_complex_samples():
    record = {"measurement_id": "m1", "segment_index": 47, "label": "drone"}
    value, metadata = read_upload(export_sample(tone(), record), "sample.npz")
    np.testing.assert_array_equal(value, tone())
    assert metadata["source_record"] == record
    assert not metadata.get("settings_assumed")


def test_plain_npy_crops_the_actual_center_cell():
    matrix = np.arange(1280, dtype=float).reshape(5, 256).astype(complex) + 1j
    for array in (matrix, matrix.ravel()):
        buffer = BytesIO()
        np.save(buffer, array)
        value, metadata = read_upload(buffer.getvalue(), "sample.npy")
        np.testing.assert_array_equal(value, matrix[2, 53:203])
        assert metadata["settings_assumed"]


@pytest.mark.parametrize("values", [np.ones(150), np.zeros((3, 64, 128), complex),
    np.full(150, complex(np.nan, 0)), np.array([object()], object)])
def test_incompatible_inputs_rejected(values):
    with pytest.raises(ValueError):
        central_signal(values)


@pytest.mark.parametrize("field,value", [("prf_hz", 2000), ("carrier_hz", 60e9),
    ("domain", "bgt60tr13c"), ("feature_version", "old"), ("source_record", [])])
def test_uploaded_settings_and_reference_schema_checked(field, value):
    metadata = {**sample_metadata(), field: value}
    with pytest.raises(ValueError):
        read_upload(packed_sample(tone(), metadata), "sample.npz")


def test_object_uploads_never_unpickled():
    buffer = BytesIO()
    np.save(buffer, np.array([{"not": "numeric"}], dtype=object))
    with pytest.raises(ValueError, match="contains Python objects"):
        read_upload(buffer.getvalue(), "source.npy")


def test_constant_or_zero_signal_cannot_produce_a_target_prediction():
    for values in (np.zeros(150, complex), np.full(150, 4 + 5j)):
        with pytest.raises(ValueError, match="no varying"):
            analyze_signal(values, {})


def test_bgt_and_mismatched_model_schemas_rejected():
    meta = {**sample_metadata(), "feature_dimension": 42,
        "feature_names": FEATURE_NAMES, "is_synthetic": False}
    bundle = {"classes": np.array(["bird", "drone", "human"]), "metadata": meta}
    check_model(bundle)
    for change in ({"domain": "bgt60tr13c"}, {"feature_dimension": 46}, {"is_synthetic": True}):
        with pytest.raises(ValueError):
            check_model({**bundle, "metadata": {**meta, **change}})


def test_demo_filter_preserves_all_indices_and_excludes_only_label_mismatches():
    records = [{"label": label} for label in ("bird", "drone", "human", "bird")]
    scores = {"rf": np.array([[1,0,0], [0,1,0], [0,0,1], [0,1,0]]),
              "cnn": np.array([[1,0,0], [0,0,1], [0,0,1], [1,0,0]])}
    np.testing.assert_array_equal(demo_positions(records, scores, "Correct demo examples"), [0,2])
    np.testing.assert_array_equal(demo_positions(records, scores, "Model mistakes / disagreements"), [1,3])
    np.testing.assert_array_equal(demo_positions(records, scores, "All test samples"), [0,1,2,3])


def test_combination_uses_only_model_scores_and_the_fixed_weight():
    predictions = {"Random Forest": {"prediction": "bird", "scores": {"bird": .8, "drone": .1, "human": .1}},
        "CNN": {"prediction": "drone", "scores": {"bird": .1, "drone": .8, "human": .1}}}
    combined = combined_prediction(predictions, .25)
    assert combined["prediction"] == "drone"
    assert combined["scores"]["drone"] == pytest.approx(.625)
    assert combined_prediction({"CNN": predictions["CNN"]})["method"] == "CNN"


@pytest.mark.skipif(not (REFINED_ROOT / "metrics.json").exists(), reason="refinement unavailable")
def test_refined_selection_and_inference_match_the_complete_evaluation():
    selection = model_selection()
    for name, row in selection["models"].items():
        candidates = [r["validation_macro_f1"] for r in selection["candidates"] if r["name"] == name]
        assert row["macro_f1"] == max(candidates)
    signals, records = load_demo_bank()
    bundles = {name: load_model(row["path"]) for name, row in selection["models"].items()}
    references = {name: np.load(REFINED_ROOT / f"{slug}_test_scores.npy", allow_pickle=False)
        for name, slug in (("Random Forest", "rf"), ("CNN", "cnn"), ("Combined", "combined"))}
    for label in ("bird", "drone", "human"):
        index = next(i for i, r in enumerate(records) if r["label"] == label)
        _, predictions = analyze_signal(signals[index], bundles)
        for name, result in predictions.items():
            np.testing.assert_allclose(list(result["scores"].values()), references[name][index], atol=1e-6)
        combined = combined_prediction(predictions, selection["fusion_rf_weight"])
        np.testing.assert_allclose(list(combined["scores"].values()), references["Combined"][index], atol=1e-6)


@pytest.mark.skipif(not (EXPERIMENT_ROOT / "dashboard_samples.npz").exists(), reason="local measured artifacts unavailable")
def test_actual_dashboard_samples_match_saved_features_images_and_predictions():
    import csv
    signals, records = load_demo_bank()
    assert len(records) == 1423
    assert len({r["measurement_id"] for r in records}) == 25
    indices = [next(i for i, row in enumerate(records) if row["label"] == label)
               for label in ("bird", "drone", "human")]
    bundles = {name: load_model(EXPERIMENT_ROOT / "models" / file)
        for name, file in (("rf", "rf_kth77.joblib"), ("cnn", "cnn_kth77.pt"))}
    references = {name: list(csv.DictReader((EXPERIMENT_ROOT / f"{name}_test_predictions.csv").open()))
                  for name in bundles}
    with np.load(EXPERIMENT_ROOT / "processed" / "test.npz", allow_pickle=False) as test:
        for index in indices:
            extracted, predictions = analyze_signal(signals[index], bundles)
            np.testing.assert_allclose(extracted["features"], test["features"][index], rtol=1e-10, atol=1e-10)
            np.testing.assert_array_equal(extracted["spectrogram"], test["spectrograms"][index])
            for name, result in predictions.items():
                assert result["prediction"] == references[name][index]["prediction"]
                assert max(result["scores"].values()) == pytest.approx(
                    float(references[name][index]["confidence_score"]), abs=1e-6)


@pytest.mark.skipif(not (EXPERIMENT_ROOT / "dashboard_samples.npz").exists(), reason="local measured artifacts unavailable")
def test_dashboard_switching_samples_models_and_upload_round_trip(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    app = AppTest.from_file(str(EXPERIMENT_ROOT.parents[1] / "src" / "radar_dashboard.py"))
    app.run(timeout=30)
    assert not app.exception
    assert app.metric[0].label == "Prediction"
    collection = next(s for s in app.selectbox if s.label == "Example collection")
    assert collection.value == "All test samples"
    collection.select("Correct demo examples").run(timeout=20)
    assert not app.exception
    assert app.metric[0].value == "Drone"
    assert any("Curated demo" in c.value for c in app.caption)
    next(s for s in app.selectbox if s.label == "Recorded target").select("bird").run(timeout=20)
    assert not app.exception
    assert "Recorded label: Bird" in " ".join(c.value for c in app.caption)
    next(s for s in app.selectbox if s.label == "Example collection").select("All test samples").run(timeout=20)
    assert not app.exception
    app.multiselect[0].set_value(["Random Forest"]).run(timeout=20)
    assert not app.exception
    assert len(app.metric) == 1
    # Exercise the actual upload branch with bytes exported from a measured segment.
    signals, records = load_demo_bank()
    uploaded = BytesIO(export_sample(signals[0], records[0]))
    uploaded.name = "recorded.npz"
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploaded)
    app.radio[0].set_value("Upload a sample").run(timeout=20)
    assert not app.exception
    assert len(app.metric) == 1
    assert "Reference label included in upload" in " ".join(c.value for c in app.caption)
    uploaded = BytesIO(packed_sample(np.ones(150)))
    uploaded.name = "invalid.npz"
    app.run(timeout=20)
    assert not app.exception
    assert len(app.error) == 1
    assert not app.metric  # A failed upload must not leave an earlier prediction visible.


@pytest.mark.skipif(not (EXPERIMENT_ROOT / "dashboard_samples.npz").exists(), reason="local measured artifacts unavailable")
def test_dashboard_navigation_boundaries_and_analysis_views():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(EXPERIMENT_ROOT.parents[1] / "src" / "radar_dashboard.py"))
    app.run(timeout=30)
    assert not app.exception
    previous = lambda: next(b for b in app.button if b.label == "Previous")
    following = lambda: next(b for b in app.button if b.label == "Next")
    slider = lambda: next(s for s in app.select_slider if s.label == "Sample within recording")
    assert previous().disabled
    following().click().run(timeout=20)
    assert slider().value == 1 and not previous().disabled
    previous().click().run(timeout=20)
    assert slider().value == 0 and previous().disabled
    slider().set_value(len(slider().options) - 1).run(timeout=20)
    assert following().disabled
    next(s for s in app.selectbox if s.label == "Recorded target").select("bird").run(timeout=20)
    assert slider().value == 0
    assert not app.exception
    for view in ("Model comparison", "Benchmark", "Signal"):
        next(r for r in app.radio if r.label == "Analysis view").set_value(view).run(timeout=20)
        assert not app.exception
        assert len(app.metric) == 1
        assert bool(app.dataframe) == (view != "Signal")
    next(s for s in app.selectbox if s.label == "Signal view").select("Waveform").run(timeout=20)
    assert not app.exception
    assert any("amplitude normalized" in c.value for c in app.caption)
