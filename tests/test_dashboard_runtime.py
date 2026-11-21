"""Regression checks for feature reuse and model output consistency."""
import json

import numpy as np
import pytest

from src.dashboard_data import EXPERIMENT_ROOT, model_selection
from src import dashboard_runtime as runtime

pytestmark = pytest.mark.skipif(not (EXPERIMENT_ROOT / "dashboard_samples.npz").is_file(),
                               reason="local recorded samples unavailable")


def test_recorded_fast_path_matches_upload_predictions_without_recomputing_emd(monkeypatch):
    from src import kth77_dataset
    signals, records = runtime.samples()
    paths = tuple((name, row["path"]) for name, row in model_selection()["models"].items())
    indices = [next(i for i, record in enumerate(records) if record["label"] == label)
               for label in ("bird", "drone", "human")]
    expected = {i: runtime.classify(signals[i], paths) for i in indices}

    def fail_on_emd(*args, **kwargs):
        raise AssertionError("Recorded samples must reuse the saved EMD features")

    monkeypatch.setattr(kth77_dataset, "extract_oh2019_emd_features", fail_on_emd)
    runtime._classify_recorded.clear()
    for i in indices:
        actual, predictions = runtime.classify_recorded(i, paths)
        reference, original_predictions = expected[i]
        for field in reference:
            np.testing.assert_array_equal(actual[field], reference[field])
        for name in predictions:
            assert predictions[name]["prediction"] == original_predictions[name]["prediction"]
            np.testing.assert_allclose(list(predictions[name]["scores"].values()),
                                       list(original_predictions[name]["scores"].values()), atol=1e-6)
    assert not signals.flags.writeable
    records[0]["label"] = "modified"
    assert runtime.samples()[1][0]["label"] != "modified"


def test_processed_metadata_order_is_checked_before_reusing_features(tmp_path):
    resources = list(runtime.processed_versions())
    document = json.loads((EXPERIMENT_ROOT / "processed/test_segments.json").read_text())
    document["records"][0], document["records"][1] = document["records"][1], document["records"][0]
    changed = tmp_path / "test_segments.json"
    changed.write_text(json.dumps(document))
    resources[2] = runtime.file_version(changed)
    with pytest.raises(ValueError, match="metadata do not match"):
        runtime._processed(tuple(resources))
