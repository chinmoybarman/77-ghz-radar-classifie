"""Numerical oracles, malformed-input checks, provenance and offline integration.

All observations made here are synthetic fixtures, never hardware validation.
"""
from dataclasses import replace
import csv
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import C0, DEFAULT_CONFIG, RadarConfig
from src import radar_dsp as dsp
from src.dataset_manager import (
    CLASSES, generate_synthetic_pretrain_dataset, read_registry, split_sessions,
    peter_rotor_return, passafiume_displacement,
)
from src.inspect_raw_recording import load_recording, validate_raw_data, _HEADER_FIELDS
from src.evaluate_pipeline import (
    ScanTrackFilter, center_burst_indices, assert_no_session_leakage,
    config_signature, evaluate_predictions,
)
from src.models_ml import (
    TERRMClassifier, make_random_forest, save_random_forest, load_model,
    model_probabilities, reduced_multivariate_features, train_cnn, cnn_probabilities,
)
from src.predict_offline import _check_compatibility, predict_recording


@pytest.fixture
def cfg():
    return RadarConfig.from_file(DEFAULT_CONFIG)


def tone(cfg, range_bin=17, doppler_bin=8):
    n = np.arange(cfg.num_samples_per_chirp)[None, :]
    l = np.arange(cfg.num_chirps_per_frame)[:, None]
    phase = 2*np.pi*(range_bin*n/cfg.num_samples_per_chirp + doppler_bin*l/cfg.num_chirps_per_frame)
    return np.stack([.3*np.cos(phase + offset) for offset in (.2, -.4, 0)])


def save_fixture(tmp_path, cfg, data=None, **metadata):
    path = tmp_path / "recording.npy"
    np.save(path, tone(cfg)[None] if data is None else data)
    path.with_suffix(".json").write_text(json.dumps({"config": cfg.to_dict(), "is_synthetic": True, **metadata}))
    return path


def test_config_physical_scales(cfg):
    assert cfg.bandwidth_Hz == 2e9
    assert cfg.slope_Hz_s == pytest.approx(1.5625e13)
    assert cfg.range_resolution_m == pytest.approx(C0/4e9)
    assert cfg.prf_Hz == 2000
    assert cfg.wavelength_m == pytest.approx(C0/60e9)


@pytest.mark.parametrize("update", [
    {"rx_mask": 3}, {"data_format": 1}, {"sample_rate_Hz": float("nan")},
    {"chirp_repetition_time_s": 1e-6}, {"frame_repetition_time_s": .001},
    {"end_frequency_Hz": 70e9}, {"num_samples_per_chirp": 128.5},
])
def test_invalid_config(cfg, update):
    with pytest.raises(ValueError):
        RadarConfig.from_dict({**cfg.to_dict(), **update})


def test_sdk_effective_config_wins_over_requested(cfg):
    effective = replace(cfg, chirp_repetition_time_s=.0006)
    assert RadarConfig.from_dict({"config": effective.to_dict(), "requested_config": cfg.to_dict()}) == effective


def test_nested_json_and_xml_config(tmp_path, cfg):
    nested = {"device": {"fmcw": cfg.to_dict()}}
    assert RadarConfig.from_dict(nested) == cfg
    xml = tmp_path / "config.xml"
    fields = "".join(f'<parameter name="{k}" value="{v}"/>' for k, v in cfg.to_dict().items() if v is not None)
    xml.write_text(f"<radar>{fields}</radar>")
    assert RadarConfig.from_file(xml) == cfg
    with pytest.raises(ValueError, match="Conflicting aliases"):
        RadarConfig.from_dict({**cfg.to_dict(), "num_samples": 256})


@pytest.mark.parametrize("layout", ["canonical", "single", "rx_flat", "frame_flat", "one_dimensional"])
def test_npy_layouts(tmp_path, cfg, layout):
    original = np.stack([tone(cfg), tone(cfg, doppler_bin=-5)])
    forms = {"canonical": original, "single": original[0],
             "rx_flat": original.reshape(6, 64, 128),
             "frame_flat": original.reshape(2, 192, 128),
             "one_dimensional": original.ravel()}
    result = load_recording(save_fixture(tmp_path, cfg, forms[layout]))
    np.testing.assert_allclose(result.frames, original[:1] if layout == "single" else original)
    assert result.metadata["is_synthetic"] is True


def test_metadata_is_required(tmp_path, cfg):
    path = tmp_path / "bare.npy"
    np.save(path, tone(cfg))
    with pytest.raises(ValueError, match="Missing acquisition metadata"):
        load_recording(path)


def test_external_object_dataset_gets_actionable_error(tmp_path, cfg):
    path = tmp_path / "external.npy"
    np.save(path, np.array([["D1", np.zeros(1280, dtype=complex)]], dtype=object))
    with pytest.raises(ValueError, match="dedicated adapter"):
        load_recording(path, cfg=cfg)


def test_explicit_config_cannot_hide_conflict(tmp_path, cfg):
    path = save_fixture(tmp_path, cfg)
    wrong = tmp_path / "wrong.json"
    replace(cfg, chirp_repetition_time_s=.0006).save(wrong)
    with pytest.raises(ValueError, match="disagree"):
        load_recording(path, config_path=wrong)
    with pytest.raises(FileNotFoundError):
        load_recording(path, config_path=tmp_path / "missing.json")


def test_explicit_metadata_cannot_relabel_synthetic(tmp_path, cfg):
    path = save_fixture(tmp_path, cfg)
    extra = tmp_path / "override.json"
    extra.write_text(json.dumps({"config": cfg.to_dict(), "is_synthetic": False}))
    assert load_recording(path, config_path=extra).metadata["is_synthetic"] is True


def test_changed_ramp_is_rejected(tmp_path, cfg):
    path = save_fixture(tmp_path, cfg)
    wrong = tmp_path / "ramp.json"
    replace(cfg, ramp_duration_s=.0002).save(wrong)
    with pytest.raises(ValueError, match="slope"):
        load_recording(path, config_path=wrong)
    metadata = {"feature_dimension": 46, "feature_names": dsp.FEATURE_NAMES,
                "training_configs": [config_signature(cfg)]}
    with pytest.raises(ValueError, match="acquisition settings"):
        _check_compatibility(metadata, replace(cfg, ramp_duration_s=.0002), dsp.FEATURE_NAMES)


@pytest.mark.parametrize("kind", ["columns", "structured", "text"])
def test_rule8_tracking_logs_rejected(tmp_path, cfg, kind):
    if kind == "text":
        path = tmp_path / "tracks.txt"
        path.write_text("range_m,velocity_mps,target_id,azimuth_deg\n1,2,3,4\n")
    else:
        data = np.ones((128, 4)) if kind == "columns" else np.zeros(10, dtype=[("range_m", "f8")])
        path = save_fixture(tmp_path, cfg, data)
    with pytest.raises(ValueError, match="Rule 8"):
        load_recording(path, cfg=cfg)


@pytest.mark.parametrize("fault", ["nan", "inf", "dead", "clipped", "complex"])
def test_invalid_adc_rejected(tmp_path, cfg, fault):
    data = tone(cfg)[None].copy()
    if fault in ("nan", "inf"):
        data[0, 0, 0, 0] = float(fault)
    elif fault == "dead":
        data[:, 1] = .2
    elif fault == "clipped":
        data[:, 1, :2] = 1
    else:
        data = data.astype(complex)
    with pytest.raises(ValueError):
        load_recording(save_fixture(tmp_path, cfg, data))


@pytest.mark.parametrize("token", ["nan", "Inf", "-Infinity"])
def test_text_nonfinite_rows_not_silently_skipped(tmp_path, cfg, token):
    path = tmp_path / "raw.txt"
    np.savetxt(path, tone(cfg).ravel())
    with path.open("a") as handle:
        handle.write(token + "\n")
    with pytest.raises(ValueError, match="NaN or Inf"):
        load_recording(path, cfg=cfg)


def test_counts_and_raw_text(tmp_path, cfg):
    data = np.rint((tone(cfg)[None] + .5) * 4095).astype(np.uint16)
    normalized, quality = validate_raw_data(data, replace(cfg, adc_format="counts"))
    np.testing.assert_allclose(normalized, data / 4095)
    assert quality["adc_format"] == "counts"
    path = tmp_path / "raw.txt"
    np.savetxt(path, tone(cfg).ravel())
    np.testing.assert_allclose(load_recording(path, cfg=cfg).frames[0], tone(cfg))


def test_clipping_in_one_frame_cannot_be_diluted(cfg):
    data = np.stack([tone(cfg)]*20)
    data[0, 0, :2] = 1.0  # 3.125% in one RX/frame; only 0.156% across the file.
    with pytest.raises(ValueError, match="at least one frame/RX"):
        validate_raw_data(data, cfg)


@pytest.mark.parametrize("fault", ["complex", "shape", "fractional_counts"])
def test_sdk_adc_validation_is_strict(cfg, fault):
    data = tone(cfg)[None]
    profile = cfg
    if fault == "complex":
        data = data.astype(complex) + .1j
    elif fault == "shape":
        data = data[0]
    else:
        data = (data + .5)*4095
        profile = replace(cfg, adc_format="counts")
    with pytest.raises(ValueError):
        validate_raw_data(data, profile)


@pytest.mark.parametrize("times", [[0], [0, 0], [1, 0], [0, float("nan")], [0, float("inf")], ["0", "1"]])
def test_invalid_frame_timestamps_rejected(tmp_path, cfg, times):
    path = save_fixture(tmp_path, cfg, np.stack([tone(cfg)]*2), frame_timestamps_s=times)
    with pytest.raises(ValueError, match="frame_timestamps_s"):
        load_recording(path)


def test_nominal_timestamps_are_labeled(tmp_path, cfg):
    recording = load_recording(save_fixture(tmp_path, cfg, np.stack([tone(cfg)]*2)))
    np.testing.assert_allclose(recording.metadata["frame_timestamps_s"], [0, cfg.frame_repetition_time_s])
    assert "nominal" in recording.metadata["timestamps_source"]


def write_ifrb(path, cfg, *, endian="<", size=92, interleaved=True, counters=(1, 2)):
    values = [1, 3, 1, 7, int(interleaved), 0, 0, cfg.start_frequency_Hz/1000,
              cfg.end_frequency_Hz/1000, cfg.sample_rate_Hz/1000, 12, 1, 0, 64, 128,
              3*64*128, cfg.chirp_duration_s, cfg.chirp_repetition_time_s, cfg.frame_repetition_time_s]
    header = b"IFRB" + struct.pack(endian + "fi", 1.0, size)
    header += b"".join(struct.pack(endian + dtype, value) for (_, dtype), value in zip(_HEADER_FIELDS, values))
    header += bytes(size - len(header))
    original = tone(cfg) + .5
    layout = original.transpose(1, 2, 0) if interleaved else original.transpose(1, 0, 2)
    payload = layout.astype(endian + "f4").tobytes()
    path.write_bytes(header + b"".join(struct.pack(endian + "I", c) + payload for c in counters))
    return np.stack([original]*len(counters))


@pytest.mark.parametrize("endian,size,interleaved", [("<", 88, False), ("<", 92, True), (">", 92, True)])
def test_legacy_binary_layouts(tmp_path, cfg, endian, size, interleaved):
    path = tmp_path / "capture.raw.bin"
    expected = write_ifrb(path, cfg, endian=endian, size=size, interleaved=interleaved)
    recording = load_recording(path)
    np.testing.assert_allclose(recording.frames, expected, atol=4e-8)
    # Valid binary headers must also survive downstream uniform-grid checks.
    result = dsp.extract_fast_time_interpolated_signal(recording.frames[0], recording.config)
    assert result["idle_samples"] == 372


def test_legacy_truncation_and_missing_frames(tmp_path, cfg):
    path = tmp_path / "capture.raw.bin"
    write_ifrb(path, cfg, counters=(1, 3))
    with pytest.raises(ValueError, match="missing/reordered"):
        load_recording(path)
    path.write_bytes(path.read_bytes()[:-3])
    with pytest.raises(ValueError, match="Truncated"):
        load_recording(path)


@pytest.mark.parametrize("doppler_bin", [-9, 8, 40])
def test_real_fft_retains_signed_and_aliased_doppler(cfg, doppler_bin):
    result = dsp.compute_range_doppler_and_target_bin(tone(cfg, doppler_bin=doppler_bin), cfg)
    assert result["target_bin"] == 17
    assert result["range_fft"].shape == (3, 64, 65)
    assert np.iscomplexobj(result["range_fft"])
    peak = result["doppler_axis_hz"][result["rv_power"][:, 17].argmax()]
    expected_bin = (doppler_bin + 32) % 64 - 32
    assert peak == pytest.approx(expected_bin*cfg.prf_Hz/64)
    assert result["target_range_m"] == pytest.approx(17*cfg.range_resolution_m)
    np.testing.assert_allclose(np.abs(result["target_signals"]).mean(axis=1), 1)


def test_mti_cancels_static_clutter(cfg):
    clutter = tone(cfg, range_bin=6, doppler_bin=0)*5
    mixed = clutter + tone(cfg)
    result = dsp.compute_range_doppler_and_target_bin(mixed, cfg)
    assert result["target_bin"] == 17
    assert np.max(np.abs(result["range_fft_mti"][:, :, 6])) < 1e-10
    stationary = dsp.compute_range_doppler_and_target_bin(clutter, cfg)
    assert stationary["pruning"]["empty_signal"]


def test_aoa_pair_stability_and_range_gate(cfg):
    result = dsp.compute_range_doppler_and_target_bin(tone(cfg), cfg)
    assert result["antenna_pairs"] == {"azimuth": [1, 3], "elevation": [2, 3]}
    assert result["aoa_azimuth_phase_std_rad"] < 1e-6
    gated = replace(cfg, range_min_m=1, range_max_m=2)
    with pytest.raises(ValueError, match="outside"):
        dsp.compute_range_doppler_and_target_bin(tone(cfg), gated, target_bin=2)


def test_interpolation_preserves_samples_and_timestamps(cfg):
    original = tone(cfg)
    joined = dsp.interpolate_chirp_idle_time(original, cfg)
    assert joined["signal"].shape == (3, 64*128 + 63*372)
    assert joined["observed_sample_mask"].sum() == 64*128
    for l in (0, 1, 63):
        np.testing.assert_array_equal(joined["signal"][:, l*500:l*500+128], original[:, l])
    first = original[:, 0, -1] + (original[:, 1, 0] - original[:, 0, -1])/373
    np.testing.assert_allclose(joined["signal"][:, 128], first)
    assert joined["time_s"][500] == cfg.chirp_repetition_time_s
    noninteger = replace(cfg, chirp_repetition_time_s=.0005005)
    with pytest.raises(ValueError):
        dsp.interpolate_chirp_idle_time(original, noninteger)


def test_ftmdse_conjugate_filter_and_allpass_doppler(cfg):
    raw = tone(cfg)
    result = dsp.extract_fast_time_interpolated_signal(raw, cfg, 17, sigma_range_hz=100)
    np.testing.assert_allclose(result["filtered_chirps"], raw, atol=1e-13)
    assert result["range_filter"][17] == result["range_filter"][-17] == 1
    assert result["duty_cycle"] == pytest.approx(.256)
    assert result["unaliased_recovery_guaranteed"] is False


def test_emd_reconstruction_and_monotone_residual():
    t = np.linspace(0, 1, 192)
    values = np.sin(2*np.pi*20*t) + .4*np.sin(2*np.pi*4*t) + t
    modes, residual = dsp.emd_decompose(values, return_residual=True)
    np.testing.assert_allclose(modes.sum(axis=0) + residual, values, atol=1e-14)
    modes, residual = dsp.emd_decompose(t, return_residual=True)
    np.testing.assert_array_equal(modes, 0)
    np.testing.assert_array_equal(residual, t)


def test_emd_feature_numerical_oracles():
    modes = np.array([[1, -1, 1, -1], [2, -2, 2, -2], [0, 0, 0, 0]], dtype=float)
    result = dsp.features_from_imfs(modes)
    assert result.shape == (34,)
    np.testing.assert_array_equal(result[:3], [3, 3, 0])
    np.testing.assert_allclose(result[3:6], [.2, .8, 0])
    np.testing.assert_allclose(result[15:18], [16, 64, 0])  # Parseval: d * sum(m**2)
    np.testing.assert_allclose(result[24:26], [1, 0])
    np.testing.assert_allclose(result[26:28], [4, 0])
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(dsp.extract_oh2019_emd_features(np.zeros(64)), np.zeros(34))


def test_complete_feature_and_image_contract(cfg):
    result = dsp.extract_burst_features(tone(cfg), cfg)
    assert result["features"].shape == (46,)
    assert len(set(result["feature_names"])) == 46
    assert np.isfinite(result["features"]).all()
    np.testing.assert_allclose(result["spectrogram_time_s"][[0,-1]], [.008,.024])
    np.testing.assert_allclose(result["spectrogram_frequency_hz"][[0,-1]], [-1000,968.75])
    for key in ("spectrogram", "ftmdse_spectrogram"):
        assert result[key].shape == (32, 32)
        assert np.isfinite(result[key]).all()
        assert result[key].min() >= 0 and result[key].max() <= 1


def test_rotor_model_zero_projection_matches_analytic_body(cfg):
    fast = np.arange(128)[None]/cfg.sample_rate_Hz
    absolute = np.arange(64)[:, None]*cfg.chirp_repetition_time_s + fast
    result = peter_rotor_return(fast, absolute, cfg, range_m=1, velocity_mps=.1,
        blade_length_m=.06, rpm=8000, elevation_rad=np.pi/2, blade_amplitude=2)
    phase = 4*np.pi*(cfg.start_frequency_Hz + cfg.slope_Hz_s*fast)/C0*(1+.1*absolute)
    np.testing.assert_allclose(result, (1 + 2*.06*2)*np.exp(1j*phase), atol=1e-12)
    displacement = passafiume_displacement(absolute, np.zeros((1,4)), np.ones((1,4)), np.zeros((1,4)))
    np.testing.assert_array_equal(displacement, 0)


def test_session_split_and_center_selection():
    rows = [{"session_id": f"{label}_{session}", "label": label}
            for label in CLASSES for session in range(5) for _ in range(2)]
    parts = split_sessions(rows)
    assert_no_session_leakage(parts)
    assert parts == split_sessions(rows)
    assert sum(map(len, parts.values())) == len(rows)
    assert all({r["label"] for r in part} == set(CLASSES) for part in parts.values())
    with pytest.raises(ValueError, match="leakage"):
        assert_no_session_leakage({"train": rows, "test": rows[:1]})
    np.testing.assert_array_equal(center_burst_indices(10), np.arange(1, 8))
    np.testing.assert_array_equal(center_burst_indices(10, False), [5])


def test_track_filter_warmup_majority_and_score_update():
    tracker = ScanTrackFilter(["drone", "bird"])
    assert [tracker.update(x, .5) for x in ["drone", "bird", "drone"]] == [None]*3
    assert tracker.update("drone", .5) == "drone"
    assert tracker.update("bird", 1) == "drone"  # tied total score follows class order
    assert tracker.update("bird", 1) == "bird"
    tracker.reset()
    assert tracker.update("bird", .9) is None


def test_track_gap_restarts_warmup():
    tracker = ScanTrackFilter(["drone", "bird"])
    labels = [tracker.update("drone", .8, time_s=t, max_gap_s=.15) for t in [0, .1, .2, .3, 1.0]]
    assert labels == [None, None, None, "drone", None]
    assert tracker.history == ["drone"]
    with pytest.raises(ValueError, match="strictly increasing"):
        tracker.update("drone", .8, time_s=1.0, max_gap_s=.15)


def test_track_metrics_reset_between_recordings(tmp_path):
    records = [{"session_id": "s", "recording_id": rid, "frame_index": i,
                "label": "drone" if rid == "a" else "bird", "is_synthetic": True, "center_burst": i == 2, "pruning": {}}
               for rid in ("a", "b") for i in range(4)]
    data = {"y": np.array([r["label"] for r in records]), "records": records}
    result = evaluate_predictions(data, np.array([[1., 0.]]*4 + [[0., 1.]]*4), ["drone", "bird"], tmp_path/"predictions.csv")
    assert result["track_warmup_abstentions"] == 6
    assert result["track_filtered"]["n"] == 2
    assert result["single_scan"]["balanced_accuracy"] == 1


def test_evaluation_resets_on_recorded_time_gap(tmp_path):
    records = [{"session_id": "s", "recording_id": "r", "frame_index": i,
                "label": "drone" if i < 4 else "bird", "is_synthetic": True,
                "center_burst": False, "pruning": {}, "track_max_gap_s": .15,
                "time_s": i*.1 if i < 4 else 1 + (i-4)*.1} for i in range(8)]
    data = {"y": np.array([r["label"] for r in records]), "records": records}
    result = evaluate_predictions(data, np.array([[1.,0.]]*4 + [[0.,1.]]*4),
                                  ["drone", "bird"], tmp_path/"timed.csv")
    assert result["track_warmup_abstentions"] == 6
    assert result["track_filtered"]["accuracy"] == 1


@pytest.fixture(scope="module")
def synthetic_registry(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    return generate_synthetic_pretrain_dataset(root, sessions_per_class=3, frames_per_recording=4, seed=17)


def test_synthetic_registry_provenance(synthetic_registry):
    with pytest.raises(ValueError, match="allow-synthetic"):
        read_registry(synthetic_registry)
    rows = read_registry(synthetic_registry, allow_synthetic=True)
    assert len(rows) == 12
    assert all(row["is_synthetic"] for row in rows)
    assert_no_session_leakage(split_sessions(rows))
    recording = load_recording(rows[0]["filepath"])
    assert recording.frames.shape == (4, 3, 64, 128)


def test_registry_sidecar_disagreement(tmp_path, synthetic_registry):
    rows = read_registry(synthetic_registry, allow_synthetic=True)
    rows[0]["is_synthetic"] = False
    registry = tmp_path / "registry.csv"
    with registry.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="provenance"):
        read_registry(registry, allow_synthetic=True)


@pytest.mark.parametrize("kind", ["rf", "terrm"])
def test_classical_models_roundtrip_and_separation(tmp_path, kind):
    rng = np.random.default_rng(12)
    x = np.r_[rng.normal(-2, .1, (12, 3)), rng.normal(2, .1, (12, 3))]
    y = np.array(["bird"]*12 + ["drone"]*12)
    model = make_random_forest() if kind == "rf" else TERRMClassifier()
    model.fit(x, y)
    np.testing.assert_array_equal(model.predict(x), y)
    path = tmp_path / ("model.joblib" if kind == "rf" else "model.npz")
    if kind == "rf":
        save_random_forest(model, path, {"is_synthetic": True})
    else:
        assert reduced_multivariate_features(np.zeros((1,46))).shape == (1,141)
        model.save(path, {"is_synthetic": True})
    bundle = load_model(path)
    np.testing.assert_allclose(model_probabilities(bundle, x, None), model.predict_proba(x))
    assert bundle["metadata"]["is_synthetic"]


def test_cnn_training_checkpoint_roundtrip(tmp_path):
    rng = np.random.default_rng(9)
    images = rng.uniform(size=(8,32,32)).astype("float32")
    labels = np.array(["bird", "drone"]*4)
    path = tmp_path / "cnn.pt"
    model, history, classes = train_cnn(images, labels, images[:4], labels[:4], path,
                                      {"is_synthetic": True}, epochs=2, seed=9)
    assert len(history) == 2 and set(classes) == {"bird", "drone"}
    predicted = cnn_probabilities(model, images)
    np.testing.assert_allclose(predicted.sum(axis=1), 1, atol=1e-6)
    bundle = load_model(path)
    np.testing.assert_allclose(cnn_probabilities(bundle["model"], images), predicted)


def test_prediction_exports_and_provenance(tmp_path, cfg):
    times = [10, 10 + cfg.frame_repetition_time_s, 10 + 2*cfg.frame_repetition_time_s, 11]
    path = save_fixture(tmp_path, cfg, np.stack([tone(cfg)]*4), frame_timestamps_s=times)
    feature = dsp.extract_burst_features(tone(cfg), cfg, include_ftmdse=False)["features"]
    x = np.stack([feature, feature+.1, -feature, -feature-.1])
    model = make_random_forest().fit(x, ["drone", "drone", "bird", "bird"])
    model_path = tmp_path / "model.joblib"
    save_random_forest(model, model_path, {"is_synthetic": True, "feature_dimension": 46,
        "feature_names": dsp.FEATURE_NAMES, "training_configs": [config_signature(cfg)]})
    result = predict_recording(path, model_path, output_dir=tmp_path / "prediction")
    assert result["predicted_class"] == "drone"
    assert result["input_is_synthetic"] and result["model_trained_with_synthetic"]
    assert len(result["rows"]) == 4 and result["rows"][0]["track_label"] == "warmup"
    assert result["rows"][-1]["track_label"] == "warmup"
    assert [r["time_s"] for r in result["rows"]] == times
    assert Path(result["csv"]).is_file() and Path(result["plot"]).is_file()


def test_acquisition_missing_sdk_has_actionable_error(tmp_path, cfg, monkeypatch):
    from src.acquire_bgt60tr13c import acquire
    monkeypatch.setitem(sys.modules, "ifxradarsdk", None)
    monkeypatch.setitem(sys.modules, "ifxradarsdk.fmcw", None)
    with pytest.raises(RuntimeError, match="Radar Development Kit"):
        acquire(cfg, 1, tmp_path/"capture.npy", "session", "drone")


@pytest.mark.parametrize("outcome", ["ok", "usb_error", "invalid_readback"])
def test_finite_sdk_capture_disconnect_and_readback(tmp_path, cfg, monkeypatch, outcome):
    """Mock transport only: checks SDK lifecycle, not actual board operation."""
    from src.acquire_bgt60tr13c import acquire

    class Device:
        closed = False
        calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def get_sensor_type(self):
            return "BGT60TR13C"

        def create_simple_sequence(self, config):
            self.requested = config
            return config

        def set_acquisition_sequence(self, sequence):
            self.sequence = sequence

        def get_acquisition_sequence(self):
            # Mimic SDK timing quantization, so requested and effective differ.
            chirp_loop = SimpleNamespace(num_repetitions=64, repetition_time_s=.0001 if outcome == "invalid_readback" else .000501,
                sub_sequence=SimpleNamespace(contents=SimpleNamespace(chirp=self.requested.chirp)))
            return SimpleNamespace(loop=SimpleNamespace(repetition_time_s=cfg.frame_repetition_time_s,
                sub_sequence=SimpleNamespace(contents=SimpleNamespace(loop=chirp_loop))))

        def get_next_frame(self):
            self.calls += 1
            if outcome == "usb_error":
                raise RuntimeError("mock USB disconnect")
            return [tone(cfg)]

    device = Device()
    monkeypatch.setitem(sys.modules, "ifxradarsdk", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "ifxradarsdk.fmcw", SimpleNamespace(DeviceFmcw=lambda: device))
    monkeypatch.setitem(sys.modules, "ifxradarsdk.fmcw.types", SimpleNamespace(
        FmcwSimpleSequenceConfig=SimpleNamespace, FmcwSequenceChirp=SimpleNamespace))
    output = tmp_path / "capture.npy"
    if outcome == "invalid_readback":
        with pytest.raises(ValueError, match="shorter"):
            acquire(cfg, 2, output, "mock_session", "drone")
        assert not output.exists() and device.calls == 0
    elif outcome == "usb_error":
        with pytest.raises(RuntimeError, match="USB disconnect"):
            acquire(cfg, 2, output, "mock_session", "drone")
        assert not output.exists()
    else:
        acquire(cfg, 2, output, "mock_session", "drone")
        recording = load_recording(output)
        assert recording.frames.shape == (2,3,64,128)
        assert recording.config.chirp_repetition_time_s == .000501
        assert recording.metadata["configuration_readback_verified"] is True
        assert device.calls == 2
    assert device.closed

