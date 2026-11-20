"""Synthetic format fixtures for the external adapter; no hardware claims."""
import numpy as np
import pytest

from src.kth77_dataset import (
    CROP, PRF_HZ, CLASSES, FEATURE_NAMES, coarse_label, parse_measurements,
    load_dataset, inventory, grouped_partition, segment_indices,
    target_signal, extract_segment_features, evaluate_external,
)


def fixture_table():
    names = ["D1"]*3 + ["seagull"]*3 + ["human_walk"]*3 + ["CR"]
    table = np.empty((len(names), 6), dtype=object)
    for i,name in enumerate(names):
        phase = 2*np.pi*7*np.arange(256)/256
        samples = np.concatenate([(k+1)*np.exp(1j*phase) for k in range(5)])
        matrix = np.tile(samples[:,None], (1,8))
        table[i] = [np.array([name]), matrix, np.full((8,1),20+i),
                    np.arange(8)[:,None]*.1, np.array([1,2,3,1,2,3,1,2])[:,None],
                    np.array([1,0,0,0,0,0,0,0])[:,None]]
    return table


def test_source_label_mapping():
    assert all(coarse_label(f"D{i}") == "drone" for i in range(1,7))
    assert coarse_label("pigeon") == "bird"
    assert coarse_label("human_run") == "human"
    assert coarse_label("CR") is None
    with pytest.raises(ValueError,match="Unknown"):
        coarse_label("background")


def test_external_model_cannot_enter_bgt_predictor():
    from src.predict_offline import _check_compatibility
    with pytest.raises(ValueError, match="external-dataset model"):
        _check_compatibility({"domain":"kth77_range_compressed"},None,FEATURE_NAMES)


def test_unverified_archive_not_unpickled(tmp_path, monkeypatch):
    path=tmp_path/"unverified.npy"
    path.write_bytes(b"not the official archive")
    def disallowed(*args,**kwargs):
        raise AssertionError("Unverified file was unpickled")
    monkeypatch.setattr(np,"load",disallowed)
    with pytest.raises(ValueError,match="not loaded"):
        load_dataset(path)


def test_parse_inventory_and_exact_target_cell():
    measurements=parse_measurements(fixture_table())
    report=inventory(measurements,{"fixture":True})
    assert report["measurement_count"] == 10
    assert report["segment_count"] == 80
    assert report["edge_segments"] == 10
    assert len(report["author_split_measurements_in_multiple_subsets"]) == 10
    assert report["coarse_label_counts"]["excluded_reflector"] == 8
    expected=3*np.exp(2j*np.pi*7*np.arange(256)/256)[CROP]
    np.testing.assert_allclose(target_signal(measurements[0],1),expected)


@pytest.mark.parametrize("fault",["shape","real","nonfinite","label","time","split","edge","range_length"])
def test_malformed_measurements(fault):
    table=fixture_table()
    if fault == "shape":
        table[0,1]=np.zeros((100,8),complex)
    elif fault == "real":
        table[0,1]=table[0,1].real
    elif fault == "nonfinite":
        table[0,1][0,0]=np.nan
    elif fault == "label":
        table[0,0]=np.array(["unexpected"])
    elif fault == "time":
        table[0,3][1]=0
    elif fault == "split":
        table[0,4][0]=4
    elif fault == "edge":
        table[0,5][0]=2
    else:
        table[0,2]=np.ones(4)
    with pytest.raises(ValueError):
        parse_measurements(table)


def test_measurement_groups_are_disjoint_with_class_coverage():
    measurements=parse_measurements(fixture_table())
    parts=grouped_partition(measurements)
    sets=[{row["session_id"] for row in part} for part in parts.values()]
    assert sum(map(len,sets)) == 9
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    assert all({row["label"] for row in part} == set(CLASSES) for part in parts.values())
    assert parts == grouped_partition(measurements)


def test_subset_preserves_source_segment_identity_and_excludes_edges():
    m=parse_measurements(fixture_table())[0]
    np.testing.assert_array_equal(segment_indices(m,3),[1,4,7])
    np.testing.assert_array_equal(segment_indices(m,0),np.arange(1,8))
    with pytest.raises(ValueError):
        segment_indices(m,-1)


def test_external_feature_schema_and_tone_frequency():
    # A known tone exactly on the 150-sample Doppler grid.
    values=np.exp(2j*np.pi*12*np.arange(150)/150)
    result=extract_segment_features(values)
    assert result["features"].shape == (42,)
    assert len(set(FEATURE_NAMES)) == 42
    assert result["features"][34] == pytest.approx(12*PRF_HZ/150)
    assert result["spectrogram"].shape == (32,32)
    np.testing.assert_allclose(result["time_s"][[0,-1]],[32/PRF_HZ,112/PRF_HZ])
    assert result["frequency_hz"][0] == -PRF_HZ/2
    zero=extract_segment_features(np.zeros(150,dtype=complex))
    assert np.isfinite(zero["features"]).all()
    np.testing.assert_array_equal(zero["spectrogram"],0)


def test_external_evaluation_aggregates_only_within_measurements(tmp_path):
    records=[{"measurement_id":f"m{i}","label":name,"segment_index":j,"partition":"test"}
             for i,name in enumerate(CLASSES) for j in range(2)]
    data={"labels":np.array([r["label"] for r in records]),"records":records}
    scores=np.repeat(np.eye(3),2,axis=0)
    result=evaluate_external(data,scores,np.array(CLASSES),tmp_path/"predictions.csv")
    assert result["segments"]["n"] == 6
    assert result["segments"]["accuracy"] == 1
    assert result["measurement_average_scores"]["n"] == 3
    assert result["measurement_average_scores"]["accuracy"] == 1

