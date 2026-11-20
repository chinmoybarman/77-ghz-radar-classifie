"""Separate measured-data experiment for the official KTH/SAAB 77 GHz archive.

The input is range-compressed complex data, never BGT60TR13C ADC data.
Only the exact published archive is unpickled, after checksum verification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import signal

from src.dataset_manager import split_sessions
from src.evaluate_pipeline import write_json, _metrics
from src.models_ml import make_random_forest, save_random_forest, train_cnn, cnn_probabilities
from src.radar_dsp import (EMD_FEATURE_NAMES, extract_oh2019_emd_features,
                           normalize_complex_burst, standardized_spectrogram)

SOURCE_URL = "https://zenodo.org/records/5896641"
SOURCE_MD5 = "01d66ba7b1ccc04477a9e69b2813f251"
SOURCE_SHA256 = "e3caa44ae92a56561d05ab51426a71c05f5d7eebf153fc7480fb3409c18f528d"
PRF_HZ = 17_000.0
CARRIER_HZ = 77e9
RANGE_CELLS = 5
SWEEPS = 256
CROP = slice(53, 203)  # Exact zero-based central 150 samples, recorded in metadata.
FEATURE_NAMES = list(EMD_FEATURE_NAMES) + [
    "peak_doppler_hz", "centroid_hz", "bandwidth_hz", "spectral_entropy",
    "doppler_papr_db", "envelope_variation", "cadence_ratio", "spectral_edge_fraction",
]
FEATURE_VERSION = "kth77-center150-mti-emd34-spectral8-v1"
CLASSES = ("bird", "drone", "human")
BIRDS = {"seagull", "pigeon", "raven", "black-headed gull",
         "seagull and black-headed gull", "heron"}
NOTICE = "MEASURED KTH/SAAB 77 GHz DATA | held-out recordings | separate external experiment"


@dataclass
class Measurement:
    index: int
    source_label: str
    label: str | None
    samples: np.ndarray
    ranges_m: np.ndarray
    times_s: np.ndarray
    author_split: np.ndarray
    edge: np.ndarray

    @property
    def identity(self):
        return f"kth_measurement_{self.index:03d}"


def coarse_label(name):
    if name in {f"D{i}" for i in range(1, 7)}:
        return "drone"
    if name in BIRDS:
        return "bird"
    if name in {"human_walk", "human_run"}:
        return "human"
    if name == "CR":
        return None  # A reflector is not an empty/background scene.
    raise ValueError(f"Unknown KTH source label: {name!r}")


def verify_archive(path):
    path = Path(path).resolve()
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b""):
            md5.update(chunk)
            sha.update(chunk)
    fingerprints = {"md5": md5.hexdigest(), "sha256": sha.hexdigest(),
                    "bytes": path.stat().st_size, "path": str(path)}
    if fingerprints["md5"] != SOURCE_MD5 or fingerprints["sha256"] != SOURCE_SHA256:
        raise ValueError("Input does not match the verified official KTH v2 archive; object data were not loaded")
    return fingerprints


def parse_measurements(table):
    if not isinstance(table, np.ndarray) or table.ndim != 2 or table.shape[1] != 6:
        raise ValueError("KTH data must be a measurement-by-six object table")
    measurements = []
    for index, row in enumerate(table):
        name_array = np.asarray(row[0])
        if name_array.size != 1:
            raise ValueError(f"Measurement {index}: expected a single source label")
        name = str(name_array.reshape(-1)[0])
        samples = np.asarray(row[1])
        if samples.ndim != 2 or samples.shape[0] != RANGE_CELLS*SWEEPS or samples.shape[1] < 1:
            raise ValueError(f"Measurement {index}: expected complex (1280,segments) data")
        if samples.dtype.kind != "c" or not np.isfinite(samples).all():
            raise ValueError(f"Measurement {index}: samples must be finite complex range-compressed data")
        count = samples.shape[1]
        columns = []
        for name_field, value in zip(("range", "time", "split", "edge"), row[2:]):
            array = np.asarray(value)
            if array.dtype.kind not in "fiub" or array.size != count or array.ndim > 2:
                raise ValueError(f"Measurement {index}: invalid {name_field} vector")
            if array.ndim == 2 and 1 not in array.shape:
                raise ValueError(f"Measurement {index}: {name_field} must be a vector")
            vector = array.reshape(-1)
            if not np.isfinite(vector).all():
                raise ValueError(f"Measurement {index}: nonfinite {name_field}")
            columns.append(vector)
        ranges, times, splits, edges = columns
        if np.any(ranges < 0) or np.any(np.diff(times) <= 0):
            raise ValueError(f"Measurement {index}: invalid ranges or non-increasing time")
        if not np.isin(splits, [1, 2, 3]).all() or not np.isin(edges, [0, 1]).all():
            raise ValueError(f"Measurement {index}: invalid split/edge codes")
        measurements.append(Measurement(index, name, coarse_label(name), samples,
            ranges.astype(float), times.astype(float), splits.astype(int), edges.astype(bool)))
    if not measurements:
        raise ValueError("KTH table contains no measurements")
    return measurements


def load_dataset(path):
    fingerprints = verify_archive(path)
    # The published file stores nested NumPy objects. Never enable pickle for
    # arbitrary files: both fingerprints above must match first.
    table = np.load(path, allow_pickle=True)
    if table.shape != (130, 6):
        raise ValueError("Verified archive must have the documented (130,6) table")
    return parse_measurements(table), fingerprints


def inventory(measurements, fingerprints):
    source_counts, class_counts = Counter(), Counter()
    shared = []
    rows = []
    for m in measurements:
        count = m.samples.shape[1]
        source_counts[m.source_label] += count
        class_counts[m.label or "excluded_reflector"] += count
        split_values, split_counts = np.unique(m.author_split, return_counts=True)
        if len(split_values) > 1:
            shared.append(m.identity)
        rows.append({"measurement_id": m.identity, "source_label": m.source_label,
            "label": m.label, "segments": count, "edge_segments": int(m.edge.sum()),
            "author_split_counts": dict(zip(map(str, split_values), map(int, split_counts))),
            "range_min_m": float(m.ranges_m.min()), "range_max_m": float(m.ranges_m.max()),
            "time_start_s": float(m.times_s[0]), "time_end_s": float(m.times_s[-1])})
    return {"source": SOURCE_URL, "fingerprints": fingerprints, "is_synthetic": False,
        "measurement_count": len(measurements), "segment_count": sum(source_counts.values()),
        "source_label_counts": dict(source_counts), "coarse_label_counts": dict(class_counts),
        "edge_segments": sum(int(m.edge.sum()) for m in measurements),
        "author_split_measurements_in_multiple_subsets": shared,
        "author_split_audit": "Segments from a measurement span supplied subsets; this run instead holds out entire measurement rows.",
        "measurements": rows}


def grouped_partition(measurements, seed=42):
    rows = [{"session_id": m.identity, "label": m.label, "index": m.index}
            for m in measurements if m.label is not None and np.any(~m.edge)]
    parts = split_sessions(rows, seed=seed)
    identities = [{row["session_id"] for row in part} for part in parts.values()]
    if any(a & b for i, a in enumerate(identities) for b in identities[i+1:]):
        raise ValueError("Measurement leakage")
    return parts


def segment_indices(measurement, limit=64):
    if not isinstance(limit, int) or limit < 0:
        raise ValueError("Segment limit must be a nonnegative integer (0 means all)")
    available = np.flatnonzero(~measurement.edge)
    if limit and len(available) > limit:
        available = available[np.linspace(0, len(available)-1, limit, dtype=int)]
    return available


def target_signal(measurement, segment_index):
    # Five RANGE cells, not five antennas. Center range is exactly cell 2.
    return measurement.samples[2*SWEEPS:3*SWEEPS, segment_index][CROP]


def extract_segment_features(values):
    values = np.asarray(values)
    if values.shape != (150,) or values.dtype.kind != "c" or not np.isfinite(values).all():
        raise ValueError("Expected a finite complex central-150 slow-time vector")
    centered = values - values.mean()
    normalized = normalize_complex_burst(centered)
    emd = extract_oh2019_emd_features(normalized)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(normalized*np.hanning(len(values)))))**2
    frequencies = np.fft.fftshift(np.fft.fftfreq(len(values), 1/PRF_HZ))
    total = spectrum.sum()
    if total > np.finfo(float).eps:
        probability = spectrum/total
        centroid = float(frequencies @ probability)
        positive = probability[probability > 0]
        peak = float(frequencies[spectrum.argmax()])
        width = float(np.sqrt(((frequencies-centroid)**2) @ probability))
        entropy = float(-(positive*np.log(positive)).sum()/np.log(len(probability)))
        papr = float(10*np.log10(spectrum.max()/spectrum.mean()))
        edge = float(probability[np.abs(frequencies) > .8*PRF_HZ/2].sum())
    else:
        peak = centroid = width = entropy = papr = edge = 0.0
    envelope = np.abs(normalized)
    mean = envelope.mean()
    variation = float(envelope.std()/mean) if mean > np.finfo(float).eps else 0.0
    ac = signal.correlate(envelope-mean, envelope-mean, mode="full")[len(envelope)-1:]
    cadence = float(np.clip(ac[2:len(envelope)//2+1].max()/ac[0], 0, 1)) if ac[0] > np.finfo(float).eps else 0.0
    features = np.r_[emd, peak, centroid, width, entropy, papr, variation, cadence, edge]
    image, freq, times = standardized_spectrogram(normalized, PRF_HZ, nperseg=64, return_axes=True)
    if features.shape != (42,) or not np.isfinite(features).all():
        raise ValueError("KTH feature contract requires 42 finite values")
    return {"features": features, "spectrogram": image, "frequency_hz": freq,
            "time_s": times, "signal": normalized}


def extract_partition(measurements, rows, partition, root, limit):
    by_index = {m.index: m for m in measurements}
    features, images, records = [], [], []
    for row in rows:
        m = by_index[row["index"]]
        for index in segment_indices(m, limit):
            result = extract_segment_features(target_signal(m, index))
            features.append(result["features"])
            images.append(result["spectrogram"])
            records.append({"measurement_id": m.identity, "source_label": m.source_label,
                "label": m.label, "segment_index": int(index), "time_s": float(m.times_s[index]),
                "range_m": float(m.ranges_m[index]), "author_split": int(m.author_split[index]),
                "partition": partition, "is_synthetic": False})
        print(f"{partition}: {m.identity} ({m.source_label}), {len(segment_indices(m, limit))} segments", flush=True)
    if not records:
        raise ValueError(f"No usable samples in {partition}")
    result = {"features": np.array(features), "images": np.array(images),
              "labels": np.array([r["label"] for r in records]), "records": records}
    directory = root/"processed"
    directory.mkdir(exist_ok=True)
    np.savez_compressed(directory/f"{partition}.npz", features=result["features"],
        spectrograms=result["images"], labels=result["labels"], feature_names=np.array(FEATURE_NAMES))
    write_json(directory/f"{partition}_segments.json", {"records": records, "feature_version": FEATURE_VERSION})
    return result


def evaluate_external(data, probability, classes, path):
    import csv
    classes = np.asarray(classes)
    if probability.shape != (len(data["labels"]), len(classes)) or not np.isfinite(probability).all():
        raise ValueError("Invalid external model prediction shape/values")
    predicted = classes[probability.argmax(axis=1)]
    grouped = {}
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(data["records"][0]) + ["prediction", "confidence_score"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(data["records"]):
            grouped.setdefault(row["measurement_id"], []).append(i)
            writer.writerow({**row, "prediction": predicted[i], "confidence_score": float(probability[i].max())})
    truth, aggregate = [], []
    for positions in grouped.values():
        truth.append(data["labels"][positions[0]])
        aggregate.append(classes[probability[positions].mean(axis=0).argmax()])
    return {"segments": _metrics(data["labels"], predicted, classes),
            "measurement_average_scores": _metrics(np.array(truth), np.array(aggregate), classes),
            "aggregation_note": "Mean of selected segment scores per held-out measurement; not temporal tracking."}


def plot_examples(measurements, train_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    training = {row["index"] for row in train_rows}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for col, label in enumerate(CLASSES):
        m = next(m for m in measurements if m.index in training and m.label == label and np.any(~m.edge))
        valid = np.flatnonzero(~m.edge)
        index = int(valid[len(valid)//2])
        result = extract_segment_features(target_signal(m, index))
        time_ms = np.arange(150)/PRF_HZ*1000
        axes[0,col].plot(time_ms, result["signal"].real, label="Real", lw=.8)
        axes[0,col].plot(time_ms, result["signal"].imag, label="Imaginary", lw=.8, alpha=.7)
        axes[0,col].set(title=f"{label}: {m.source_label}, row {m.index}", xlabel="Within-segment time (ms)", ylabel="Normalized amplitude")
        axes[0,col].legend(fontsize=8)
        image = axes[1,col].pcolormesh(result["time_s"]*1000, result["frequency_hz"],
            result["spectrogram"], shading="auto", cmap="magma", vmin=0, vmax=1)
        axes[1,col].set(xlabel="STFT window center (ms)", ylabel="Doppler (Hz)")
    fig.colorbar(image, ax=axes[1,:].tolist(), label="Normalized STFT power")
    fig.suptitle("Measured KTH/SAAB 77 GHz: examples from training recordings", fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(metrics, classes, name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matrix = np.array(metrics["segments"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes,
           yticklabels=classes, xlabel="Predicted", ylabel="Measured label",
           title=f"{name.upper()} · held-out KTH recordings")
    for (i,j), value in np.ndenumerate(matrix):
        ax.text(j,i,str(value),ha="center",va="center")
    fig.colorbar(im, ax=ax, label="Selected segments")
    fig.text(.5,.015,"Measured external 77 GHz data; BGT60TR13C transfer untested",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(path,dpi=150)
    plt.close(fig)


def run_experiment(measurements, fingerprints, root, limit=64, seed=42, models=("rf","cnn"), epochs=8):
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Choose an empty output directory to preserve earlier experiments")
    if limit < 0 or epochs < 1 or not models or any(m not in ("rf","cnn") for m in models):
        raise ValueError("Use a nonnegative segment limit, positive epochs and rf/cnn models")
    root.mkdir(parents=True,exist_ok=True)
    parts = grouped_partition(measurements, seed)
    audit = inventory(measurements, fingerprints)
    write_json(root/"inventory.json", audit)
    write_json(root/"split_manifest.json", {"partitions": parts, "seed": seed,
        "split_unit": "complete measurement row; true acquisition sessions not supplied",
        "max_segments_per_measurement": limit, "exclude_edge_segments": True})
    plot_examples(measurements, parts["train"], root/"measured_examples.png")
    data = {name: extract_partition(measurements, rows, name, root, limit) for name,rows in parts.items()}
    classes = np.unique(data["train"]["labels"])
    if set(classes) != set(CLASSES) or any(set(d["labels"]) != set(classes) for d in data.values()):
        raise ValueError("Every partition must contain bird, drone and human")
    metadata = {"domain":"kth77_range_compressed", "is_synthetic":False,
        "source":SOURCE_URL, "fingerprints":fingerprints, "feature_version":FEATURE_VERSION,
        "feature_names":FEATURE_NAMES, "feature_dimension":42, "input_shape":[32,32],
        "carrier_hz":CARRIER_HZ, "prf_hz":PRF_HZ, "range_cell_zero_based":2,
        "sweep_slice_zero_based":[53,203], "subtract_segment_mean":True,
        "no_bgt_adc_range_fft_or_aoa":True, "seed":seed,
        "training_measurements":[r["session_id"] for r in parts["train"]]}
    summary = {"is_synthetic":False,"notice":NOTICE,"classes":classes.tolist(),
        "metadata":metadata,"sample_counts":{k:len(d["labels"]) for k,d in data.items()},"models":{},
        "limitations":["External 77 GHz data do not validate the 60 GHz BGT60TR13C.",
            "Only measurement-row independence is known; days/scenes/target instances may overlap.",
            "A capped subset is a baseline, not paper-performance replication.",
            "Source class, range and SNR may be confounded; nuisance transfer is untested.",
            "Reflectors excluded; no empty-scene background class or detection false-alarm evaluation.",
            "Classifier confidence scores are uncalibrated."]}
    model_dir=root/"models"
    model_dir.mkdir()
    for name in dict.fromkeys(models):
        print(f"Training measured-data {name} on {len(data['train']['labels'])} segments",flush=True)
        if name == "rf":
            model = make_random_forest(seed).fit(data["train"]["features"],data["train"]["labels"])
            checkpoint=model_dir/"rf_kth77.joblib"
            save_random_forest(model, checkpoint, metadata)
            model_classes=model.classes_
            probabilities={k:model.predict_proba(d["features"]) for k,d in data.items() if k!="train"}
        else:
            checkpoint=model_dir/"cnn_kth77.pt"
            model,history,model_classes=train_cnn(data["train"]["images"],data["train"]["labels"],
                data["val"]["images"],data["val"]["labels"],checkpoint,metadata,epochs=epochs,seed=seed)
            write_json(root/"cnn_history.json",{"history":history,"is_synthetic":False})
            probabilities={k:cnn_probabilities(model,d["images"]) for k,d in data.items() if k!="train"}
        scores={k:evaluate_external(data[k],p,model_classes,root/f"{name}_{k}_predictions.csv") for k,p in probabilities.items()}
        summary["models"][name]={"checkpoint":str(checkpoint),**scores}
        plot_confusion(scores["test"],model_classes,name,root/f"{name}_test_confusion.png")
        write_json(root/"metrics.json",summary)
        print(f"{name}: held-out segment accuracy={scores['test']['segments']['accuracy']:.3f}, macro-F1={scores['test']['segments']['macro_f1']:.3f}",flush=True)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--inspect-only",action="store_true")
    parser.add_argument("--max-segments-per-measurement",type=int,default=64,help="0 processes all non-edge segments")
    parser.add_argument("--models",nargs="+",choices=("rf","cnn"),default=["rf","cnn"])
    parser.add_argument("--epochs",type=int,default=8)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args(argv)
    try:
        if args.output_root.exists() and any(args.output_root.iterdir()):
            raise ValueError("Choose an empty output directory")
        measurements,fingerprints=load_dataset(args.input)
        print(f"Verified official archive: {len(measurements)} measurements",flush=True)
        if args.inspect_only:
            write_json(args.output_root/"inventory.json",inventory(measurements,fingerprints))
        else:
            run_experiment(measurements,fingerprints,args.output_root,args.max_segments_per_measurement,
                           args.seed,args.models,args.epochs)
    except (OSError,ValueError,RuntimeError,ImportError) as exc:
        parser.exit(2,f"KTH experiment failed: {exc}\n")
    print(f"Saved external 77 GHz outputs under {args.output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

