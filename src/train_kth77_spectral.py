"""Train the spectral classifier and compare single-scan and causal decisions on fixed KTH rows."""
import argparse
import csv
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.nn import functional as F

from src.dashboard_data import EXPERIMENT_ROOT, FUSION_ROOT, resolve_model_path
from src.evaluate_pipeline import write_json
from src.kth77_dataset import CLASSES, CROP, SWEEPS, evaluate_external, extract_segment_features, load_dataset, segment_indices
from src.models_ml import load_model, model_probabilities
from src.kth77_spectral import (FORMAT, MAX_GAP_S, MODEL_DIR, SpectralNet, blend_scores,
                                history_indices, load_spectral, mean_scores, spectral_scores, spectral_views)


WINDOWS = (1, 3, 5, 9)
WEIGHTS = (0., .25, .5, .75, 1.)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def centers(m, indices):
    return m.samples[2*SWEEPS:3*SWEEPS][CROP][:, indices].T


def training_set(rows, device, training_cap):
    signals, coarse, fine, weights = [], [], [], []
    fine_names = sorted({m.source_label for m in rows})
    for m in rows:
        ids = segment_indices(m, training_cap)
        # Each class gets equal expected draws and its recordings get equal weight.
        class_rows = sum(r.label == m.label for r in rows)
        signals.extend(spectral_views(centers(m, ids[k:k+512])) for k in range(0, len(ids), 512))
        coarse.extend([CLASSES.index(m.label)] * len(ids))
        fine.extend([fine_names.index(m.source_label)] * len(ids))
        weights.extend([1 / (class_rows * len(ids))] * len(ids))
    return (torch.as_tensor(np.concatenate(signals), device=device),
            torch.tensor(coarse, device=device), torch.tensor(fine, device=device),
            torch.tensor(weights, device=device), fine_names)


def partition(measurements, rows, part):
    records = read_json(EXPERIMENT_ROOT / "processed" / f"{part}_segments.json")["records"]
    allowed = {row["session_id"] for row in rows}
    with (FUSION_ROOT / f"combined_{part}_predictions.csv").open(newline="") as f:
        prior = list(csv.DictReader(f))
    scores = np.load(FUSION_ROOT / f"combined_{part}_scores.npy", allow_pickle=False)
    if len(prior) != len(records) or scores.shape != (len(records), len(CLASSES)):
        raise ValueError("Invalid baseline score count.")
    histories, times = [], []
    for rec, old, score in zip(records, prior, scores):
        key, idx = rec["measurement_id"], rec["segment_index"]
        m = measurements[key]
        if (key not in allowed or rec["partition"] != part or m.label != rec["label"] or m.edge[idx]
                or old["measurement_id"] != key or int(old["segment_index"]) != idx
                or old["prediction"] != CLASSES[score.argmax()] or m.times_s[idx] != rec["time_s"]):
            raise ValueError("Baseline and archive segments do not align.")
        indices = history_indices(idx, m.times_s, m.edge, max(WINDOWS))
        histories.append(centers(m, indices))
        times.append(m.times_s[indices])
    return records, scores, histories, times


def fit(rows, validation, output, device, seed, training_cap):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    train, labels, fine, weights, fine_names = training_set(rows, device, training_cap)
    model = SpectralNet(len(fine_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.004)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32, eta_min=.00012)
    valid_records, _, histories, _ = validation
    val_signals = np.stack([x[-1] for x in histories])
    val_labels = np.array([CLASSES.index(r["label"]) for r in valid_records])
    best, best_epoch, history = -1., 0, []
    for epoch in range(1, 33):
        model.train()
        batch_ids = torch.multinomial(weights, 16384, replacement=True).split(256)
        for ids in batch_ids:
            coarse_out, fine_out = model(train[ids])
            loss = (F.cross_entropy(coarse_out, labels[ids], label_smoothing=.03)
                    + .25 * F.cross_entropy(fine_out, fine[ids], label_smoothing=.03))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        schedule.step()
        model.eval()
        val = spectral_scores(model, val_signals)
        score = float(f1_score(val_labels, val.argmax(1), average="macro"))
        history.append(score)
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"format": FORMAT, "classes": list(CLASSES), "fine_classes": fine_names,
                        "state": model.state_dict()}, output / "spectral.pt")
        print(f"Epoch {epoch:2d}   validation macro F1 {score:.4f}", flush=True)
        if epoch - best_epoch >= 10:
            break
    model.load_state_dict(torch.load(output / "spectral.pt", map_location=device, weights_only=True)["state"])
    return model.eval(), {"epochs": len(history), "best_epoch": best_epoch, "validation_macro_f1": history,
                          "training_segments": len(labels), "training_rows": len(rows),
                          "parameters": sum(p.numel() for p in model.parameters())}


def window_means(raw, lengths):
    return {window: mean_scores(raw, lengths, window) for window in WINDOWS}


def window_scores(model, histories):
    return window_means(spectral_scores(model, np.concatenate(histories)),
                        np.array([len(x) for x in histories]))


def legacy_context(data, selection):
    records, old, histories, _ = data
    echoes, positions, gather = [], {}, []
    for rec, history in zip(records, histories):
        for offset, echo in enumerate(history, start=1-len(history)):
            key = (rec["measurement_id"], rec["segment_index"] + offset)
            if key not in positions:
                positions[key] = len(echoes)
                echoes.append(echo)
            gather.append(positions[key])
    chunks = []
    models = {name: load_model(resolve_model_path(r["path"])) for name, r in selection["models"].items()}
    weight = selection["fusion_rf_weight"]
    for start in range(0, len(echoes), 256):
        extracted = [extract_segment_features(x) for x in echoes[start:start+256]]
        features = np.stack([r["features"] for r in extracted])
        images = np.stack([r["spectrogram"] for r in extracted])
        probs = {}
        for name, bundle in models.items():
            order = [list(bundle["classes"]).index(label) for label in CLASSES]
            probs[name] = model_probabilities(bundle, features, images)[:, order]
        chunks.append(weight * probs["Random Forest"] + (1 - weight) * probs["CNN"])
    scores = window_means(np.concatenate(chunks)[gather], np.array([len(x) for x in histories]))
    np.testing.assert_allclose(scores[1], old, atol=1e-5)
    return scores


def run_legacy_control(archive, output):
    """Temporal averaging control using only the previously trained forest and CNN."""
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new control output directory.")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    rows, _ = load_dataset(archive)
    by_id = {m.identity: m for m in rows}
    parts = read_json(EXPERIMENT_ROOT / "split_manifest.json")["partitions"]
    manifest = read_json(FUSION_ROOT / "selected_models.json")
    summary = {}
    for part in ("val", "test"):
        data = partition(by_id, parts[part], part)
        print(f"Scoring legacy control on {part} histories.", flush=True)
        scores = legacy_context(data, manifest)
        records = data[0]
        labels = [CLASSES.index(r["label"]) for r in records]
        if part == "val":
            window = max(WINDOWS, key=lambda w: f1_score(labels, scores[w].argmax(1), average="macro"))
            write_json(output / "selection.json", {"window": window})
        dataset = {"records": records, "labels": np.asarray(CLASSES)[labels]}
        summary[part] = evaluate_external(dataset, scores[window], CLASSES, output / f"{part}.csv")
        summary[part].update(accuracy_difference(records, data[1], scores[window]))
    write_json(output / "metrics.json", {"window": window, **summary})


def select(validation, spectral):
    records, old, _, _ = validation
    truth = [CLASSES.index(r["label"]) for r in records]
    grid = []
    for window in WINDOWS:
        for weight in WEIGHTS:
            preds = blend_scores(old, spectral[window], weight).argmax(1)
            grid.append(dict(window=window, spectral_weight=weight,
                             val_macro_f1=float(f1_score(truth, preds, average="macro")),
                             val_accuracy=float(np.mean(preds == truth))))
    def best(rows):
        return max(rows, key=lambda row: row["val_macro_f1"])
    return {"single": best(r for r in grid if r["window"] == 1), "sequence": best(grid)}, grid


def accuracy_difference(records, old, new):
    truth = np.array([CLASSES.index(r["label"]) for r in records])
    groups = np.array([r["measurement_id"] for r in records])
    delta = (new.argmax(1) == truth).astype(int) - (old.argmax(1) == truth)
    rows = np.array([[delta[groups == g].sum(), (groups == g).sum()] for g in np.unique(groups)])
    rng = np.random.default_rng(42)
    draws = rows[rng.integers(len(rows), size=(10000, len(rows)))].sum(axis=1)
    return {"additional_correct": int(delta.sum()), "paired_recording_bootstrap_95pp":
            (100 * np.quantile(draws[:, 0] / draws[:, 1], [.025, .975])).tolist()}


def evaluate(part, data, new, choices, output):
    records, old, histories, times = data
    dataset = {"records": records, "labels": np.array([r["label"] for r in records])}
    results = {}
    for name, choice in {"baseline": {"window": 1, "spectral_weight": 0}, **choices}.items():
        window = choice["window"]
        scores = blend_scores(old, new[window], choice["spectral_weight"])
        metrics = evaluate_external(dataset, scores, CLASSES, output / f"{name}_{part}.csv")
        count = np.array([min(len(t), window) for t in times])
        spans = [t[-1] - t[-n] for t, n in zip(times, count)]
        results[name] = {**metrics, **accuracy_difference(records, old, scores),
                         "full_window_count": int((count == window).sum()),
                         "median_span_s": float(np.median(spans)),
                         "p95_span_s": float(np.percentile(spans, 95))}
        print(f"{part} {name}: accuracy {metrics['segments']['accuracy']:.4f}, "
              f"macro F1 {metrics['segments']['macro_f1']:.4f}", flush=True)
    if part == "test":
        # Keep actual preceding echoes for replay in the app. No predictions or labels are inputs.
        w = choices["sequence"]["window"]
        count = np.array([min(len(x), w) for x in histories])
        np.savez_compressed(output / "replay.npz", signals=np.concatenate([x[-w:] for x in histories]),
                            counts=count,
                            times=np.concatenate([t[-w:] for t in times]),
                            measurement=np.array([r["measurement_id"] for r in records]),
                            segment=np.array([r["segment_index"] for r in records]))
    return results


def run(archive, output, device, capped_control=False):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new output folder to preserve the previous experiment.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python installation.")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    rows, fingerprints = load_dataset(archive)
    by_id = {m.identity: m for m in rows}
    parts = read_json(EXPERIMENT_ROOT / "split_manifest.json")["partitions"]
    ids = [r["session_id"] for p in parts.values() for r in p]
    if len(set(ids)) != len(ids):
        raise ValueError("Overlapping measurement rows.")
    caps = [64] if capped_control else [64, 0]
    protocol = {"seed": 42, "windows": WINDOWS, "weights": WEIGHTS, "max_gap_s": MAX_GAP_S,
                "training_caps_per_recording": caps,
                "selection_and_serving_device": "cpu",
                "source_sha256": fingerprints["sha256"], "device": device, "torch": str(torch.__version__),
                "python": platform.python_version(), "numpy": np.__version__,
                "hardware": torch.cuda.get_device_name() if device == "cuda" else "CPU",
                "selection": "For each checkpoint choose window and blend by validation macro F1, shortest window then lowest weight on ties. Choose training budget by single-scan validation macro F1, then sequence macro F1, then lower training cap."}
    write_json(output / "protocol.json", protocol)
    validation = partition(by_id, parts["val"], "val")
    training = [by_id[r["session_id"]] for r in parts["train"]]
    print(f"Training on {protocol['hardware']} with {len(training)} training recordings.", flush=True)
    candidates = []
    for cap in caps:
        candidate_dir = output / ("capped" if cap else "full")
        candidate_dir.mkdir()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        fit_start = time.perf_counter()
        model, history = fit(training, validation, candidate_dir, device, 42, cap)
        history["fit_seconds"] = round(time.perf_counter() - fit_start, 1)
        if device == "cuda":
            history["peak_cuda_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        choices, grid = select(validation, window_scores(model.cpu(), validation[2]))
        candidates.append({"cap": cap, "folder": candidate_dir.name, "training": history,
                           "selection": choices, "validation_grid": grid})
        del model
    winner = max(candidates, key=lambda row: (row["selection"]["single"]["val_macro_f1"],
                                               row["selection"]["sequence"]["val_macro_f1"]))
    choices = winner["selection"]
    shutil.copyfile(output / winner["folder"] / "spectral.pt", output / "spectral.pt")
    model = load_spectral(output / "spectral.pt")
    val_scores = window_scores(model, validation[2])
    write_json(output / "selection.json", choices)
    results = {"protocol": protocol, "training": winner["training"], "selected_folder": winner["folder"],
               "selection": choices, "validation_grid": winner["validation_grid"], "candidates": candidates,
               "val": evaluate("val", validation, val_scores, choices, output)}
    if not capped_control:
        test = partition(by_id, parts["test"], "test")
        results["test"] = evaluate("test", test, window_scores(model, test[2]), choices, output)
    results["elapsed_seconds"] = round(time.perf_counter() - start, 1)
    # The dashboard treats this file as the completion marker.
    staged_metrics = output / "metrics.pending"
    write_json(staged_metrics, results)
    staged_metrics.replace(output / "metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument("--legacy-control", action="store_true", help="Evaluate temporal averaging without retraining.")
    controls.add_argument("--capped-control", action="store_true", help="Train on the original 64-clip cap and evaluate validation only.")
    args = parser.parse_args()
    if args.legacy_control:
        run_legacy_control(args.input, args.output or MODEL_DIR.parent / "kth77_temporal_control")
    else:
        default = MODEL_DIR.parent / "kth77_spectral_capped" if args.capped_control else MODEL_DIR
        run(args.input, args.output or default, args.device, args.capped_control)
