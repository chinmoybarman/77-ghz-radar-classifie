"""Bounded validation-only tuning; retain the original test set and baseline files."""
from __future__ import annotations

import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import f1_score

from src.dashboard_data import EXPERIMENT_ROOT
from src.evaluate_pipeline import write_json
from src.kth77_dataset import evaluate_external, plot_confusion
from src.models_ml import (load_model, make_random_forest, model_probabilities,
    save_random_forest, train_cnn)

OUTPUT = EXPERIMENT_ROOT.parent / "kth77_refined"


def macro_f1(labels, scores, classes):
    return float(f1_score(labels, classes[np.argmax(scores, axis=1)], average="macro"))


def main():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise ValueError("Refined output already exists; preserving the earlier run.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "models").mkdir()
    # These candidates and the selection rule are fixed before the test archive is opened.
    protocol = {"selection": "highest validation segment macro-F1; ties keep the earlier candidate",
        "rf_candidates": [{"max_features": .5, "min_samples_leaf": 1},
                          {"max_features": 1.0, "min_samples_leaf": 1},
                          {"max_features": .5, "min_samples_leaf": 2}],
        "rf_trees": 500, "cnn_epochs": 32, "cnn_patience": 7,
        "seed": 42, "fusion_rf_weights": [0, .25, .5, .75, 1],
        "development_note": "The original test results were previously inspected. Tuning uses validation only; a fresh independent test set is still needed."}
    write_json(OUTPUT / "protocol.json", protocol)
    data = {}
    for part in ("train", "val"):
        with np.load(EXPERIMENT_ROOT / "processed" / f"{part}.npz", allow_pickle=False) as saved:
            data[part] = {key: saved[key].copy() for key in ("features", "spectrograms", "labels")}
    selected, candidates = {}, []
    metadata = json.loads((EXPERIMENT_ROOT / "metrics.json").read_text())["metadata"]
    classes = np.array(["bird", "drone", "human"])
    for name, filename in (("Random Forest", "rf_kth77.joblib"), ("CNN", "cnn_kth77.pt")):
        path = EXPERIMENT_ROOT / "models" / filename
        bundle = load_model(path)
        scores = model_probabilities(bundle, data["val"]["features"], data["val"]["spectrograms"])
        score = macro_f1(data["val"]["labels"], scores, classes)
        selected[name] = {"path": str(path), "macro_f1": score, "val_scores": scores}
        candidates.append({"name": name, "candidate": "baseline", "validation_macro_f1": score})
    for index, settings in enumerate(protocol["rf_candidates"]):
        model = make_random_forest(42)
        model.set_params(classifier__n_estimators=500,
            **{f"classifier__{key}": value for key, value in settings.items()})
        model.fit(data["train"]["features"], data["train"]["labels"])
        scores = model.predict_proba(data["val"]["features"])
        score = macro_f1(data["val"]["labels"], scores, classes)
        candidate = {"name": "Random Forest", "candidate": index, **settings, "validation_macro_f1": score}
        candidates.append(candidate)
        print(candidate, flush=True)
        if score > selected["Random Forest"]["macro_f1"]:
            path = OUTPUT / "models" / f"rf_candidate_{index}.joblib"
            save_random_forest(model, path, {**metadata, "refinement": candidate})
            selected["Random Forest"] = {"path": str(path), "macro_f1": score, "val_scores": scores}
    cnn_path = OUTPUT / "models" / "cnn_32epochs.pt"
    print("Training the longer CNN; validation loss chooses its checkpoint.", flush=True)
    model, history, cnn_classes = train_cnn(data["train"]["spectrograms"], data["train"]["labels"],
        data["val"]["spectrograms"], data["val"]["labels"], cnn_path,
        {**metadata, "refinement": "up to 32 epochs; validation-loss early stopping"},
        epochs=32, patience=7, seed=42)
    write_json(OUTPUT / "cnn_history.json", {"history": history})
    scores = model_probabilities({"kind": "cnn", "model": model},
        data["val"]["features"], data["val"]["spectrograms"])
    score = macro_f1(data["val"]["labels"], scores, cnn_classes)
    candidates.append({"name": "CNN", "candidate": "32epochs", "validation_macro_f1": score})
    print(candidates[-1], flush=True)
    if score > selected["CNN"]["macro_f1"]:
        selected["CNN"] = {"path": str(cnn_path), "macro_f1": score, "val_scores": scores}
    best_weight, best_score = 0., -1.
    for weight in protocol["fusion_rf_weights"]:
        scores = weight * selected["Random Forest"]["val_scores"] + (1 - weight) * selected["CNN"]["val_scores"]
        score = macro_f1(data["val"]["labels"], scores, classes)
        if score > best_score:
            best_weight, best_score = weight, score
    manifest = {"models": {name: {k: v for k, v in row.items() if k != "val_scores"} for name, row in selected.items()},
        "fusion_rf_weight": best_weight, "fusion_validation_macro_f1": best_score,
        "candidates": candidates, "protocol": protocol}
    # Seal selection before reading the test features or labels.
    write_json(OUTPUT / "selected_models.json", manifest)
    summary = {"selection": manifest, "classes": classes.tolist(), "models": {}}
    with np.load(EXPERIMENT_ROOT / "processed" / "test.npz", allow_pickle=False) as saved:
        data["test"] = {key: saved[key].copy() for key in ("features", "spectrograms", "labels")}
    for part in ("val", "test"):
        records = json.loads((EXPERIMENT_ROOT / "processed" / f"{part}_segments.json").read_text())["records"]
        dataset = {"labels": data[part]["labels"], "records": records}
        scores_by_model = {}
        for name, row in selected.items():
            bundle = load_model(row["path"])
            scores_by_model[name] = model_probabilities(bundle, data[part]["features"], data[part]["spectrograms"])
        scores_by_model["Combined"] = (best_weight * scores_by_model["Random Forest"] +
                                        (1 - best_weight) * scores_by_model["CNN"])
        for name, scores in scores_by_model.items():
            slug = {"Random Forest": "rf", "CNN": "cnn", "Combined": "combined"}[name]
            metrics = evaluate_external(dataset, scores, classes, OUTPUT / f"{slug}_{part}_predictions.csv")
            summary["models"].setdefault(name, {})[part] = metrics
            np.save(OUTPUT / f"{slug}_{part}_scores.npy", scores)
            if part == "test":
                plot_confusion(metrics, classes, slug, OUTPUT / f"{slug}_test_confusion.png")
                print(f"{name} test: {metrics['segments']['accuracy']:.4f}, macro-F1 {metrics['segments']['macro_f1']:.4f}", flush=True)
    write_json(OUTPUT / "metrics.json", summary)
    return summary


if __name__ == "__main__":
    main()

