"""Bounded validation-only tuning; retain the original test set and baseline files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import f1_score

from src.dashboard_data import EXPERIMENT_ROOT, PROJECT_ROOT, resolve_model_path
from src.evaluate_pipeline import write_json
from src.kth77_dataset import evaluate_external, plot_confusion
from src.models_ml import (load_model, make_random_forest, model_probabilities,
    save_random_forest, train_cnn)

OUTPUT = EXPERIMENT_ROOT.parent / "kth77_refined"


def macro_f1(labels, scores, classes):
    return float(f1_score(labels, classes[np.argmax(scores, axis=1)], average="macro"))


def load_partition(part):
    with np.load(EXPERIMENT_ROOT / "processed" / f"{part}.npz", allow_pickle=False) as saved:
        return {key: saved[key] for key in ("features", "spectrograms", "labels")}


def evaluate_selection(output, manifest, classes):
    summary = {"selection": manifest, "classes": classes.tolist(), "models": {}}
    best_weight = manifest["fusion_rf_weight"]
    for part in ("val", "test"):
        data = load_partition(part)
        records = json.loads((EXPERIMENT_ROOT / "processed" / f"{part}_segments.json").read_text())["records"]
        dataset = {"labels": data["labels"], "records": records}
        scores_by_model = {}
        for name, row in manifest["models"].items():
            bundle = load_model(resolve_model_path(row["path"]))
            order = [list(bundle["classes"]).index(label) for label in classes]
            scores_by_model[name] = model_probabilities(bundle, data["features"], data["spectrograms"])[:, order]
        scores_by_model["Combined"] = (best_weight * scores_by_model["Random Forest"] +
                                        (1 - best_weight) * scores_by_model["CNN"])
        for name, scores in scores_by_model.items():
            slug = {"Random Forest": "rf", "CNN": "cnn", "Combined": "combined"}[name]
            metrics = evaluate_external(dataset, scores, classes, output / f"{slug}_{part}_predictions.csv")
            summary["models"].setdefault(name, {})[part] = metrics
            np.save(output / f"{slug}_{part}_scores.npy", scores)
            if part == "test":
                plot_confusion(metrics, classes, slug, output / f"{slug}_test_confusion.png")
                print(f"{name} test: {metrics['segments']['accuracy']:.4f}, macro-F1 {metrics['segments']['macro_f1']:.4f}", flush=True)
    write_json(output / "metrics.json", summary)
    return summary


def select_fusion(labels, classes, rf, cnn, weights):
    def score(weight):
        return macro_f1(labels, weight * rf + (1 - weight) * cnn, classes)

    best_weight = max(weights, key=score)
    return best_weight, score(best_weight)


def main(output=OUTPUT, *, extended=False, fusion_only=False):
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refined output already exists; preserving the earlier run.")
    output.mkdir(parents=True, exist_ok=True)
    classes = np.array(["bird", "drone", "human"])
    if fusion_only:
        prior = json.loads((OUTPUT / "selected_models.json").read_text(encoding="utf-8"))
        validation = load_partition("val")
        model_scores = {}
        for name, row in prior["models"].items():
            path = resolve_model_path(row["path"])
            row["path"] = str(path.relative_to(PROJECT_ROOT))
            bundle = load_model(path)
            order = [list(bundle["classes"]).index(label) for label in classes]
            model_scores[name] = model_probabilities(bundle, validation["features"], validation["spectrograms"])[:, order]
        incumbent_weight = prior["fusion_rf_weight"]
        weights = [incumbent_weight, *(i / 20 for i in range(21) if i / 20 != incumbent_weight)]
        best_weight, best_score = select_fusion(validation["labels"], classes,
            model_scores["Random Forest"], model_scores["CNN"], weights)
        prior["fusion_rf_weight"] = best_weight
        prior["fusion_validation_macro_f1"] = best_score
        prior["protocol"] = {**prior["protocol"], "selection": "Fixed checkpoints; highest validation segment macro-F1",
            "fusion_rf_weights": weights}
        write_json(output / "selected_models.json", prior)
        return evaluate_selection(output, prior, classes)

    (output / "models").mkdir()
    cnn_seeds = [42, 17, 73] if extended else [42]
    fusion_weights = [.25, *(i / 20 for i in range(21) if i != 5)] if extended else [0, .25, .5, .75, 1]
    # These candidates and the selection rule are fixed before the test archive is opened.
    protocol = {"selection": "highest validation segment macro-F1; ties keep the earlier candidate",
        "rf_candidates": [{"max_features": .5, "min_samples_leaf": 1},
                          {"max_features": 1.0, "min_samples_leaf": 1},
                          {"max_features": .5, "min_samples_leaf": 2}],
        "rf_trees": 500, "cnn_epochs": 32, "cnn_patience": 7,
        "seed": 42, "cnn_seeds": cnn_seeds, "fusion_rf_weights": fusion_weights,
        "development_note": "The original test results were previously inspected. Tuning uses validation only; a fresh independent test set is still needed."}
    write_json(output / "protocol.json", protocol)
    data = {part: load_partition(part) for part in ("train", "val")}
    selected, candidates = {}, []
    metadata = json.loads((EXPERIMENT_ROOT / "metrics.json").read_text())["metadata"]
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
            path = output / "models" / f"rf_candidate_{index}.joblib"
            save_random_forest(model, path, {**metadata, "refinement": candidate})
            selected["Random Forest"] = {"path": str(path), "macro_f1": score, "val_scores": scores}
    for seed in cnn_seeds:
        candidate_name = f"32epochs_seed{seed}" if extended else "32epochs"
        cnn_path = output / "models" / f"cnn_{candidate_name}.pt"
        print(f"Training CNN with seed {seed}; validation loss chooses its checkpoint.", flush=True)
        model, history, cnn_classes = train_cnn(data["train"]["spectrograms"], data["train"]["labels"],
            data["val"]["spectrograms"], data["val"]["labels"], cnn_path,
            {**metadata, "refinement": "up to 32 epochs; validation-loss early stopping"},
            epochs=32, patience=7, seed=seed)
        history_name = f"cnn_history_seed{seed}.json" if extended else "cnn_history.json"
        write_json(output / history_name, {"history": history})
        scores = model_probabilities({"kind": "cnn", "model": model},
            data["val"]["features"], data["val"]["spectrograms"])
        score = macro_f1(data["val"]["labels"], scores, cnn_classes)
        candidates.append({"name": "CNN", "candidate": candidate_name, "validation_macro_f1": score})
        print(candidates[-1], flush=True)
        if score > selected["CNN"]["macro_f1"]:
            selected["CNN"] = {"path": str(cnn_path), "macro_f1": score, "val_scores": scores}
    best_weight, best_score = select_fusion(data["val"]["labels"], classes,
        selected["Random Forest"]["val_scores"], selected["CNN"]["val_scores"], fusion_weights)
    models = {}
    for name, row in selected.items():
        path = Path(row["path"])
        models[name] = {"path": str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path),
                        "macro_f1": row["macro_f1"]}
    manifest = {"models": models,
        "fusion_rf_weight": best_weight, "fusion_validation_macro_f1": best_score,
        "candidates": candidates, "protocol": protocol}
    # Seal selection before reading the test features or labels.
    write_json(output / "selected_models.json", manifest)
    return evaluate_selection(output, manifest, classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--extended", action="store_true", help="Try three CNN seeds and a finer fusion grid.")
    mode.add_argument("--fusion-only", action="store_true", help="Keep the refined checkpoints and tune their blend.")
    args = parser.parse_args()
    out = args.output_root or (OUTPUT.parent / "kth77_fusion" if args.fusion_only else OUTPUT)
    main(out, extended=args.extended, fusion_only=args.fusion_only)
