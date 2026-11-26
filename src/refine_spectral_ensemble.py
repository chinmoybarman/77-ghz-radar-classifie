"""Compare independently trained spectral CNNs using fixed measurement splits."""
import argparse
from pathlib import Path

import numpy as np
import torch

from src.dashboard_data import EXPERIMENT_ROOT
from src.evaluate_pipeline import write_json
from src.kth77_dataset import load_dataset
from src.kth77_spectral import MODEL_DIR, load_spectral, spectral_scores
from src.train_kth77_spectral import (
    accuracy_difference, evaluate, fit, partition, read_json, select, window_means,
)


def run(archive, output):
    if output.exists():
        raise ValueError("Choose a new experiment output directory.")
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires the CUDA training environment.")
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    protocol = {
        "seeds": [7, 123], "training_cap": 0, "ensemble": "Equal probability mean",
        "selection": "Highest single-scan validation macro F1, then recent-scan macro F1.",
        "test_policy": "Evaluate only the validation-selected candidate.",
        "baseline_checkpoint": str(MODEL_DIR / "spectral.pt"),
        "hardware": torch.cuda.get_device_name(),
    }
    write_json(output / "protocol.json", protocol)
    rows, fingerprints = load_dataset(archive)
    protocol["source_sha256"] = fingerprints["sha256"]
    parts = read_json(EXPERIMENT_ROOT / "split_manifest.json")["partitions"]
    split_ids = [r["session_id"] for group in parts.values() for r in group]
    if len(set(split_ids)) != len(split_ids):
        raise ValueError("Measurement partitions overlap.")
    by_id = {r.identity: r for r in rows}
    training = [by_id[r["session_id"]] for r in parts["train"]]
    validation = partition(by_id, parts["val"], "val")
    echoes = np.concatenate(validation[2])
    lengths = np.array([len(x) for x in validation[2]])
    checkpoints = {"baseline": MODEL_DIR / "spectral.pt"}
    raw_scores = {"baseline": spectral_scores(load_spectral(checkpoints["baseline"]), echoes)}
    histories = {}
    for seed in protocol["seeds"]:
        name = f"seed_{seed}"
        folder = output / name
        folder.mkdir()
        print(f"Training {name}", flush=True)
        model, history = fit(training, validation, folder, "cuda", seed, 0)
        histories[name] = history
        checkpoints[name] = folder / "spectral.pt"
        raw_scores[name] = spectral_scores(model.cpu(), echoes)
        del model
        torch.cuda.empty_cache()
    candidates = []
    members = [[name] for name in checkpoints] + [list(checkpoints)]
    for names in members:
        raw = np.mean([raw_scores[name] for name in names], axis=0)
        choices, grid = select(validation, window_means(raw, lengths))
        candidates.append({"members": names, "selection": choices, "validation_grid": grid})
    winner = max(candidates, key=lambda item: (
        item["selection"]["single"]["val_macro_f1"],
        item["selection"]["sequence"]["val_macro_f1"],
        -len(item["members"]),
    ))
    result = {"protocol": protocol, "training": histories, "candidates": candidates,
              "winner": winner}
    write_json(output / "validation_selection.json", result)
    print("Validation-selected members:", winner["members"], flush=True)
    if winner["members"] == ["baseline"]:
        write_json(output / "metrics.json", result)
        print("Baseline retained by validation.", flush=True)
        return
    raw = np.mean([raw_scores[name] for name in winner["members"]], axis=0)
    result["val"] = evaluate("val", validation, window_means(raw, lengths),
                             winner["selection"], output)
    test = partition(by_id, parts["test"], "test")
    echoes = np.concatenate(test[2])
    lengths = np.array([len(x) for x in test[2]])
    selected = [spectral_scores(load_spectral(checkpoints[name]), echoes)
                for name in winner["members"]]
    scores = window_means(np.mean(selected, axis=0), lengths)
    result["test"] = evaluate("test", test, scores, winner["selection"], output)
    baseline = window_means(spectral_scores(load_spectral(checkpoints["baseline"]), echoes), lengths)
    from src.kth77_spectral import blend_scores
    baseline_choices = read_json(MODEL_DIR / "selection.json")
    result["comparison"] = {}
    for mode in ("single", "sequence"):
        old_choice, new_choice = baseline_choices[mode], winner["selection"][mode]
        old = blend_scores(test[1], baseline[old_choice["window"]], old_choice["spectral_weight"])
        new = blend_scores(test[1], scores[new_choice["window"]], new_choice["spectral_weight"])
        result["comparison"][mode] = accuracy_difference(test[0], old, new)
    write_json(output / "metrics.json", result)
    print(result["comparison"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output)
