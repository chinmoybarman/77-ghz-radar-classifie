"""Multi-resolution spectral classifier for cropped, range-compressed KTH echoes."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.kth77_dataset import CLASSES

FORMAT = "kth77-center150-multires-v1"
MAX_GAP_S = 0.15  # The archive's nominal interval between scans is 0.1 seconds.
MODEL_DIR = Path(__file__).resolve().parents[1] / "experiments" / "kth77_spectral"


def spectral_views(signals):
    """Build absolute and peak-aligned Doppler spectra at two time resolutions."""
    x = np.asarray(signals)
    if x.ndim != 2 or x.shape[1] != 150 or x.dtype.kind != "c" or not np.isfinite(x).all():
        raise ValueError("Expected a batch of finite, complex, central-150 KTH echoes.")
    # Removing only the overall gain preserves differences within the sweep.
    scale = np.maximum(np.abs(x).max(axis=1, keepdims=True), np.finfo(float).tiny)
    x = x / scale
    full = np.abs(np.fft.fft(x * np.hanning(150))) ** 2
    local = np.stack([np.abs(np.fft.fft(x[:, i:i+66] * np.hanning(66), n=150)) ** 2
                      for i in (0, 42, 84)], axis=1)
    indices = (np.arange(150) - 75 + full.argmax(axis=1)[:, None]) % 150
    aligned = np.take_along_axis(np.concatenate((full[:, None], local), axis=1),
                                 indices[:, None, :], axis=2)
    power = np.concatenate((np.fft.fftshift(full, axes=-1)[:, None], aligned), axis=1)
    ref = np.maximum(full.mean(axis=1)[:, None, None], 1e-15)
    return (np.log(np.maximum(power / ref, 1e-7)).clip(-12, 6) / 4).astype(np.float32)


class SpectralNet(nn.Module):
    def __init__(self, fine_classes):
        super().__init__()
        blocks = []
        widths = (5, 32, 48, 64)
        for n_in, n_out, kernel in zip(widths, widths[1:], (7, 5, 3)):
            blocks.extend([nn.Conv1d(n_in, n_out, kernel, padding=kernel // 2, bias=False),
                           nn.BatchNorm1d(n_out), nn.SiLU(), nn.AvgPool1d(2)])
        self.encoder = nn.Sequential(*blocks, nn.AdaptiveAvgPool1d(6), nn.Flatten(),
                                     nn.Linear(64 * 6, 80), nn.SiLU(), nn.Dropout(.2))
        self.coarse = nn.Linear(80, len(CLASSES))
        self.fine = nn.Linear(80, fine_classes)

    def forward(self, x):
        embedding = self.encoder(x)
        return self.coarse(embedding), self.fine(embedding)


def load_spectral(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["format"] != FORMAT or tuple(saved["classes"]) != CLASSES:
        raise ValueError("Incompatible spectral checkpoint.")
    model = SpectralNet(len(saved["fine_classes"]))
    model.load_state_dict(saved["state"])
    return model.eval()


@torch.inference_mode()
def spectral_scores(model, signals):
    device = next(model.parameters()).device
    chunks = []
    for start in range(0, len(signals), 512):
        features = torch.from_numpy(spectral_views(signals[start:start+512])).to(device)
        logits, _ = model(features)
        chunks.append(logits.softmax(1).cpu().numpy())
    return np.concatenate(chunks)


def history_indices(index, times, edge, window):
    """Use consecutive past scans only, resetting on omitted scans and invalid edges."""
    start = index
    while (index - start + 1 < window and start > 0 and not edge[start-1]
           and 0 < times[start] - times[start-1] <= MAX_GAP_S):
        start -= 1
    return np.arange(start, index + 1)


def replay_bank(path, records, anchors):
    with np.load(path, allow_pickle=False) as replay:
        signals, counts, times = replay["signals"], replay["counts"], replay["times"]
        ends = counts.cumsum()
        if (not np.array_equal(replay["measurement"], [r["measurement_id"] for r in records])
                or not np.array_equal(replay["segment"], [r["segment_index"] for r in records])
                or counts.shape != (len(records),) or np.any(counts < 1)
                or len(signals) != int(counts.sum()) or times.shape != (len(signals),)
                or not np.array_equal(signals[ends-1], anchors)):
            raise ValueError("Spectral replay and the recorded sample bank do not match.")
    return signals, counts, ends, times


def mean_scores(raw, lengths, window):
    ends = lengths.cumsum()
    count = np.minimum(lengths, window)
    prefix = np.vstack([np.zeros(len(CLASSES)), raw.cumsum(axis=0, dtype=np.float64)])
    return (prefix[ends] - prefix[ends - count]) / count[:, None]


def blend_scores(baseline, spectral, weight):
    return (1. - weight) * baseline + weight * spectral


def recent_history(signals, times, window):
    indices = history_indices(len(times)-1, times, np.zeros(len(times), dtype=bool), window)
    return signals[indices], times[indices]


def predict_file(path, root=MODEL_DIR, recent=False):
    """Score an exported sample without starting the app or loading unused models."""
    import json
    from src.dashboard_data import (MAX_UPLOAD_BYTES, analyze_signal, check_variation, combined_prediction,
                                    model_selection, read_upload_with_history)
    from src.models_ml import load_model
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError("Choose a sample smaller than 8 MB.")
    current, _, history = read_upload_with_history(path.read_bytes(), path.name)
    check_variation(current)
    choices = json.loads((root / "selection.json").read_text())
    use_history = recent and history is not None
    choice = choices["sequence" if use_history else "single"]
    echoes, times = recent_history(*history, choice["window"]) if use_history else (current[None], np.zeros(1))
    weight = choice["spectral_weight"]
    scores = np.zeros(len(CLASSES))
    if weight:
        model = load_spectral(root / "spectral.pt")
        scores += weight * spectral_scores(model, echoes).mean(axis=0, dtype=np.float64)
    if weight < 1:
        base = model_selection()
        bundles = {name: load_model(row["path"]) for name, row in base["models"].items()}
        _, predictions = analyze_signal(current, bundles)
        mixed = combined_prediction(predictions, base["fusion_rf_weight"])
        scores += (1 - weight) * np.array([mixed["scores"][label] for label in CLASSES])
    return {"prediction": CLASSES[int(scores.argmax())], "scores": dict(zip(CLASSES, map(float, scores))),
            "scans_used": len(echoes), "span_s": float(times[-1]-times[0])}


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Classify a recorded 77 GHz sample or timestamped sequence.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--run", type=Path, default=MODEL_DIR)
    parser.add_argument("--recent", action="store_true", help="Use previous scans when included in the input.")
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(predict_file(args.input, args.run, args.recent), indent=2))
