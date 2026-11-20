"""Reproducible RF, reduced-polynomial TER, and optional PyTorch classifiers.

TERRM uses Oh et al. (2019), equation (3), with the reduced multivariate
regressor in Toh et al. (2004), Appendix A.  It is not ordinary polynomial
ridge regression. The default targets are 0/1 (tau=eta=0.5). The published
tau=1, eta=0 setting would make the printed equation's targets identical;
this explicit parameter departure avoids claiming an exact reproduction.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.special import softmax
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_SCHEMA_VERSION = 1


def make_random_forest(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=250,
            class_weight="balanced", random_state=seed, n_jobs=-1)),
    ])


def reduced_multivariate_features(x: np.ndarray, order: int = 2) -> np.ndarray:
    """Toh 2004 Appendix A RM expansion, including its redundant columns.

    At r=2: [1, x, x**2, sum(x), sum(x)**2, x*sum(x)].
    The dimension is 1+r+d*(2*r-1), not the full-polynomial dimension.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("RM input must be a finite samples-by-features matrix")
    if not isinstance(order, int) or not 1 <= order <= 5:
        raise ValueError("RM order must be an integer in [1,5]")
    sums = x.sum(axis=1, keepdims=True)
    terms = [np.ones((len(x), 1))]
    terms.extend(x ** i for i in range(1, order + 1))
    terms.extend(sums ** i for i in range(1, order + 1))
    terms.extend(x * sums ** (i - 1) for i in range(2, order + 1))
    return np.concatenate(terms, axis=1)


class TERRMClassifier:
    """Class-balanced, one-vs-rest analytic TER with train-only min-max scaling.

    Scores are uncalibrated regressions; ``predict_proba`` is a softmax score
    transform for reporting, not a calibrated probability of detection.
    """
    def __init__(self, order: int = 2, regularization: float = 1e-4,
                 tau: float = 0.5, eta: float = 0.5):
        self.order, self.regularization = order, regularization
        self.tau, self.eta = tau, eta

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TERRMClassifier":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y).astype(str)
        if x.ndim != 2 or not len(x) or len(x) != len(y) or not np.isfinite(x).all():
            raise ValueError("TERRM requires aligned finite nonempty X and y")
        if self.regularization <= 0 or self.eta <= 0:
            raise ValueError("regularization and eta must be positive; eta=0 has identical targets")
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("TERRM needs at least two training classes")
        self.minimum_ = x.min(axis=0)
        span = x.max(axis=0) - self.minimum_
        self.scale_ = np.where(span > np.finfo(float).eps, span, 1.0)
        self.n_features_in_ = x.shape[1]
        p = reduced_multivariate_features((x - self.minimum_) / self.scale_, self.order)
        self.coef_ = np.empty((p.shape[1], len(self.classes_)))
        ridge = self.regularization * np.eye(p.shape[1])
        for i, label in enumerate(self.classes_):
            positive = y == label
            weights = np.where(positive, 1.0 / positive.sum(), 1.0 / (~positive).sum())
            targets = np.where(positive, self.tau + self.eta, self.tau - self.eta)
            self.coef_[:, i] = np.linalg.solve(
                ridge + p.T @ (weights[:, None] * p), p.T @ (weights * targets))
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise ValueError("TERRM has not been fitted")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError("Feature dimension differs from fitted TERRM")
        p = reduced_multivariate_features((x - self.minimum_) / self.scale_, self.order)
        return p @ self.coef_

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.decision_function(x), axis=1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(x), axis=1)]

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, coef=self.coef_, minimum=self.minimum_, scale=self.scale_,
            classes=self.classes_, order=self.order, regularization=self.regularization,
            tau=self.tau, eta=self.eta, metadata=json.dumps(metadata or {}))

    @classmethod
    def load(cls, path: str | Path) -> tuple["TERRMClassifier", dict]:
        with np.load(path, allow_pickle=False) as saved:
            model = cls(int(saved["order"]), float(saved["regularization"]),
                        float(saved["tau"]), float(saved["eta"]))
            model.coef_ = saved["coef"].copy()
            model.minimum_, model.scale_ = saved["minimum"].copy(), saved["scale"].copy()
            model.classes_ = saved["classes"].astype(str)
            model.n_features_in_ = len(model.minimum_)
            metadata = json.loads(str(saved["metadata"]))
        return model, metadata


def _torch():
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError("CNN requires a working PyTorch installation; install requirements.txt") from exc
    return torch


def _cnn_class():
    """Construct the real nn.Module class lazily so RF-only use needs no torch."""
    torch = _torch()
    nn = torch.nn

    class CompactRadarCNN(nn.Module):
        def __init__(self, num_classes: int = 4):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 24, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)))
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(24 * 4 * 4, 48),
                                            nn.ReLU(), nn.Dropout(0.15), nn.Linear(48, num_classes))

        def forward(self, x):
            return self.classifier(self.features(x))

    return CompactRadarCNN


def __getattr__(name: str):
    if name == "CompactRadarCNN":
        model_class = _cnn_class()
        globals()[name] = model_class
        return model_class
    raise AttributeError(name)


def _images(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        x = x[:, None, :, :]
    if x.ndim != 4 or x.shape[1:] != (1, 32, 32) or not np.isfinite(x).all():
        raise ValueError("CNN input must be finite (samples,32,32) or (samples,1,32,32)")
    return x


def train_cnn(train_images: np.ndarray, train_labels: np.ndarray,
              val_images: np.ndarray, val_labels: np.ndarray,
              checkpoint: str | Path, metadata: dict | None = None,
              epochs: int = 12, patience: int = 4, batch_size: int = 16,
              seed: int = 42) -> tuple[Any, list[dict], np.ndarray]:
    """CPU deterministic Adam training; only validation loss chooses checkpoint."""
    torch = _torch()
    if epochs < 1 or patience < 1 or batch_size < 1:
        raise ValueError("epochs, patience and batch_size must be positive")
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    classes = np.unique(np.asarray(train_labels).astype(str))
    lookup = {label: i for i, label in enumerate(classes)}
    if len(classes) < 2 or not len(val_labels):
        raise ValueError("CNN needs at least two training classes and nonempty validation")
    if not set(np.asarray(val_labels).astype(str)) <= set(classes):
        raise ValueError("Validation contains classes absent from training")
    train_x = torch.from_numpy(_images(train_images))
    val_x = torch.from_numpy(_images(val_images))
    train_y = torch.tensor([lookup[str(x)] for x in train_labels], dtype=torch.long)
    val_y = torch.tensor([lookup[str(x)] for x in val_labels], dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(train_x, train_y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0)
    model = __getattr__("CompactRadarCNN")(len(classes))
    counts = np.bincount(train_y.numpy(), minlength=len(classes))
    weights = torch.tensor(len(train_y) / (len(classes) * counts), dtype=torch.float32)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history, best_state, best_loss, stale = [], None, float("inf"), 0
    for epoch in range(epochs):
        model.train()
        loss_total = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(y_batch)
        model.eval()
        with torch.no_grad():
            logits = model(val_x)
            val_loss = float(criterion(logits, val_y))
            val_accuracy = float((logits.argmax(1) == val_y).float().mean())
        history.append({"epoch": epoch + 1, "train_loss": loss_total / len(dataset),
                        "val_loss": val_loss, "val_accuracy": val_accuracy})
        if val_loss < best_loss - 1e-7:
            best_loss, stale, best_state = val_loss, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "classes": classes.tolist(),
        "metadata": metadata or {}, "history": history,
        "schema_version": MODEL_SCHEMA_VERSION}, checkpoint)
    return model, history, classes


def cnn_probabilities(model: Any, images: np.ndarray) -> np.ndarray:
    torch = _torch()
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(_images(images))), dim=1).cpu().numpy()


def save_random_forest(model: Pipeline, path: str | Path, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata,
                 "schema_version": MODEL_SCHEMA_VERSION}, path)


def load_model(path: str | Path) -> dict:
    """Load locally generated trusted artifacts. Never load untrusted joblib files."""
    path = Path(path)
    if path.suffix == ".npz":
        model, metadata = TERRMClassifier.load(path)
        return {"kind": "terrm", "model": model, "classes": model.classes_, "metadata": metadata}
    if path.suffix == ".pt":
        torch = _torch()
        saved = torch.load(path, map_location="cpu", weights_only=True)
        classes = np.asarray(saved["classes"])
        model = __getattr__("CompactRadarCNN")(len(classes))
        model.load_state_dict(saved["state_dict"])
        model.eval()
        return {"kind": "cnn", "model": model, "classes": classes, "metadata": saved["metadata"]}
    if path.suffix == ".joblib":
        saved = joblib.load(path)
        return {"kind": "rf", "model": saved["model"], "classes": saved["model"].classes_,
                "metadata": saved["metadata"]}
    raise ValueError("Model must be .joblib (RF), .npz (TERRM), or .pt (CNN)")


def model_probabilities(bundle: dict, features: np.ndarray, spectrograms: np.ndarray) -> np.ndarray:
    if bundle["kind"] == "cnn":
        return cnn_probabilities(bundle["model"], spectrograms)
    return bundle["model"].predict_proba(features)

