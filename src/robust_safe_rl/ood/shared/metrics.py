"""Reconstruction-error evaluation and detection metrics for the autoencoder.

Everything here operates on per-sample reconstruction MSE ("anomaly scores"):

- :func:`evaluate_errors` runs the model to produce per-sample errors + latents.
- :func:`detection_report` derives ID-based thresholds and reports false-alarm
  and detection rates at each.
- :func:`binary_curves` computes threshold-free AUROC / average precision.
- :func:`compute_confusion_matrix` / :func:`threshold_metrics` give confusion
  counts and rates at fixed thresholds.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def evaluate_errors(model, x, batch_size, device):
    """Return (per-sample reconstruction MSE, latent codes) as numpy arrays."""
    model.eval()
    errors, latents = [], []
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (xb,) in loader:
            err, z = model.reconstruction_error(xb.to(device), reduction="none")
            errors.append(err.cpu())
            latents.append(z.cpu())

    return torch.cat(errors).numpy(), torch.cat(latents).numpy()


def detection_report(id_errors, ood_errors):
    """Derive ID-based thresholds and report ID false-alarm / OOD detection.

    Returns
    -------
    thresholds : dict of threshold_name -> value.
    lines : list of (name, threshold, id_false_alarm_rate, ood_detection_rate).
    """
    thresholds = {
        "id_max": float(np.max(id_errors)),
        "id_p95": float(np.percentile(id_errors, 95.0)),
        "id_p99": float(np.percentile(id_errors, 99.0)),
        "id_p995": float(np.percentile(id_errors, 99.5)),
        "id_mean_plus_3std": float(np.mean(id_errors) + 3.0 * np.std(id_errors)),
    }
    lines = []
    for name, threshold in thresholds.items():
        lines.append((
            name,
            threshold,
            float(np.mean(id_errors > threshold)),
            float(np.mean(ood_errors > threshold)),
        ))
    return thresholds, lines


def compute_confusion_matrix(y_true, y_pred):
    """Return (tn, fp, fn, tp). Labels/preds: 0 = ID, 1 = OOD."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    return tn, fp, fn, tp


def binary_curves(id_errors, ood_errors):
    """Compute ROC / PR curves and their summaries (AUROC, average precision).

    Threshold-free: ranks all samples by score and sweeps every distinct value.
    """
    scores = np.concatenate([id_errors, ood_errors]).astype(float)
    labels = np.concatenate([
        np.zeros(len(id_errors), dtype=np.int8),
        np.ones(len(ood_errors), dtype=np.int8),
    ])
    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order]
    scores_sorted = scores[order]

    tp = np.cumsum(labels_sorted == 1)
    fp = np.cumsum(labels_sorted == 0)
    positives = max(int(np.sum(labels == 1)), 1)
    negatives = max(int(np.sum(labels == 0)), 1)

    distinct = np.r_[np.where(np.diff(scores_sorted))[0], len(scores_sorted) - 1]
    tpr = np.r_[0.0, tp[distinct] / positives, 1.0]
    fpr = np.r_[0.0, fp[distinct] / negatives, 1.0]
    thresholds = np.r_[np.inf, scores_sorted[distinct], -np.inf]
    # np.trapezoid is the current name; np.trapz was removed in NumPy 2.0.
    # Resolve lazily so referencing a missing name never raises at call time.
    trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    auroc = float(trapezoid(tpr, fpr))

    precision_all = tp / np.maximum(tp + fp, 1)
    recall_all = tp / positives
    positive_positions = np.where(labels_sorted == 1)[0]
    average_precision = float(np.mean(precision_all[positive_positions])) if len(positive_positions) else 0.0
    precision = np.r_[1.0, precision_all[distinct]]
    recall = np.r_[0.0, recall_all[distinct]]

    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": thresholds,
        "precision": precision,
        "recall": recall,
        "auroc": auroc,
        "average_precision": average_precision,
    }


def summarize(values):
    """Return count/mean/std/median/percentiles/max for an array of scores."""
    values = np.asarray(values, dtype=float)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "p995": float(np.percentile(values, 99.5)),
        "max": float(np.max(values)),
    }


def threshold_metrics(id_errors, ood_errors, checkpoint_thresholds):
    """Confusion counts + rates at both checkpoint and fresh-test thresholds."""
    candidates = {}
    for name in ("id_p95", "id_p99", "id_p995", "id_max", "id_mean_plus_3std"):
        if name in checkpoint_thresholds:
            candidates[name] = float(checkpoint_thresholds[name])

    # Always include thresholds derived from this independent ID set for diagnosis.
    candidates.update({
        "test_id_p95": float(np.percentile(id_errors, 95)),
        "test_id_p99": float(np.percentile(id_errors, 99)),
        "test_id_p995": float(np.percentile(id_errors, 99.5)),
        "test_id_max": float(np.max(id_errors)),
    })

    rows = []
    for name, threshold in candidates.items():
        id_pred = id_errors > threshold
        ood_pred = ood_errors > threshold
        tn = int(np.sum(~id_pred))
        fp = int(np.sum(id_pred))
        fn = int(np.sum(~ood_pred))
        tp = int(np.sum(ood_pred))
        rows.append({
            "threshold_name": name,
            "threshold": threshold,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "id_false_alarm_rate": float(fp / max(fp + tn, 1)),
            "ood_detection_rate": float(tp / max(tp + fn, 1)),
            "ood_miss_rate": float(fn / max(tp + fn, 1)),
            "balanced_accuracy": float(0.5 * (tn / max(tn + fp, 1) + tp / max(tp + fn, 1))),
        })
    return rows
