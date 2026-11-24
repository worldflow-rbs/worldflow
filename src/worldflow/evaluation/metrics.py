"""Native 3D trajectory metrics for the WorldFlow point-query protocol."""

import numpy as np


PWT_THRESHOLDS = {"pwt_1cm": 0.01, "pwt_2cm": 0.02, "pwt_5cm": 0.05}


def trajectory_metrics(predicted_offsets, true_increments):
    """Score frame-zero-relative predictions against successive ground-truth steps, in meters."""
    predicted = np.asarray(predicted_offsets, dtype=np.float32)
    steps = np.asarray(true_increments, dtype=np.float32)
    if predicted.shape != steps.shape or steps.ndim != 3 or steps.shape[-1] != 3 or min(steps.shape) == 0:
        raise ValueError("Expected matching, non-empty (future frames, points, 3) trajectories")
    if not np.isfinite(predicted).all() or not np.isfinite(steps).all():
        raise ValueError("Cannot evaluate non-finite trajectories")
    error = np.linalg.norm(predicted - np.cumsum(steps, axis=0), axis=-1)
    if not np.isfinite(error).all():
        raise ValueError("Non-finite trajectory distance")
    thresholds = {name: float((error <= threshold).mean()) for name, threshold in PWT_THRESHOLDS.items()}
    return {
        "ade": float(error.mean()),
        "fde": float(error[-1].mean()),
        **thresholds,
        "pwt": sum(thresholds.values()) / len(thresholds),
    }


class MetricMeans:
    """Accumulate per-clip means in constant memory."""

    def __init__(self):
        self.samples = 0
        self.totals = {}

    def add(self, metrics):
        self.samples += 1
        for name, value in metrics.items():
            self.totals[name] = self.totals.get(name, 0.0) + value

    def result(self):
        return {"samples": self.samples, **{name: total / self.samples for name, total in self.totals.items()}}
