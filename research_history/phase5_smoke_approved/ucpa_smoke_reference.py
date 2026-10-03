"""Uncertainty-Compatible Peer Alignment (UCPA).

UCPA builds a personalized explanation prior for each federated client by
combining two notions of compatibility:

1. whole-explanation similarity (Jensen-Shannon divergence), and
2. coordinate-wise differences standardized by the two clients' sampling
   uncertainty for that attribution coordinate.

This module contains only the alignment operator.  It does not implement the
full xFedAlign surrogate-training pipeline.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-15


def _normalize_rows(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("expected a 2-D client x feature matrix")
    a = np.maximum(a, 0.0)
    s = a.sum(axis=1, keepdims=True)
    if np.any(s <= EPS):
        raise ValueError("each attribution vector must contain positive mass")
    return a / s


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.maximum(p, EPS); q = np.maximum(q, EPS)
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def align(
    explanation_means: np.ndarray,
    variances_of_means: np.ndarray,
    *,
    z: float = 2.0,
    h: float = 0.048,
    variance_floor_fraction: float = 0.05,
    lambda_floor: float = 0.0,
    return_diagnostics: bool = False,
):
    """Align federated explanation summaries with UCPA.

    Parameters
    ----------
    explanation_means:
        Non-negative array of shape (clients, features). Rows are normalized
        internally to probability-like attribution distributions.
    variances_of_means:
        Array with the same shape. Entry (i,j) estimates the sampling variance
        of client i's estimated mean attribution at coordinate j.
    z:
        Width of the coordinate uncertainty-compatibility kernel.
    h:
        Width of the whole-explanation JSD kernel.
    variance_floor_fraction:
        Stabilizing variance floor as a fraction of the median positive
        variance entry.
    lambda_floor:
        Optional floor on whole-explanation compatibility. The smoke-tested
        UCPA uses zero.
    return_diagnostics:
        If True, also return summary diagnostics.

    Returns
    -------
    aligned : ndarray, shape (clients, features)
        Personalized aligned explanation distributions.
    diagnostics : dict, optional
        Effective peer count and mean whole-explanation weight.
    """
    E = _normalize_rows(explanation_means)
    V = np.asarray(variances_of_means, dtype=np.float64)
    if V.shape != E.shape:
        raise ValueError("variances_of_means must match explanation_means")
    if np.any(~np.isfinite(E)) or np.any(~np.isfinite(V)):
        raise ValueError("inputs must be finite")
    if np.any(V < 0):
        raise ValueError("variances must be non-negative")
    if z <= 0 or h <= 0:
        raise ValueError("z and h must be positive")
    if not (0 <= lambda_floor <= 1):
        raise ValueError("lambda_floor must lie in [0,1]")

    C, _ = E.shape
    pos = V[V > 0]
    floor = (float(np.median(pos)) * variance_floor_fraction) if len(pos) else 1e-12

    # Coordinate-wise uncertainty compatibility.
    diff = E[:, None, :] - E[None, :, :]
    denom = np.sqrt(V[:, None, :] + V[None, :, :] + floor + EPS)
    d = np.abs(diff) / denom
    coordinate_weight = np.exp(-0.5 * (d / z) ** 2)

    # Whole-explanation compatibility.
    J = np.zeros((C, C), dtype=np.float64)
    for i in range(C):
        for k in range(i + 1, C):
            v = _jsd(E[i], E[k])
            J[i, k] = J[k, i] = v
    global_weight = np.exp(-J / h)
    global_weight = lambda_floor + (1.0 - lambda_floor) * global_weight

    W = coordinate_weight * global_weight[:, :, None]
    idx = np.arange(C)
    W[idx, idx, :] = 1.0

    aligned = np.einsum("ikd,kd->id", W, E) / np.maximum(W.sum(axis=1), EPS)
    aligned = _normalize_rows(aligned)

    if not return_diagnostics:
        return aligned

    effective_peers = (W.sum(axis=1) ** 2) / (np.sum(W ** 2, axis=1) + EPS)
    diagnostics = {
        "effective_peers_mean": float(effective_peers.mean()),
        "effective_peers_median": float(np.median(effective_peers)),
        "whole_explanation_weight_mean": float(global_weight.mean()),
        "whole_explanation_jsd_mean": float(J.mean()),
        "variance_floor": float(floor),
    }
    return aligned, diagnostics
