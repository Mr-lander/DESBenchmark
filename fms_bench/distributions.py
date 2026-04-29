"""Sampling utilities for DESBench stochasticity."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def gamma_from_mean_cv(mean: float, cv: float) -> Tuple[float, float]:
    """Return (shape, scale) for a Gamma distribution."""
    if cv <= 0:
        return 1.0, max(mean, 1e-6)
    shape = 1.0 / (cv * cv)
    scale = mean / shape
    return shape, scale


def sample_processing_time(mean: float, cv: float, rng: np.random.Generator) -> float:
    shape, scale = gamma_from_mean_cv(mean, cv)
    return float(rng.gamma(shape, scale))


def sample_poisson_count(lam: float, rng: np.random.Generator) -> int:
    return int(rng.poisson(lam))


def sample_exponential(mean: float, rng: np.random.Generator) -> float:
    if mean <= 0:
        return 0.0
    return float(rng.exponential(mean))


def hour_to_tou_price(hour: float, peak_window: Tuple[int, int], peak: float, offpeak: float) -> float:
    start, end = peak_window
    hour = int(math.floor(hour)) % 24
    if start <= hour < end:
        return peak
    return offpeak
