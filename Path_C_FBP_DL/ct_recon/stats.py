from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.stats import rankdata, wilcoxon


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for mean of values."""
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty array")

    rng = np.random.default_rng(random_state)
    n = values.size
    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = np.mean(sample)

    alpha = (1.0 - ci) / 2.0
    low = float(np.quantile(means, alpha))
    high = float(np.quantile(means, 1.0 - alpha))
    return low, high


def paired_wilcoxon(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Paired Wilcoxon signed-rank test for matched samples."""
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    if x.size == 0:
        raise ValueError("x and y cannot be empty")

    stat, p_value = wilcoxon(x, y, zero_method="wilcox", correction=False, alternative="two-sided")
    return {"statistic": float(stat), "p_value": float(p_value)}


def paired_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 10000,
    random_state: int = 42,
) -> Dict[str, float]:
    """Two-sided paired permutation test on mean difference."""
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    diffs = x - y
    obs = float(np.mean(diffs))
    rng = np.random.default_rng(random_state)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(n_permutations, diffs.size), replace=True)
    perm_means = np.mean(signs * diffs, axis=1)
    p_value = float((np.sum(np.abs(perm_means) >= abs(obs)) + 1) / (n_permutations + 1))
    return {"observed_mean_diff": obs, "p_value": p_value}


def rank_biserial_effect_size(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-biserial effect size for paired samples based on signed ranks."""
    if x.shape != y.shape:
        raise ValueError("x and y must have same shape")

    d = x - y
    d = d[d != 0]
    if d.size == 0:
        return 0.0

    ranks = rankdata(np.abs(d), method="average")
    w_pos = np.sum(ranks[d > 0])
    w_neg = np.sum(ranks[d < 0])
    denom = w_pos + w_neg
    if denom == 0:
        return 0.0
    return float((w_pos - w_neg) / denom)


def holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni corrected p-values."""
    m = p_values.size
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=np.float64)

    running_max = 0.0
    for i, idx in enumerate(order):
        adj = (m - i) * p_values[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(1.0, running_max)

    return adjusted
