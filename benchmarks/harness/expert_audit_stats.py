"""Agreement statistics for expert / LLM judge calibration."""

from __future__ import annotations

import math
from typing import List, Optional


def pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    """Pearson r; returns None for degenerate inputs."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx2 = sum((x - mx) ** 2 for x in xs)
    dy2 = sum((y - my) ** 2 for y in ys)
    if dx2 == 0 or dy2 == 0:
        return None
    return num / math.sqrt(dx2 * dy2)


def cohens_kappa(labels_a: List[bool], labels_b: List[bool]) -> Optional[float]:
    """Cohen's κ on paired binary labels."""
    n = len(labels_a)
    if n == 0 or len(labels_b) != n:
        return None
    obs = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    pa_pos = sum(1 for a in labels_a if a) / n
    pb_pos = sum(1 for b in labels_b if b) / n
    pe = pa_pos * pb_pos + (1 - pa_pos) * (1 - pb_pos)
    if pe == 1:
        return None
    return (obs - pe) / (1 - pe)


def summarise_paired_scores(
    scores_a: List[float],
    scores_b: List[float],
    *,
    pass_mark: float,
) -> dict:
    """Pearson r, pass agreement, κ, and mean delta for paired scores."""
    paired = [(a, b) for a, b in zip(scores_a, scores_b)
              if a is not None and b is not None]
    n = len(paired)
    if n == 0:
        return {
            "n": 0,
            "score_correlation": None,
            "pass_agreement": None,
            "cohens_kappa": None,
            "mean_score_delta": None,
            "bias_flag": False,
        }
    xs = [a for a, _ in paired]
    ys = [b for _, b in paired]
    a_pass = [a >= pass_mark for a in xs]
    b_pass = [b >= pass_mark for b in ys]
    delta = sum(b - a for a, b in paired) / n
    pass_agree = sum(1 for a, b in zip(a_pass, b_pass) if a == b) / n
    return {
        "n": n,
        "score_correlation": pearson(xs, ys),
        "pass_agreement": pass_agree,
        "cohens_kappa": cohens_kappa(a_pass, b_pass),
        "mean_score_delta": delta,
        "bias_flag": abs(delta) > 5.0,
    }
