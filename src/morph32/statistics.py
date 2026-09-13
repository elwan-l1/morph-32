"""Small, dependency-free summaries; intervals describe task sampling, not training variance."""

import math
import statistics


def accuracy_interval(correct, count):
    """Wilson 95% binomial interval. Repeated retrieval prompts are not independent tasks."""
    if not 0 <= correct <= count or count < 1:
        raise ValueError("Require 0 <= correct <= count and count > 0")
    p, z = correct / count, 1.959963984540054
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return dict(
        correct=correct,
        count=count,
        accuracy=p,
        ci95=[center - radius, center + radius],
        interval_method="Wilson; fixed benchmark task sampling only",
    )


def timing_summary(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Expected positive finite timing measurements")
    q = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else values * 3
    return dict(
        count=len(values),
        median=statistics.median(values),
        q1=q[0],
        q3=q[2],
        minimum=min(values),
        maximum=max(values),
    )
