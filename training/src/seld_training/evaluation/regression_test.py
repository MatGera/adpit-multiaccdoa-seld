"""Regression test: assert SELD-error does not increase beyond threshold."""

from __future__ import annotations

import json
from pathlib import Path


def check_seld_regression(
    current_metrics: dict[str, float],
    baseline_path: str | Path,
    max_seld_error_increase: float = 0.05,
) -> tuple[bool, str]:
    """Check if current SELD metrics regressed beyond threshold.

    Args:
        current_metrics: Current evaluation metrics dict.
        baseline_path: Path to baseline metrics JSON file.
        max_seld_error_increase: Maximum allowed increase in SELD-error.

    Returns:
        (passed, message) tuple.
    """
    baseline_path = Path(baseline_path)

    if not baseline_path.exists():
        return True, "No baseline found, skipping regression check"

    with open(baseline_path) as f:
        baseline = json.load(f)

    baseline_error = baseline.get("seld_error", 1.0)
    current_error = current_metrics.get("seld_error", 1.0)

    delta = current_error - baseline_error

    if delta > max_seld_error_increase:
        return False, (
            f"SELD-error regression: {current_error:.4f} vs baseline {baseline_error:.4f} "
            f"(delta={delta:+.4f}, max allowed={max_seld_error_increase})"
        )

    return True, (
        f"SELD-error OK: {current_error:.4f} vs baseline {baseline_error:.4f} "
        f"(delta={delta:+.4f})"
    )
