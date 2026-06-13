import numpy as np
import pandas as pd

from src.models.tune_logistic import (
    baseline_params,
    is_valid_candidate,
    search_candidates,
    select_threshold,
)


def test_search_candidates_includes_baseline_and_is_deterministic():
    first = search_candidates(iterations=30, random_state=42)
    second = search_candidates(iterations=30, random_state=42)

    assert first == second
    assert first[0] == baseline_params()
    assert len(first) == 30
    assert all(is_valid_candidate(candidate) for candidate in first)


def test_threshold_selection_uses_f1_then_recall():
    y_true = pd.Series([0, 0, 1, 1])
    scores = np.array([0.1, 0.6, 0.55, 0.9])

    _, threshold, metrics = select_threshold(y_true, scores, 0.5, 0.7, 0.1)

    assert threshold == 0.5
    assert metrics["f1"] > 0.79
    assert metrics["recall"] == 1.0
