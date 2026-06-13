from src.models.compare_baselines_xai import build_leaderboard_dataframe


def make_record(model, f1, roc_auc, fit_time):
    metrics = {
        "accuracy": 0.7,
        "precision": 0.4,
        "recall": 0.8,
        "f1": f1,
        "roc_auc": roc_auc,
        "tn": 10,
        "fp": 2,
        "fn": 1,
        "tp": 4,
    }
    return {
        "model": model,
        "validation": metrics,
        "test": metrics,
        "timing": {
            "fit_time_sec": fit_time,
            "validation_predict_time_sec": 0.01,
            "test_predict_time_sec": 0.02,
        },
    }


def test_leaderboard_ranks_by_validation_f1_then_auc_and_keeps_timing():
    records = [
        make_record("model_b", 0.5, 0.8, 2.0),
        make_record("model_a", 0.6, 0.7, 1.0),
        make_record("model_c", 0.5, 0.9, 3.0),
    ]

    leaderboard = build_leaderboard_dataframe(records)

    assert leaderboard["model"].tolist() == ["model_a", "model_c", "model_b"]
    assert leaderboard["fit_time_sec"].tolist() == [1.0, 3.0, 2.0]
