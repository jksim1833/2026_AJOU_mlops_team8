import json

import pandas as pd

from src.data.prepare_data import (
    build_eda_profile,
    calculate_outlier_summary,
    missing_value_profile,
    remove_exact_duplicates,
    select_eda_features,
    split_dataset,
    split_overlap_counts,
    write_eda_plots,
)


def test_remove_exact_duplicates_preserves_label_conflicts():
    df = pd.DataFrame(
        {
            "feature_a": [1.0, 1.0, 1.0, 2.0],
            "feature_b": [3.0, 3.0, 3.0, 4.0],
            "defect_label": [0, 0, 1, 1],
        }
    )

    deduplicated, removed = remove_exact_duplicates(df)

    assert removed == 1
    assert len(deduplicated) == 3
    assert set(deduplicated["defect_label"]) == {0, 1}


def test_split_has_no_exact_row_overlap():
    rows = 100
    df = pd.DataFrame(
        {
            "feature_a": range(rows),
            "feature_b": [value * 2 for value in range(rows)],
            "defect_label": [value % 2 for value in range(rows)],
        }
    )
    cfg = {
        "data": {
            "random_state": 42,
            "test_size": 0.15,
            "validation_size": 0.15,
        }
    }

    splits = split_dataset(df, "defect_label", cfg)

    split_sizes = {name: len(split) for name, split in splits.items()}
    assert sum(split_sizes.values()) == rows
    assert split_sizes == {"train": 69, "valid": 16, "test": 15}
    assert split_overlap_counts(splits) == {
        "train_valid": 0,
        "train_test": 0,
        "valid_test": 0,
    }


def test_select_eda_features_excludes_target_and_constants_with_stable_ties():
    df = pd.DataFrame(
        {
            "z_feature": [0.0, 0.0, 1.0, 1.0],
            "a_feature": [0.0, 0.0, 1.0, 1.0],
            "constant": [5.0, 5.0, 5.0, 5.0],
            "defect_label": [0, 0, 1, 1],
        }
    )

    selected, scores, constants = select_eda_features(
        df, "defect_label", top_n=2
    )

    assert selected == ["a_feature", "z_feature"]
    assert set(scores) == {"a_feature", "z_feature"}
    assert constants == ["constant"]


def test_outlier_summary_reports_without_changing_data():
    df = pd.DataFrame({"feature": [1.0, 1.0, 2.0, 2.0, 100.0]})
    original = df.copy(deep=True)

    summary = calculate_outlier_summary(df, ["feature"])

    assert summary[0]["count"] == 1
    assert summary[0]["ratio"] == 0.2
    pd.testing.assert_frame_equal(df, original)


def test_missing_value_profile_separates_raw_and_numeric_conversion_missing():
    raw_df = pd.DataFrame(
        {
            "feature_a": [1.0, None, 3.0],
            "feature_b": ["1", "invalid", "3"],
            "Short_Shot_1": [0, 1, 0],
        }
    )

    profile = missing_value_profile(
        raw_df,
        defect_columns=["Short_Shot_1"],
        target_column="defect_label",
    )

    assert profile["raw_missing_total"] == 1
    assert profile["after_numeric_conversion_total"] == 2
    assert profile["imputed_total"] == 2
    assert profile["processed_missing_total"] == 0


def test_eda_profile_is_json_serializable_and_tracks_missing_values():
    df = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0, 4.0, 5.0, 6.0],
            "feature_b": [2.0, 2.0, 3.0, 3.0, 8.0, 9.0],
            "constant": [1.0] * 6,
            "defect_label": [0, 0, 0, 1, 1, 1],
        }
    )
    missing = {
        "raw_missing_total": 0,
        "after_numeric_conversion_total": 0,
        "imputed_total": 0,
        "processed_missing_total": 0,
    }

    profile = build_eda_profile(df, "defect_label", missing, top_n=2)

    assert profile["missing_values"]["imputed_total"] == 0
    assert profile["constant_features"]["features"] == ["constant"]
    assert profile["feature_selection"]["selected_features"] == [
        "feature_a",
        "feature_b",
    ]
    json.dumps(profile)


def test_write_eda_plots_creates_all_outputs(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "feature_a": range(20),
            "feature_b": [value * 2 for value in range(20)],
            "defect_label": [0] * 10 + [1] * 10,
        }
    )
    missing = {
        "raw_missing_total": 0,
        "after_numeric_conversion_total": 0,
        "imputed_total": 0,
        "processed_missing_total": 0,
    }
    profile = build_eda_profile(df, "defect_label", missing, top_n=2)
    plot_paths = {
        "class_distribution": "class.png",
        "feature_distributions": "hist.png",
        "feature_boxplots": "box.png",
        "correlation_heatmap": "corr.png",
    }
    monkeypatch.setattr("src.data.prepare_data.ROOT", tmp_path)
    monkeypatch.setattr("src.data.prepare_data.EDA_PLOT_PATHS", plot_paths)

    write_eda_plots(df, "defect_label", profile)

    assert all((tmp_path / path).is_file() for path in plot_paths.values())
