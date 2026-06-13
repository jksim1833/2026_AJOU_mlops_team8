import pandas as pd

from src.data.prepare_data import remove_exact_duplicates, split_dataset, split_overlap_counts


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
