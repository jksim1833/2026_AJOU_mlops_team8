from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "params.yaml"


DEFECT_KEYWORDS = [
    "Short_Shot",
    "Bubble",
    "Exfoliation",
    "Blow_Hole",
    "Stain",
    "Dent",
    "Deformation",
    "Contamination",
    "Impurity",
    "Crack",
    "Scratch",
    "Buring_Mark",
    "Burning_Mark",
    "Inclusions",
]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]
    return df


def find_defect_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        normalized = col.lower()
        if any(keyword.lower() in normalized for keyword in DEFECT_KEYWORDS):
            cols.append(col)
    return cols


def build_binary_dataset(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, list[str]]:
    df = clean_columns(df)
    defect_columns = find_defect_columns(df)
    if not defect_columns:
        raise ValueError("No defect columns found. Check raw data schema.")

    defect_frame = df[defect_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    df[target_column] = (defect_frame.sum(axis=1) > 0).astype(int)

    feature_df = df.drop(columns=defect_columns)
    numeric_feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    numeric_feature_df[target_column] = df[target_column]
    numeric_feature_df = numeric_feature_df.dropna(axis=1, how="all")
    numeric_feature_df = numeric_feature_df.fillna(numeric_feature_df.median(numeric_only=True))
    return numeric_feature_df, defect_columns


def remove_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    deduplicated = df.drop_duplicates(keep="first").reset_index(drop=True)
    return deduplicated, int(len(df) - len(deduplicated))


def split_dataset(df: pd.DataFrame, target_column: str, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    random_state = int(cfg["data"]["random_state"])
    test_size = float(cfg["data"]["test_size"])
    validation_size = float(cfg["data"]["validation_size"])

    train_valid, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_column],
    )
    valid_ratio_within_train_valid = validation_size / (1.0 - test_size)
    train, valid = train_test_split(
        train_valid,
        test_size=valid_ratio_within_train_valid,
        random_state=random_state,
        stratify=train_valid[target_column],
    )
    return {"train": train, "valid": valid, "test": test}


def split_overlap_counts(splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    row_hashes = {
        name: set(pd.util.hash_pandas_object(split, index=False).astype(str))
        for name, split in splits.items()
    }
    return {
        "train_valid": len(row_hashes["train"] & row_hashes["valid"]),
        "train_test": len(row_hashes["train"] & row_hashes["test"]),
        "valid_test": len(row_hashes["valid"] & row_hashes["test"]),
    }


def write_split_files(splits: dict[str, pd.DataFrame], processed_dir: Path) -> None:
    for name, split_df in splits.items():
        split_df.to_csv(processed_dir / f"{name}.csv", index=False)


def main() -> None:
    cfg = load_config()
    raw_path = ROOT / cfg["data"]["raw_path"]
    processed_path = ROOT / cfg["data"]["processed_path"]
    processed_dir = processed_path.parent
    target_column = cfg["data"]["target_column"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(raw_path)
    processed_with_duplicates, defect_columns = build_binary_dataset(raw_df, target_column)
    processed_df, duplicates_removed = remove_exact_duplicates(processed_with_duplicates)
    processed_df.to_csv(processed_path, index=False)
    splits = split_dataset(processed_df, target_column, cfg)
    overlap_counts = split_overlap_counts(splits)
    if any(overlap_counts.values()):
        raise RuntimeError(f"Exact row overlap detected between splits: {overlap_counts}")
    write_split_files(splits, processed_dir)

    report = {
        "source": str(raw_path.relative_to(ROOT)),
        "processed": str(processed_path.relative_to(ROOT)),
        "data_version": cfg["project"]["data_version"],
        "source_rows": int(processed_with_duplicates.shape[0]),
        "rows": int(processed_df.shape[0]),
        "duplicates_removed": duplicates_removed,
        "deduplication_rule": "drop fully identical processed rows before splitting",
        "columns": int(processed_df.shape[1]),
        "target_column": target_column,
        "class_counts": processed_df[target_column].value_counts().sort_index().to_dict(),
        "defect_columns_removed": defect_columns,
        "split_counts": {
            name: split[target_column].value_counts().sort_index().to_dict()
            for name, split in splits.items()
        },
        "split_row_counts": {name: int(len(split)) for name, split in splits.items()},
        "exact_row_overlap": overlap_counts,
    }
    report_path = ROOT / "artifacts" / "reports" / "data_profile.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
