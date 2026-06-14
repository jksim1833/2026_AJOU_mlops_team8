from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "params.yaml"
EDA_FEATURE_COUNT = 8
EDA_PLOT_PATHS = {
    "class_distribution": "artifacts/plots/eda_class_distribution.png",
    "feature_distributions": "artifacts/plots/eda_feature_distributions.png",
    "feature_boxplots": "artifacts/plots/eda_feature_boxplots.png",
    "correlation_heatmap": "artifacts/plots/eda_correlation_heatmap.png",
}


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


def missing_value_profile(
    raw_df: pd.DataFrame,
    defect_columns: list[str],
    target_column: str,
) -> dict[str, Any]:
    cleaned = clean_columns(raw_df)
    feature_df = cleaned.drop(columns=defect_columns)
    raw_features = feature_df.drop(columns=[target_column], errors="ignore")
    numeric_features = raw_features.apply(pd.to_numeric, errors="coerce")
    numeric_features = numeric_features.dropna(axis=1, how="all")

    raw_counts = raw_features[numeric_features.columns].isna().sum()
    numeric_counts = numeric_features.isna().sum()
    return {
        "raw_missing_total": int(raw_counts.sum()),
        "raw_missing_by_feature": {
            feature: int(count)
            for feature, count in raw_counts.items()
            if count > 0
        },
        "after_numeric_conversion_total": int(numeric_counts.sum()),
        "after_numeric_conversion_by_feature": {
            feature: int(count)
            for feature, count in numeric_counts.items()
            if count > 0
        },
        "imputation_strategy": "median per numeric feature",
        "imputed_total": int(numeric_counts.sum()),
        "processed_missing_total": 0,
    }


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


def select_eda_features(
    df: pd.DataFrame,
    target_column: str,
    top_n: int = EDA_FEATURE_COUNT,
) -> tuple[list[str], dict[str, float], list[str]]:
    feature_columns = [
        column
        for column in df.select_dtypes(include="number").columns
        if column != target_column
    ]
    constant_features = sorted(
        column for column in feature_columns if df[column].nunique(dropna=False) <= 1
    )
    candidate_features = [
        column for column in feature_columns if column not in constant_features
    ]

    scores: dict[str, float] = {}
    for feature in candidate_features:
        standard_deviation = float(df[feature].std(ddof=0))
        normal_mean = float(df.loc[df[target_column] == 0, feature].mean())
        defect_mean = float(df.loc[df[target_column] == 1, feature].mean())
        scores[feature] = abs(defect_mean - normal_mean) / standard_deviation

    ranked = sorted(candidate_features, key=lambda feature: (-scores[feature], feature))
    selected = ranked[: min(top_n, len(ranked))]
    return selected, scores, constant_features


def calculate_outlier_summary(
    df: pd.DataFrame,
    features: list[str],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for feature in features:
        q1 = float(df[feature].quantile(0.25))
        q3 = float(df[feature].quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask = (df[feature] < lower_bound) | (df[feature] > upper_bound)
        count = int(mask.sum())
        summary.append(
            {
                "feature": feature,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "count": count,
                "ratio": count / len(df),
            }
        )
    return sorted(summary, key=lambda item: (-item["count"], item["feature"]))


def build_eda_profile(
    df: pd.DataFrame,
    target_column: str,
    missing_values: dict[str, Any],
    top_n: int = EDA_FEATURE_COUNT,
) -> dict[str, Any]:
    selected, scores, constant_features = select_eda_features(
        df, target_column, top_n
    )
    numeric_features = [
        column
        for column in df.select_dtypes(include="number").columns
        if column != target_column and column not in constant_features
    ]
    class_counts = df[target_column].value_counts().sort_index()
    descriptive = df[selected].describe().T

    class_comparison: dict[str, dict[str, float]] = {}
    for feature in selected:
        class_comparison[feature] = {
            "normal_mean": float(df.loc[df[target_column] == 0, feature].mean()),
            "normal_median": float(df.loc[df[target_column] == 0, feature].median()),
            "defect_mean": float(df.loc[df[target_column] == 1, feature].mean()),
            "defect_median": float(df.loc[df[target_column] == 1, feature].median()),
            "standardized_mean_difference": float(scores[feature]),
            "target_correlation": float(df[feature].corr(df[target_column])),
        }

    return {
        "feature_selection": {
            "method": "absolute class mean difference divided by population standard deviation",
            "tie_breaker": "feature name ascending",
            "requested_count": top_n,
            "candidate_count": len(numeric_features),
            "selected_features": selected,
            "scores": {feature: float(scores[feature]) for feature in selected},
        },
        "class_distribution": {
            str(int(label)): {
                "count": int(count),
                "ratio": float(count / len(df)),
            }
            for label, count in class_counts.items()
        },
        "missing_values": missing_values,
        "constant_features": {
            "count": len(constant_features),
            "features": constant_features,
        },
        "descriptive_statistics": {
            feature: {
                key: float(value)
                for key, value in descriptive.loc[feature].items()
            }
            for feature in selected
        },
        "class_comparison": class_comparison,
        "outliers": {
            "method": "IQR 1.5 rule",
            "policy": "report_only",
            "features": calculate_outlier_summary(df, numeric_features),
        },
        "correlation_matrix": {
            row: {
                column: float(value)
                for column, value in values.items()
            }
            for row, values in (
                df[selected + [target_column]].corr().to_dict(orient="index").items()
            )
        },
        "plot_paths": EDA_PLOT_PATHS,
    }


def plot_class_distribution(
    df: pd.DataFrame,
    target_column: str,
    output_path: Path,
) -> None:
    counts = df[target_column].value_counts().reindex([0, 1], fill_value=0)
    labels = ["Normal (0)", "Defect (1)"]
    colors = ["#4C78A8", "#E45756"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts.values, color=colors)
    ax.set_title("Class Distribution")
    ax.set_ylabel("Rows")
    ax.set_ylim(0, max(counts.values) * 1.15)
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\n({count / len(df):.1%})",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_feature_distributions(
    df: pd.DataFrame,
    target_column: str,
    features: list[str],
    output_path: Path,
) -> None:
    rows = int(np.ceil(len(features) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(13, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, feature in zip(axes, features):
        ax.hist(
            df.loc[df[target_column] == 0, feature],
            bins=25,
            alpha=0.6,
            density=True,
            label="Normal",
            color="#4C78A8",
        )
        ax.hist(
            df.loc[df[target_column] == 1, feature],
            bins=25,
            alpha=0.6,
            density=True,
            label="Defect",
            color="#E45756",
        )
        ax.set_title(feature)
        ax.set_ylabel("Density")
        ax.legend()
    for ax in axes[len(features):]:
        ax.axis("off")
    fig.suptitle("Selected Feature Distributions by Class", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_feature_boxplots(
    df: pd.DataFrame,
    target_column: str,
    features: list[str],
    output_path: Path,
) -> None:
    rows = int(np.ceil(len(features) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(13, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, feature in zip(axes, features):
        boxplot = ax.boxplot(
            [
                df.loc[df[target_column] == 0, feature],
                df.loc[df[target_column] == 1, feature],
            ],
            patch_artist=True,
            showfliers=True,
        )
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Normal", "Defect"])
        for patch, color in zip(boxplot["boxes"], ["#4C78A8", "#E45756"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax.set_title(feature)
    for ax in axes[len(features):]:
        ax.axis("off")
    fig.suptitle("Selected Feature Boxplots by Class", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    target_column: str,
    features: list[str],
    output_path: Path,
) -> None:
    columns = features + [target_column]
    correlation = df[columns].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)), labels=columns, rotation=45, ha="right")
    ax.set_yticks(range(len(columns)), labels=columns)
    for row in range(len(columns)):
        for column in range(len(columns)):
            value = correlation.iloc[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.55 else "black",
                fontsize=8,
            )
    ax.set_title("Correlation Heatmap")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_eda_plots(
    df: pd.DataFrame,
    target_column: str,
    eda_profile: dict[str, Any],
) -> None:
    selected = eda_profile["feature_selection"]["selected_features"]
    paths = {
        name: ROOT / relative_path
        for name, relative_path in EDA_PLOT_PATHS.items()
    }
    paths["class_distribution"].parent.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(df, target_column, paths["class_distribution"])
    plot_feature_distributions(
        df, target_column, selected, paths["feature_distributions"]
    )
    plot_feature_boxplots(df, target_column, selected, paths["feature_boxplots"])
    plot_correlation_heatmap(
        df, target_column, selected, paths["correlation_heatmap"]
    )


def main() -> None:
    cfg = load_config()
    raw_path = ROOT / cfg["data"]["raw_path"]
    processed_path = ROOT / cfg["data"]["processed_path"]
    processed_dir = processed_path.parent
    target_column = cfg["data"]["target_column"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(raw_path)
    processed_with_duplicates, defect_columns = build_binary_dataset(raw_df, target_column)
    missing_values = missing_value_profile(raw_df, defect_columns, target_column)
    processed_df, duplicates_removed = remove_exact_duplicates(processed_with_duplicates)
    processed_df.to_csv(processed_path, index=False)
    splits = split_dataset(processed_df, target_column, cfg)
    overlap_counts = split_overlap_counts(splits)
    if any(overlap_counts.values()):
        raise RuntimeError(f"Exact row overlap detected between splits: {overlap_counts}")
    write_split_files(splits, processed_dir)
    eda_profile = build_eda_profile(processed_df, target_column, missing_values)
    write_eda_plots(processed_df, target_column, eda_profile)

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
        "eda": eda_profile,
    }
    report_path = ROOT / "artifacts" / "reports" / "data_profile.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
