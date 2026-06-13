from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "params.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "processed" / f"{name}.csv")


def split_xy(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[target_column]), df[target_column]


def evaluate(model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }


def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["normal", "defect"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_roc_curve(y_true: pd.Series, y_score: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_feature_importance(model: RandomForestClassifier, feature_names: list[str], json_path: Path, plot_path: Path) -> list[dict[str, float]]:
    importances = model.feature_importances_
    records = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in zip(feature_names, importances)
    ]
    records.sort(key=lambda item: item["importance"], reverse=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    top = records[:12]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([item["feature"] for item in reversed(top)], [item["importance"] for item in reversed(top)])
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return records


def save_sample_inputs(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    path: Path,
    normal_request_path: Path,
    defect_request_path: Path,
) -> None:
    samples: dict[str, dict[str, float]] = {}
    for label, name in [(0, "normal_sample"), (1, "defect_sample")]:
        predictions = pd.Series(model.predict(X), index=X.index)
        correct_indices = y[(y == label) & (predictions == label)].index
        idx = correct_indices[0] if len(correct_indices) else y[y == label].index[0]
        samples[name] = {col: float(X.loc[idx, col]) for col in X.columns}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    normal_request_path.write_text(json.dumps({"features": samples["normal_sample"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    defect_request_path.write_text(json.dumps({"features": samples["defect_sample"]}, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    cfg = load_config()
    target_column = cfg["data"]["target_column"]
    train_df = load_split("train")
    valid_df = load_split("valid")
    test_df = load_split("test")
    X_train, y_train = split_xy(train_df, target_column)
    X_valid, y_valid = split_xy(valid_df, target_column)
    X_test, y_test = split_xy(test_df, target_column)

    model_cfg = cfg["model"]
    model = RandomForestClassifier(
        n_estimators=int(model_cfg["n_estimators"]),
        max_depth=int(model_cfg["max_depth"]),
        min_samples_leaf=int(model_cfg["min_samples_leaf"]),
        class_weight=model_cfg["class_weight"],
        random_state=int(model_cfg["random_state"]),
        n_jobs=-1,
    )

    tracking_uri = cfg["tracking"]["tracking_uri"]
    if "://" in tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        tracking_dir = ROOT / tracking_uri
        tracking_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment(cfg["tracking"]["experiment_name"])

    artifact_paths = cfg["artifacts"]
    model_path = ROOT / artifact_paths["rf_model_path"]
    metadata_path = ROOT / artifact_paths["rf_metadata_path"]
    metrics_path = ROOT / artifact_paths["rf_metrics_path"]
    feature_importance_path = ROOT / artifact_paths["rf_feature_importance_path"]
    confusion_matrix_path = ROOT / artifact_paths["rf_confusion_matrix_path"]
    roc_curve_path = ROOT / artifact_paths["rf_roc_curve_path"]
    feature_importance_plot_path = ROOT / artifact_paths["rf_feature_importance_plot_path"]

    with mlflow.start_run(run_name="rf_binary_baseline") as run:
        model.fit(X_train, y_train)
        valid_metrics = evaluate(model, X_valid, y_valid)
        test_metrics = evaluate(model, X_test, y_test)
        y_test_pred = model.predict(X_test)
        y_test_score = model.predict_proba(X_test)[:, 1]

        save_confusion_matrix(y_test, y_test_pred, confusion_matrix_path)
        save_roc_curve(y_test, y_test_score, roc_curve_path)
        feature_importance = save_feature_importance(
            model,
            list(X_train.columns),
            feature_importance_path,
            feature_importance_plot_path,
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

        metrics = {
            "validation": valid_metrics,
            "test": test_metrics,
            "classification_report": classification_report(y_test, y_test_pred, target_names=["normal", "defect"], output_dict=True),
        }
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        metadata = {
            "project": cfg["project"]["name"],
            "model_version": "rf_binary_baseline_v2",
            "data_version": cfg["project"]["data_version"],
            "artifact_role": "baseline_only",
            "target": {"0": "normal", "1": "defect"},
            "features": list(X_train.columns),
            "run_id": run.info.run_id,
            "primary_metric": "validation_f1",
            "validation_metrics": valid_metrics,
            "test_metrics": test_metrics,
            "top_features": feature_importance[:8],
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        mlflow.log_params(model_cfg)
        mlflow.log_param("data_version", cfg["project"]["data_version"])
        mlflow.log_param("target_column", target_column)
        for key, value in valid_metrics.items():
            mlflow.log_metric(f"valid_{key}", value)
        for key, value in test_metrics.items():
            mlflow.log_metric(f"test_{key}", value)
        mlflow.set_tag("stage", "baseline")
        mlflow.set_tag("serving_candidate", "false")
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(metadata_path), artifact_path="model")
        mlflow.log_artifact(str(metrics_path), artifact_path="reports")
        mlflow.log_artifact(str(confusion_matrix_path), artifact_path="plots")
        mlflow.log_artifact(str(roc_curve_path), artifact_path="plots")
        mlflow.log_artifact(str(feature_importance_plot_path), artifact_path="plots")
        mlflow.log_artifact(str(feature_importance_path), artifact_path="reports")

    print(json.dumps({"validation": valid_metrics, "test": test_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
