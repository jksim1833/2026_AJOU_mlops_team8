from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "params.yaml"
REPORT_DIR = ROOT / "artifacts" / "reports"
PLOT_DIR = ROOT / "artifacts" / "plots"
MODEL_DIR = ROOT / "artifacts" / "models"
DOC_REPORT_PATH = ROOT / "docs" / "baseline_xai_report.md"


@dataclass
class ModelSpec:
    name: str
    estimator: BaseEstimator
    params: dict[str, Any]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_split(name: str) -> pd.DataFrame:
    path = ROOT / "data" / "processed" / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.relative_to(ROOT)}. Put the raw CSV in "
            "data/raw/DieCasting_Quality_Raw_Data_product1.csv and run "
            "`python -m src.data.prepare_data` first."
        )
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[target_column]), df[target_column].astype(int)


def configure_mlflow(cfg: dict[str, Any]) -> None:
    tracking_uri = cfg["tracking"]["tracking_uri"]
    if "://" in tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        tracking_dir = ROOT / tracking_uri
        tracking_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment(cfg["tracking"]["experiment_name"])


def optional_import(name: str) -> Any | None:
    try:
        module = __import__(name)
    except ImportError:
        return None
    return module


def build_model_specs(cfg: dict[str, Any], y_train: pd.Series) -> tuple[list[ModelSpec], list[str]]:
    random_state = int(cfg["data"]["random_state"])
    model_cfg = cfg["model"]
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count if pos_count else 1.0

    specs = [
        ModelSpec(
            name="logistic_regression",
            estimator=Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            params={
                "model_type": "LogisticRegression",
                "class_weight": "balanced",
                "max_iter": 2000,
            },
        ),
        ModelSpec(
            name="decision_tree",
            estimator=DecisionTreeClassifier(
                max_depth=6,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=random_state,
            ),
            params={
                "model_type": "DecisionTreeClassifier",
                "max_depth": 6,
                "min_samples_leaf": 5,
                "class_weight": "balanced",
            },
        ),
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                n_estimators=int(model_cfg["n_estimators"]),
                max_depth=int(model_cfg["max_depth"]),
                min_samples_leaf=int(model_cfg["min_samples_leaf"]),
                class_weight=model_cfg["class_weight"],
                random_state=int(model_cfg["random_state"]),
                n_jobs=-1,
            ),
            params={
                "model_type": "RandomForestClassifier",
                "n_estimators": int(model_cfg["n_estimators"]),
                "max_depth": int(model_cfg["max_depth"]),
                "min_samples_leaf": int(model_cfg["min_samples_leaf"]),
                "class_weight": model_cfg["class_weight"],
            },
        ),
    ]

    skipped: list[str] = []
    xgboost = optional_import("xgboost")
    if xgboost is None:
        skipped.append("xgboost_baseline skipped because xgboost is not installed.")
    else:
        specs.append(
            ModelSpec(
                name="xgboost",
                estimator=xgboost.XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                    random_state=random_state,
                    n_jobs=-1,
                ),
                params={
                    "model_type": "XGBClassifier",
                    "n_estimators": 300,
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "scale_pos_weight": round(scale_pos_weight, 6),
                },
            )
        )

    lightgbm = optional_import("lightgbm")
    if lightgbm is None:
        skipped.append("lightgbm_baseline skipped because lightgbm is not installed.")
    else:
        specs.append(
            ModelSpec(
                name="lightgbm",
                estimator=lightgbm.LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=-1,
                ),
                params={
                    "model_type": "LGBMClassifier",
                    "n_estimators": 300,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "class_weight": "balanced",
                },
            )
        )

    return specs, skipped


def positive_class_scores(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(X))
        return 1.0 / (1.0 + np.exp(-decision))
    raise TypeError(f"{type(model).__name__} cannot produce probability scores.")


def evaluate(model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    pred = np.asarray(model.predict(X)).astype(int)
    score = positive_class_scores(model, X)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    row = {"model": record["model"]}
    for split in ["validation", "test"]:
        for metric, value in record[split].items():
            row[f"{split}_{metric}"] = value
    return row


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(format_value(row.get(col, "")) for col in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def save_metric_plot(df: pd.DataFrame, path: Path) -> None:
    metrics = ["validation_f1", "validation_roc_auc", "validation_recall", "validation_precision"]
    labels = ["F1", "ROC-AUC", "Recall", "Precision"]
    x = np.arange(len(df))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, metric in enumerate(metrics):
        ax.bar(x + (idx - 1.5) * width, df[metric], width=width, label=labels[idx])
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Validation Baseline Metrics")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_confusion_grid(records: list[dict[str, Any]], path: Path) -> None:
    n = len(records)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.7 * rows))
    axes_array = np.atleast_1d(axes).ravel()

    for ax, record in zip(axes_array, records):
        metrics = record["test"]
        cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
        image = ax.imshow(cm, cmap="Blues")
        ax.set_title(record["model"])
        ax.set_xticks([0, 1], labels=["normal", "defect"])
        ax.set_yticks([0, 1], labels=["normal", "defect"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for (row, col), value in np.ndenumerate(cm):
            ax.text(col, row, int(value), ha="center", va="center", color="black")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes_array[n:]:
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_candidate_artifact(
    model: BaseEstimator,
    record: dict[str, Any],
    feature_names: list[str],
    cfg: dict[str, Any],
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    candidate_path = MODEL_DIR / "baseline_candidate.joblib"
    metadata_path = MODEL_DIR / "baseline_candidate_metadata.json"
    joblib.dump(model, candidate_path)

    metadata = {
        "project": cfg["project"]["name"],
        "owner": "team8_modeling",
        "artifact_role": "baseline_candidate_for_tuning",
        "selection_rule": "highest validation F1, validation ROC-AUC as tie-breaker",
        "selected_model": record["model"],
        "features": feature_names,
        "validation_metrics": record["validation"],
        "test_metrics": record["test"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_class_one_values(raw_values: Any, n_features: int) -> np.ndarray:
    values = raw_values.values if hasattr(raw_values, "values") else raw_values
    if isinstance(values, list):
        return np.asarray(values[1])
    values_array = np.asarray(values)
    if values_array.ndim == 3:
        if values_array.shape[2] == 2:
            return values_array[:, :, 1]
        if values_array.shape[0] == 2:
            return values_array[1, :, :]
    if values_array.ndim == 2 and values_array.shape[1] == n_features:
        return values_array
    if values_array.ndim == 2 and values_array.shape[0] == n_features:
        return values_array.T
    raise ValueError(f"Unsupported SHAP value shape: {values_array.shape}")


def extract_expected_value(expected_value: Any) -> float:
    expected = np.asarray(expected_value)
    if expected.ndim == 0:
        return float(expected)
    flat = expected.ravel()
    if flat.size >= 2:
        return float(flat[1])
    return float(flat[0])


def compute_shap_values(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    X_explain: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    import shap

    if isinstance(model, Pipeline):
        estimator = model.steps[-1][1]
        transformer = model[:-1]
        X_train_transformed = transformer.transform(X_train)
        X_explain_transformed = transformer.transform(X_explain)
        explainer = shap.LinearExplainer(estimator, X_train_transformed)
        raw_values = explainer.shap_values(X_explain_transformed)
        return extract_class_one_values(raw_values, X_train.shape[1]), extract_expected_value(
            explainer.expected_value
        )

    explainer = shap.TreeExplainer(model)
    try:
        raw_values = explainer.shap_values(X_explain, check_additivity=False)
    except TypeError:
        raw_values = explainer.shap_values(X_explain)
    return extract_class_one_values(raw_values, X_train.shape[1]), extract_expected_value(
        explainer.expected_value
    )


def save_shap_outputs(
    model: BaseEstimator,
    best_record: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sample_size: int,
) -> dict[str, Any]:
    shap_module = optional_import("shap")
    if shap_module is None:
        message = "SHAP skipped because shap is not installed."
        (REPORT_DIR / "xai_feature_interpretation.md").write_text(message + "\n", encoding="utf-8")
        return {"status": "skipped", "message": message}

    rng = int(42)
    X_background = X_train.sample(n=min(300, len(X_train)), random_state=rng)
    X_explain = X_test.sample(n=min(sample_size, len(X_test)), random_state=rng)
    y_explain = y_test.loc[X_explain.index]
    shap_values, expected_value = compute_shap_values(model, X_background, X_explain)
    feature_names = list(X_train.columns)

    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    rows: list[dict[str, Any]] = []
    for idx in np.argsort(mean_abs)[::-1]:
        feature = feature_names[idx]
        feature_values = X_explain[feature].to_numpy()
        shap_column = shap_values[:, idx]
        if np.std(feature_values) == 0 or np.std(shap_column) == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(feature_values, shap_column)[0, 1])
        if corr > 0.2:
            direction = "higher values tend to push the prediction toward defect"
        elif corr < -0.2:
            direction = "higher values tend to push the prediction toward normal"
        else:
            direction = "mixed or nonlinear effect"
        rows.append(
            {
                "rank": len(rows) + 1,
                "feature": feature,
                "mean_abs_shap": float(mean_abs[idx]),
                "mean_shap": float(mean_signed[idx]),
                "effect_direction": direction,
                "correlation_feature_value_vs_shap": corr,
            }
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    interpretation_df = pd.DataFrame(rows)
    interpretation_df.to_csv(REPORT_DIR / "xai_feature_interpretation.csv", index=False)
    (REPORT_DIR / "xai_feature_interpretation.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    top_rows = rows[:12]
    interpretation_md = [
        "# SHAP XAI Feature Interpretation",
        "",
        "Positive SHAP values push the prediction toward the defect class. "
        "Direction is estimated from the correlation between feature value and SHAP value "
        "on the explained test sample.",
        "",
        markdown_table(
            top_rows,
            [
                "rank",
                "feature",
                "mean_abs_shap",
                "mean_shap",
                "effect_direction",
            ],
        ),
        "",
    ]
    (REPORT_DIR / "xai_feature_interpretation.md").write_text(
        "\n".join(interpretation_md),
        encoding="utf-8",
    )

    import shap

    bar_path = PLOT_DIR / "shap_summary_bar.png"
    beeswarm_path = PLOT_DIR / "shap_beeswarm.png"
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        plot_type="bar",
        max_display=12,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(bar_path, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        max_display=12,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=160, bbox_inches="tight")
    plt.close()

    proba = positive_class_scores(model, X_explain)
    predictions = np.asarray(model.predict(X_explain)).astype(int)
    defect_positions = np.where(predictions == 1)[0]
    if len(defect_positions):
        local_pos = defect_positions[np.argmax(proba[defect_positions])]
    else:
        local_pos = int(np.argmax(proba))

    local_rows = []
    for idx in np.argsort(np.abs(shap_values[local_pos]))[::-1][:10]:
        local_rows.append(
            {
                "feature": feature_names[idx],
                "input_value": float(X_explain.iloc[local_pos, idx]),
                "shap_value": float(shap_values[local_pos, idx]),
            }
        )
    local_payload = {
        "model": best_record["model"],
        "sample_index": int(X_explain.index[local_pos]),
        "true_label": int(y_explain.iloc[local_pos]),
        "predicted_label": int(predictions[local_pos]),
        "defect_probability": float(proba[local_pos]),
        "top_local_features": local_rows,
    }
    (REPORT_DIR / "shap_local_explanation.json").write_text(
        json.dumps(local_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    local_md = [
        "# SHAP Local Explanation",
        "",
        f"- Model: {best_record['model']}",
        f"- Test sample index: {local_payload['sample_index']}",
        f"- True label: {local_payload['true_label']}",
        f"- Predicted label: {local_payload['predicted_label']}",
        f"- Defect probability: {local_payload['defect_probability']:.3f}",
        "",
        markdown_table(local_rows, ["feature", "input_value", "shap_value"]),
        "",
    ]
    (REPORT_DIR / "shap_local_explanation.md").write_text("\n".join(local_md), encoding="utf-8")

    waterfall_path = PLOT_DIR / "shap_waterfall_defect_sample.png"
    try:
        explanation = shap.Explanation(
            values=shap_values[local_pos],
            base_values=expected_value,
            data=X_explain.iloc[local_pos].to_numpy(),
            feature_names=feature_names,
        )
        shap.plots.waterfall(explanation, max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(waterfall_path, dpi=160, bbox_inches="tight")
        plt.close()
    except Exception as exc:  # SHAP plotting differs across versions.
        local_payload["waterfall_warning"] = str(exc)

    return {
        "status": "ok",
        "top_features": top_rows,
        "bar_plot": str(bar_path.relative_to(ROOT)),
        "beeswarm_plot": str(beeswarm_path.relative_to(ROOT)),
        "waterfall_plot": str(waterfall_path.relative_to(ROOT)),
    }


def write_reports(
    records: list[dict[str, Any]],
    skipped: list[str],
    best_record: dict[str, Any],
    shap_status: dict[str, Any],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    flat_records = [flatten_record(record) for record in records]
    df = pd.DataFrame(flat_records).sort_values(
        by=["validation_f1", "validation_roc_auc"],
        ascending=[False, False],
    )
    df.to_csv(REPORT_DIR / "baseline_comparison.csv", index=False)
    (REPORT_DIR / "baseline_comparison.json").write_text(
        json.dumps(
            {
                "selection_rule": "highest validation F1, validation ROC-AUC as tie-breaker",
                "skipped": skipped,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    metric_columns = [
        "model",
        "validation_f1",
        "validation_roc_auc",
        "validation_recall",
        "validation_precision",
        "test_f1",
        "test_roc_auc",
        "test_recall",
        "test_precision",
    ]
    table_rows = df[metric_columns].to_dict(orient="records")
    metric_table = [
        "# Baseline Metric Table",
        "",
        "Models are ranked by validation F1, with validation ROC-AUC as the tie-breaker.",
        "",
        markdown_table(table_rows, metric_columns),
        "",
    ]
    if skipped:
        metric_table.extend(["## Skipped Models", "", *[f"- {item}" for item in skipped], ""])
    (REPORT_DIR / "baseline_metric_table.md").write_text("\n".join(metric_table), encoding="utf-8")

    save_metric_plot(df, PLOT_DIR / "baseline_validation_metrics.png")
    save_confusion_grid(records, PLOT_DIR / "baseline_confusion_matrices.png")

    best_row = flatten_record(best_record)
    roc_leader = df.sort_values("validation_roc_auc", ascending=False).iloc[0].to_dict()
    recommendation_rows = [best_row]
    if roc_leader["model"] != best_row["model"]:
        recommendation_rows.append(roc_leader)

    handoff = [
        "# Candidate Handoff Note",
        "",
        "Owner: Team 8 modeling task",
        "",
        f"Primary recommendation: `{best_row['model']}` because it has the best validation F1 "
        f"({best_row['validation_f1']:.3f}) under the shared split.",
        "",
        "Recommended tuning candidates:",
        "",
        markdown_table(
            recommendation_rows,
            ["model", "validation_f1", "validation_roc_auc", "validation_recall", "validation_precision"],
        ),
        "",
        "Tuning handoff:",
        "",
        "- Tune the primary candidate first with threshold adjustment for defect recall/F1.",
        "- Keep ROC-AUC leader as backup if it differs from the F1 leader.",
        "- Use the SHAP summary and local explanation artifacts to explain model behavior in the report.",
        "",
    ]
    if skipped:
        handoff.extend(["Skipped optional baselines:", "", *[f"- {item}" for item in skipped], ""])
    (REPORT_DIR / "candidate_handoff_note.md").write_text("\n".join(handoff), encoding="utf-8")

    doc = [
        "# Baseline and SHAP XAI Report",
        "",
        "## 1. 보고서 목적",
        "",
        "본 문서는 다이캐스팅 정상/불량 이진 분류 프로젝트에서 baseline 모델링과 "
        "SHAP 기반 XAI 구현 내용을 정리한다.",
        "",
        "Baseline은 여러 후보 모델을 같은 train/validation/test split과 같은 "
        "metric으로 비교하여 후속 tuning 후보를 선정하는 단계이다. XAI는 "
        "선정된 모델이 normal/defect를 판단할 때 어떤 feature를 근거로 삼았는지 "
        "설명하는 단계이다.",
        "",
        "## 2. Baseline 구현 내용",
        "",
        "비교 후보 모델:",
        "",
        "- Logistic Regression",
        "- Decision Tree",
        "- RandomForest",
        "- XGBoost",
        "- LightGBM",
        "",
        "평가 metric은 accuracy, precision, recall, F1, ROC-AUC를 함께 사용한다. "
        "정상 class가 더 많은 class imbalance 문제이므로 validation F1을 우선 "
        "기준으로 하고 validation ROC-AUC를 tie-breaker로 사용한다.",
        "",
        "## 3. Baseline 비교 결과",
        "",
        markdown_table(table_rows, metric_columns),
        "",
        "## 4. 선택된 후보 모델",
        "",
        f"`{best_row['model']}` 모델은 shared split에서 validation F1 "
        f"({best_row['validation_f1']:.3f}) 기준 가장 높은 성능을 보여 첫 번째 "
        "tuning 후보로 선택되었다. 선택된 모델은 "
        "`artifacts/models/baseline_candidate.joblib`로 저장된다.",
        "",
        "## 5. SHAP/XAI 구현 내용",
        "",
        "SHAP은 모든 후보 모델마다 실행하지 않는다. 먼저 baseline 후보 모델들을 "
        "비교한 뒤, 선택된 best candidate model 1개에 대해서만 자동으로 SHAP "
        "분석을 수행한다.",
        "",
        "생성되는 설명:",
        "",
        "- Global explanation: 전체 test sample 기준으로 중요한 feature를 보여준다.",
        "- Local explanation: 특정 sample의 예측에 대한 feature별 기여도를 보여준다.",
        "- Positive SHAP value는 예측을 defect 방향으로 이동시킨다.",
        "- Negative SHAP value는 예측을 normal 방향으로 이동시킨다.",
        "",
        "## 6. 주요 산출물",
        "",
        "Baseline 산출물:",
        "",
        "- `artifacts/reports/baseline_metric_table.md`",
        "- `artifacts/reports/baseline_comparison.csv`",
        "- `artifacts/reports/candidate_handoff_note.md`",
        "- `artifacts/models/baseline_candidate.joblib`",
        "- `artifacts/models/baseline_candidate_metadata.json`",
        "",
        "SHAP/XAI 산출물:",
        "",
        f"- SHAP status: `{shap_status.get('status')}`",
        "- `artifacts/reports/xai_feature_interpretation.md`",
        "- `artifacts/reports/shap_local_explanation.md`",
        "- `artifacts/plots/shap_summary_bar.png`",
        "- `artifacts/plots/shap_beeswarm.png`",
        "- `artifacts/plots/shap_waterfall_defect_sample.png`",
        "",
    ]
    if skipped:
        doc.extend(["## Notes", "", "설치되지 않은 optional baseline:", "", *[f"- {item}" for item in skipped], ""])
    DOC_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_REPORT_PATH.write_text("\n".join(doc), encoding="utf-8")


def train_and_compare(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config()
    target_column = cfg["data"]["target_column"]
    train_df = load_split("train")
    valid_df = load_split("valid")
    test_df = load_split("test")
    X_train, y_train = split_xy(train_df, target_column)
    X_valid, y_valid = split_xy(valid_df, target_column)
    X_test, y_test = split_xy(test_df, target_column)
    feature_names = list(X_train.columns)

    specs, skipped = build_model_specs(cfg, y_train)
    if not args.skip_mlflow:
        configure_mlflow(cfg)

    records: list[dict[str, Any]] = []
    fitted_models: dict[str, BaseEstimator] = {}
    for spec in specs:
        model = spec.estimator
        model.fit(X_train, y_train)
        valid_metrics = evaluate(model, X_valid, y_valid)
        test_metrics = evaluate(model, X_test, y_test)
        y_test_pred = np.asarray(model.predict(X_test)).astype(int)
        record = {
            "model": spec.name,
            "params": spec.params,
            "validation": valid_metrics,
            "test": test_metrics,
            "classification_report": classification_report(
                y_test,
                y_test_pred,
                target_names=["normal", "defect"],
                output_dict=True,
                zero_division=0,
            ),
        }
        records.append(record)
        fitted_models[spec.name] = model

        if not args.skip_mlflow:
            with mlflow.start_run(run_name=f"{spec.name}_baseline"):
                mlflow.log_params(spec.params)
                mlflow.log_param("owner", "team8_modeling")
                mlflow.log_param("target_column", target_column)
                mlflow.log_param("data_version", cfg["project"]["data_version"])
                for key, value in valid_metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"valid_{key}", value)
                for key, value in test_metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"test_{key}", value)
                mlflow.set_tag("stage", "baseline_comparison")

    records.sort(
        key=lambda item: (item["validation"]["f1"], item["validation"]["roc_auc"]),
        reverse=True,
    )
    best_record = records[0]
    best_model = fitted_models[best_record["model"]]
    save_candidate_artifact(best_model, best_record, feature_names, cfg)

    shap_status = {"status": "skipped", "message": "SHAP disabled by --no-shap."}
    if not args.no_shap:
        shap_status = save_shap_outputs(
            best_model,
            best_record,
            X_train,
            X_test,
            y_test,
            sample_size=args.shap_sample_size,
        )

    write_reports(records, skipped, best_record, shap_status)

    if not args.skip_mlflow:
        with mlflow.start_run(run_name="baseline_xai_summary"):
            mlflow.log_param("owner", "team8_modeling")
            mlflow.log_param("selection_rule", "validation_f1_then_validation_roc_auc")
            mlflow.log_metric("best_validation_f1", best_record["validation"]["f1"])
            mlflow.log_metric("best_validation_roc_auc", best_record["validation"]["roc_auc"])
            mlflow.set_tag("stage", "baseline_xai_summary")
            for path in [
                REPORT_DIR / "baseline_comparison.csv",
                REPORT_DIR / "baseline_metric_table.md",
                REPORT_DIR / "candidate_handoff_note.md",
                REPORT_DIR / "xai_feature_interpretation.md",
                REPORT_DIR / "shap_local_explanation.md",
                PLOT_DIR / "baseline_validation_metrics.png",
                PLOT_DIR / "baseline_confusion_matrices.png",
                PLOT_DIR / "shap_summary_bar.png",
                PLOT_DIR / "shap_beeswarm.png",
                PLOT_DIR / "shap_waterfall_defect_sample.png",
                DOC_REPORT_PATH,
            ]:
                if path.exists():
                    mlflow.log_artifact(str(path))

    return {
        "best_model": best_record["model"],
        "validation": best_record["validation"],
        "test": best_record["test"],
        "skipped": skipped,
        "shap": shap_status,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline candidates and generate SHAP/XAI outputs."
    )
    parser.add_argument("--skip-mlflow", action="store_true", help="Do not log comparison runs to MLflow.")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP artifact generation.")
    parser.add_argument(
        "--shap-sample-size",
        type=int,
        default=300,
        help="Number of test rows to explain with SHAP.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        result = train_and_compare(parse_args())
    except FileNotFoundError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
