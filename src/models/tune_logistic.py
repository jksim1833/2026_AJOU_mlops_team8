from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "params.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def configure_mlflow(cfg: dict[str, Any]) -> None:
    mlflow.set_tracking_uri(cfg["tracking"]["tracking_uri"])
    mlflow.set_experiment(cfg["tracking"]["experiment_name"])


def load_xy(name: str, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(ROOT / "data" / "processed" / f"{name}.csv")
    return df.drop(columns=[target_column]), df[target_column].astype(int)


def baseline_params() -> dict[str, Any]:
    return {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "class_weight": "balanced",
    }


def build_model(params: dict[str, Any], random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    **params,
                    max_iter=10000,
                    tol=1e-3 if params["solver"] == "saga" else 1e-4,
                    random_state=random_state,
                ),
            ),
        ]
    )


def search_candidates(iterations: int, random_state: int) -> list[dict[str, Any]]:
    valid_solver_penalties = [
        ("lbfgs", "l2"),
        ("liblinear", "l1"),
        ("liblinear", "l2"),
        ("saga", "l1"),
        ("saga", "l2"),
    ]
    c_values = np.logspace(-3, 3, 25)
    universe = [
        {
            "C": float(c_value),
            "penalty": penalty,
            "solver": solver,
            "class_weight": class_weight,
        }
        for solver, penalty in valid_solver_penalties
        for class_weight in [None, "balanced"]
        for c_value in c_values
    ]
    baseline = baseline_params()
    universe = [candidate for candidate in universe if candidate != baseline]
    rng = np.random.default_rng(random_state)
    sample_size = min(max(0, iterations - 1), len(universe))
    selected_indices = rng.choice(len(universe), size=sample_size, replace=False)
    return [baseline, *[universe[int(index)] for index in selected_indices]]


def is_valid_candidate(params: dict[str, Any]) -> bool:
    solver = params["solver"]
    penalty = params["penalty"]
    return (solver == "lbfgs" and penalty == "l2") or (
        solver in {"liblinear", "saga"} and penalty in {"l1", "l2"}
    )


def run_parameter_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    tuning_cfg = cfg["tuning"]
    random_state = int(tuning_cfg["random_state"])
    cv = StratifiedKFold(
        n_splits=int(tuning_cfg["cv_folds"]),
        shuffle=True,
        random_state=random_state,
    )
    rows: list[dict[str, Any]] = []
    for candidate_id, params in enumerate(
        search_candidates(int(tuning_cfg["search_iterations"]), random_state),
        start=1,
    ):
        scores = cross_val_score(
            build_model(params, random_state),
            X_train,
            y_train,
            scoring="f1",
            cv=cv,
            n_jobs=1,
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "is_baseline": candidate_id == 1,
                "cv_f1_mean": float(scores.mean()),
                "cv_f1_std": float(scores.std()),
                **params,
            }
        )
    results = pd.DataFrame(rows).sort_values(
        ["cv_f1_mean", "cv_f1_std"],
        ascending=[False, True],
    )
    best = results.iloc[0]
    best_params = {
        "C": float(best["C"]),
        "penalty": str(best["penalty"]),
        "solver": str(best["solver"]),
        "class_weight": None if pd.isna(best["class_weight"]) else str(best["class_weight"]),
    }
    return results, best_params


def metrics_from_scores(
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_threshold(
    y_true: pd.Series,
    scores: np.ndarray,
    minimum: float,
    maximum: float,
    step: float,
) -> tuple[pd.DataFrame, float, dict[str, float | int]]:
    thresholds = np.round(np.arange(minimum, maximum + step / 2, step), 10)
    rows = [
        {"threshold": float(threshold), **metrics_from_scores(y_true, scores, float(threshold))}
        for threshold in thresholds
    ]
    results = pd.DataFrame(rows)
    ranked = results.assign(distance_from_half=(results["threshold"] - 0.5).abs()).sort_values(
        ["f1", "recall", "precision", "distance_from_half"],
        ascending=[False, False, False, True],
    )
    best = ranked.iloc[0]
    threshold = float(best["threshold"])
    metrics = {
        key: int(best[key]) if key in {"tn", "fp", "fn", "tp"} else float(best[key])
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "tn", "fp", "fn", "tp"]
    }
    return results, threshold, metrics


def save_threshold_plot(results: pd.DataFrame, selected: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in ["f1", "recall", "precision"]:
        ax.plot(results["threshold"], results[metric], label=metric.upper())
    ax.axvline(selected, color="black", linestyle="--", label=f"selected={selected:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Validation score")
    ax.set_ylim(0, 1)
    ax.set_title("Logistic Regression Validation Threshold Tuning")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_evaluation_plots(
    y_test: pd.Series,
    test_scores: np.ndarray,
    threshold: float,
    confusion_path: Path,
    roc_path: Path,
) -> None:
    predictions = (test_scores >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["normal", "defect"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"Champion Confusion Matrix (threshold={threshold:.2f})")
    fig.tight_layout()
    confusion_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(confusion_path, dpi=160)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, test_scores)
    auc_value = roc_auc_score(y_test, test_scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Champion ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=160)
    plt.close(fig)


def save_explanations(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
    feature_path: Path,
    coefficient_path: Path,
    permutation_path: Path,
    plot_path: Path,
    random_state: int,
    threshold: float,
) -> list[dict[str, float]]:
    estimator = model.named_steps["model"]
    coefficients = [
        {
            "feature": feature,
            "coefficient": float(coefficient),
            "absolute_coefficient": float(abs(coefficient)),
            "direction": "defect" if coefficient > 0 else "normal",
        }
        for feature, coefficient in zip(feature_names, estimator.coef_[0])
    ]
    coefficients.sort(key=lambda item: item["absolute_coefficient"], reverse=True)
    coefficient_path.write_text(
        json.dumps(coefficients, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    def threshold_f1(estimator: Pipeline, features: pd.DataFrame, labels: pd.Series) -> float:
        scores = estimator.predict_proba(features)[:, 1]
        predictions = (scores >= threshold).astype(int)
        return float(f1_score(labels, predictions, zero_division=0))

    permutation = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=threshold_f1,
        n_repeats=10,
        random_state=random_state,
        n_jobs=1,
    )
    permutation_records = [
        {
            "feature": feature,
            "importance": float(mean),
            "importance_std": float(std),
        }
        for feature, mean, std in zip(
            feature_names,
            permutation.importances_mean,
            permutation.importances_std,
        )
    ]
    permutation_records.sort(key=lambda item: item["importance"], reverse=True)
    permutation_path.write_text(
        json.dumps(permutation_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    feature_path.write_text(
        json.dumps(permutation_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    top = permutation_records[:12]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [item["feature"] for item in reversed(top)],
        [item["importance"] for item in reversed(top)],
        xerr=[item["importance_std"] for item in reversed(top)],
    )
    ax.set_title("Logistic Champion Permutation Importance")
    ax.set_xlabel("Mean F1 decrease")
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return permutation_records


def save_samples(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scores: np.ndarray,
    threshold: float,
    sample_path: Path,
    normal_path: Path,
    defect_path: Path,
) -> None:
    predictions = pd.Series((scores >= threshold).astype(int), index=X_test.index)
    samples: dict[str, dict[str, float]] = {}
    for label, name in [(0, "normal_sample"), (1, "defect_sample")]:
        correct = y_test[(y_test == label) & (predictions == label)].index
        idx = correct[0] if len(correct) else y_test[y_test == label].index[0]
        samples[name] = {column: float(X_test.loc[idx, column]) for column in X_test.columns}
    sample_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    normal_path.write_text(
        json.dumps({"features": samples["normal_sample"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    defect_path.write_text(
        json.dumps({"features": samples["defect_sample"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def demote_previous_champions(experiment_id: str, current_run_id: str) -> None:
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.stage = 'champion'",
    )
    for previous_run in runs:
        if previous_run.info.run_id == current_run_id:
            continue
        client.set_tag(previous_run.info.run_id, "stage", "superseded")
        client.set_tag(previous_run.info.run_id, "serving_candidate", "false")


def write_tuning_report(
    path: Path,
    baseline_metrics: dict[str, float | int],
    tuned_metrics: dict[str, float | int],
    test_metrics: dict[str, float | int],
    threshold: float,
    best_params: dict[str, Any],
    run_id: str,
) -> None:
    lines = [
        "# Logistic Regression Tuning Result",
        "",
        f"- MLflow champion run: `{run_id}`",
        f"- Selected threshold: `{threshold:.2f}`",
        "- Selection rule: validation F1, then recall, precision, then distance to 0.5",
        "- Final artifact fit: train split only",
        "",
        "| Result | F1 | ROC-AUC | Recall | Precision |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Baseline validation @0.50 | {baseline_metrics['f1']:.3f} | {baseline_metrics['roc_auc']:.3f} | {baseline_metrics['recall']:.3f} | {baseline_metrics['precision']:.3f} |",
        f"| Tuned validation @{threshold:.2f} | {tuned_metrics['f1']:.3f} | {tuned_metrics['roc_auc']:.3f} | {tuned_metrics['recall']:.3f} | {tuned_metrics['precision']:.3f} |",
        f"| Champion test @{threshold:.2f} | {test_metrics['f1']:.3f} | {test_metrics['roc_auc']:.3f} | {test_metrics['recall']:.3f} | {test_metrics['precision']:.3f} |",
        "",
        "## Best Parameters",
        "",
        "```json",
        json.dumps(best_params, ensure_ascii=False, indent=2),
        "```",
        "",
        "AutoGluon and H2O are excluded from the required pipeline because their runtime and "
        "serving dependencies are disproportionate to this MVP. The week11 AutoML-lite "
        "leaderboard pattern is used instead.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = load_config()
    configure_mlflow(cfg)
    target_column = cfg["data"]["target_column"]
    X_train, y_train = load_xy("train", target_column)
    X_valid, y_valid = load_xy("valid", target_column)
    X_test, y_test = load_xy("test", target_column)
    tuning_cfg = cfg["tuning"]
    artifact_cfg = cfg["artifacts"]
    random_state = int(tuning_cfg["random_state"])

    tuning_results, best_params = run_parameter_search(X_train, y_train, cfg)
    tuning_results_path = ROOT / artifact_cfg["tuning_results_path"]
    tuning_results_path.parent.mkdir(parents=True, exist_ok=True)
    tuning_results.to_csv(tuning_results_path, index=False)

    baseline_model = build_model(baseline_params(), random_state).fit(X_train, y_train)
    baseline_valid_scores = baseline_model.predict_proba(X_valid)[:, 1]
    baseline_valid_metrics = metrics_from_scores(y_valid, baseline_valid_scores, 0.5)

    champion = build_model(best_params, random_state).fit(X_train, y_train)
    valid_scores = champion.predict_proba(X_valid)[:, 1]
    threshold_results, selected_threshold, valid_metrics = select_threshold(
        y_valid,
        valid_scores,
        float(tuning_cfg["threshold_min"]),
        float(tuning_cfg["threshold_max"]),
        float(tuning_cfg["threshold_step"]),
    )
    threshold_results_path = ROOT / artifact_cfg["threshold_results_path"]
    threshold_results.to_csv(threshold_results_path, index=False)
    threshold_plot_path = ROOT / artifact_cfg["threshold_plot_path"]
    save_threshold_plot(threshold_results, selected_threshold, threshold_plot_path)

    # The test split is touched only after parameters and threshold are fixed.
    test_scores = champion.predict_proba(X_test)[:, 1]
    test_metrics = metrics_from_scores(y_test, test_scores, selected_threshold)

    model_path = ROOT / artifact_cfg["model_path"]
    metadata_path = ROOT / artifact_cfg["metadata_path"]
    metrics_path = ROOT / artifact_cfg["metrics_path"]
    feature_path = ROOT / artifact_cfg["feature_importance_path"]
    coefficient_path = ROOT / artifact_cfg["coefficient_path"]
    permutation_path = ROOT / artifact_cfg["permutation_importance_path"]
    confusion_path = ROOT / artifact_cfg["confusion_matrix_path"]
    roc_path = ROOT / artifact_cfg["roc_curve_path"]
    feature_plot_path = ROOT / artifact_cfg["feature_importance_plot_path"]
    sample_path = ROOT / artifact_cfg["sample_inputs_path"]
    normal_path = ROOT / artifact_cfg["normal_request_path"]
    defect_path = ROOT / artifact_cfg["defect_request_path"]
    tuning_report_path = ROOT / artifact_cfg["tuning_report_path"]

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, model_path)
    save_evaluation_plots(y_test, test_scores, selected_threshold, confusion_path, roc_path)
    feature_importance = save_explanations(
        champion,
        X_test,
        y_test,
        list(X_train.columns),
        feature_path,
        coefficient_path,
        permutation_path,
        feature_plot_path,
        random_state,
        selected_threshold,
    )
    save_samples(
        X_test,
        y_test,
        test_scores,
        selected_threshold,
        sample_path,
        normal_path,
        defect_path,
    )

    test_predictions = (test_scores >= selected_threshold).astype(int)
    metrics = {
        "baseline_validation_at_0_5": baseline_valid_metrics,
        "validation": valid_metrics,
        "test": test_metrics,
        "classification_report": classification_report(
            y_test,
            test_predictions,
            target_names=["normal", "defect"],
            output_dict=True,
            zero_division=0,
        ),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    with mlflow.start_run(run_name="logistic_tuned_champion") as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        metadata = {
            "project": cfg["project"]["name"],
            "artifact_role": "serving_champion",
            "model_type": "LogisticRegression",
            "model_version": cfg["project"]["model_version"],
            "data_version": cfg["project"]["data_version"],
            "target": {"0": "normal", "1": "defect"},
            "features": list(X_train.columns),
            "decision_threshold": selected_threshold,
            "mlflow_run_id": run_id,
            "best_params": best_params,
            "selection_rule": "validation F1, then recall, precision, then distance to 0.5",
            "final_fit_data": "train split only",
            "validation_metrics": valid_metrics,
            "test_metrics": test_metrics,
            "top_features": feature_importance[:8],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_tuning_report(
            tuning_report_path,
            baseline_valid_metrics,
            valid_metrics,
            test_metrics,
            selected_threshold,
            best_params,
            run_id,
        )

        mlflow.log_params({f"best_{key}": value for key, value in best_params.items()})
        mlflow.log_param("data_version", cfg["project"]["data_version"])
        mlflow.log_param("decision_threshold", selected_threshold)
        mlflow.log_param("search_iterations", int(tuning_cfg["search_iterations"]))
        mlflow.log_param("cv_folds", int(tuning_cfg["cv_folds"]))
        for prefix, values in [
            ("baseline_valid", baseline_valid_metrics),
            ("valid", valid_metrics),
            ("test", test_metrics),
        ]:
            for key, value in values.items():
                mlflow.log_metric(f"{prefix}_{key}", value)
        mlflow.set_tag("stage", "champion")
        mlflow.set_tag("serving_candidate", "true")
        mlflow.set_tag("model_family", "logistic_regression")
        for path, artifact_path in [
            (model_path, "model"),
            (metadata_path, "model"),
            (metrics_path, "reports"),
            (tuning_results_path, "reports"),
            (threshold_results_path, "reports"),
            (tuning_report_path, "reports"),
            (coefficient_path, "reports"),
            (permutation_path, "reports"),
            (confusion_path, "plots"),
            (roc_path, "plots"),
            (feature_plot_path, "plots"),
            (threshold_plot_path, "plots"),
            (normal_path, "examples"),
            (defect_path, "examples"),
        ]:
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
        demote_previous_champions(experiment_id, run_id)

    print(
        json.dumps(
            {
                "run_id": run_id,
                "decision_threshold": selected_threshold,
                "validation": valid_metrics,
                "test": test_metrics,
                "best_params": best_params,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
