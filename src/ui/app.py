from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
API_URL = "http://localhost:8000"
REPORT_DIR = ROOT / "artifacts" / "reports"
PLOT_DIR = ROOT / "artifacts" / "plots"
MODEL_DIR = ROOT / "artifacts" / "models"
SAMPLE_INPUTS_PATH = REPORT_DIR / "sample_inputs.json"
METRICS_PATH = REPORT_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = REPORT_DIR / "feature_importance.json"
BASELINE_TABLE_PATH = REPORT_DIR / "baseline_metric_table.md"
CANDIDATE_NOTE_PATH = REPORT_DIR / "candidate_handoff_note.md"
XAI_INTERPRETATION_PATH = REPORT_DIR / "xai_feature_interpretation.md"
LOCAL_EXPLANATION_PATH = REPORT_DIR / "shap_local_explanation.md"
TUNING_REPORT_PATH = REPORT_DIR / "logistic_tuning_report.md"
DATA_PROFILE_PATH = REPORT_DIR / "data_profile.json"
METADATA_PATH = MODEL_DIR / "metadata.json"


st.set_page_config(page_title="Diecasting Defect Demo", layout="wide")
st.title("Diecasting Defect Classification Dashboard")


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def show_plot(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Missing artifact: {path.relative_to(ROOT)}")


def metric_block(title: str, metrics: dict) -> None:
    st.subheader(title)
    cols = st.columns(5)
    for col, key in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        value = metrics.get(key)
        col.metric(key.upper().replace("_", "-"), f"{value:.3f}" if isinstance(value, float) else "-")


def prediction_tab() -> None:
    if not SAMPLE_INPUTS_PATH.exists():
        st.error("sample_inputs.json is missing.")
        return

    samples = load_json(SAMPLE_INPUTS_PATH)
    api_url = st.sidebar.text_input("FastAPI URL", value=API_URL)
    sample_name = st.selectbox("Sample", list(samples.keys()))
    payload = {"features": samples[sample_name]}

    left, right = st.columns([1, 1])
    with left:
        st.json(payload)

    if st.button("Predict", type="primary"):
        try:
            response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            st.error(f"API request failed: {exc}")
            return

        result = response.json()
        label = "defect" if result["prediction"] == 1 else "normal"
        with right:
            st.metric("Prediction", label)
            col1, col2 = st.columns(2)
            col1.metric("Normal probability", f"{result['probability']['normal']:.3f}")
            col2.metric("Defect probability", f"{result['probability']['defect']:.3f}")
            st.dataframe(pd.DataFrame(result["top_features"]), use_container_width=True)
            st.json(result)


def metrics_tab() -> None:
    metrics = load_json(METRICS_PATH)
    metadata = load_json(METADATA_PATH)
    data_profile = load_json(DATA_PROFILE_PATH)

    if metadata:
        st.caption(
            f"Champion model: `{metadata.get('model_version', '-')}` | "
            f"Data: `{metadata.get('data_version', '-')}`"
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model version", metadata.get("model_version", "-"))
        c2.metric("Data version", metadata.get("data_version", "-"))
        c3.metric("Feature count", len(metadata.get("features", [])))
        threshold = metadata.get("decision_threshold")
        c4.metric("Decision threshold", f"{threshold:.2f}" if isinstance(threshold, float) else "-")
        if metadata.get("mlflow_run_id"):
            st.caption(f"MLflow champion run: `{metadata['mlflow_run_id']}`")

    if data_profile:
        counts = data_profile.get("class_counts", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", data_profile.get("rows", "-"))
        c2.metric("Normal", counts.get("0", "-"))
        c3.metric("Defect", counts.get("1", "-"))

    if metrics:
        metric_block("Validation", metrics.get("validation", {}))
        metric_block("Test", metrics.get("test", {}))
        report = metrics.get("classification_report", {})
        if report:
            st.subheader("Classification Report")
            st.dataframe(pd.DataFrame(report).T, use_container_width=True)
    else:
        st.info("metrics.json is missing.")

    plot_cols = st.columns(3)
    with plot_cols[0]:
        show_plot(PLOT_DIR / "confusion_matrix.png", "Confusion matrix")
    with plot_cols[1]:
        show_plot(PLOT_DIR / "roc_curve.png", "ROC curve")
    with plot_cols[2]:
        show_plot(PLOT_DIR / "feature_importance.png", "Feature importance")


def feature_tab() -> None:
    importance = load_json(FEATURE_IMPORTANCE_PATH)
    if not importance:
        st.info("feature_importance.json is missing.")
        return

    df = pd.DataFrame(importance)
    top_n = st.slider("Top features", min_value=5, max_value=min(28, len(df)), value=min(12, len(df)))
    st.bar_chart(df.head(top_n).set_index("feature")["importance"])
    st.dataframe(df.head(top_n), use_container_width=True)


def xai_tab() -> None:
    baseline_table = load_text(BASELINE_TABLE_PATH)
    candidate_note = load_text(CANDIDATE_NOTE_PATH)
    xai_interpretation = load_text(XAI_INTERPRETATION_PATH)
    local_explanation = load_text(LOCAL_EXPLANATION_PATH)
    tuning_report = load_text(TUNING_REPORT_PATH)

    st.subheader("Champion Tuning")
    st.markdown(tuning_report or "Run `python -m src.models.tune_logistic` to generate tuning results.")
    show_plot(PLOT_DIR / "logistic_threshold_tuning.png", "Validation threshold tuning")

    st.subheader("Baseline Comparison")
    if baseline_table:
        st.markdown(baseline_table)
    else:
        st.info("Run `python -m src.models.compare_baselines_xai` after adding the raw CSV to generate comparison artifacts.")

    cols = st.columns(2)
    with cols[0]:
        show_plot(PLOT_DIR / "baseline_validation_metrics.png", "Baseline validation metrics")
    with cols[1]:
        show_plot(PLOT_DIR / "baseline_confusion_matrices.png", "Baseline confusion matrices")

    st.subheader("Candidate Handoff")
    st.markdown(candidate_note or "Candidate handoff note is not generated yet.")

    st.subheader("SHAP Summary")
    cols = st.columns(2)
    with cols[0]:
        show_plot(PLOT_DIR / "shap_summary_bar.png", "SHAP summary bar")
    with cols[1]:
        show_plot(PLOT_DIR / "shap_beeswarm.png", "SHAP beeswarm")
    show_plot(PLOT_DIR / "shap_waterfall_defect_sample.png", "SHAP local waterfall")

    st.subheader("XAI Interpretation")
    st.markdown(xai_interpretation or "XAI feature interpretation is not generated yet.")
    st.subheader("Local Explanation")
    st.markdown(local_explanation or "Local SHAP explanation is not generated yet.")


def artifacts_tab() -> None:
    rows = []
    for path in [
        METRICS_PATH,
        FEATURE_IMPORTANCE_PATH,
        BASELINE_TABLE_PATH,
        CANDIDATE_NOTE_PATH,
        XAI_INTERPRETATION_PATH,
        LOCAL_EXPLANATION_PATH,
        TUNING_REPORT_PATH,
        PLOT_DIR / "confusion_matrix.png",
        PLOT_DIR / "roc_curve.png",
        PLOT_DIR / "feature_importance.png",
        PLOT_DIR / "shap_summary_bar.png",
        PLOT_DIR / "logistic_threshold_tuning.png",
        MODEL_DIR / "model.joblib",
        MODEL_DIR / "baseline_candidate.joblib",
    ]:
        rows.append(
            {
                "artifact": str(path.relative_to(ROOT)),
                "exists": path.exists(),
                "size_kb": round(path.stat().st_size / 1024, 1) if path.exists() else None,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

tabs = st.tabs(["Predict", "Metrics", "Feature Importance", "Baseline/XAI", "Artifacts"])
with tabs[0]:
    prediction_tab()
with tabs[1]:
    metrics_tab()
with tabs[2]:
    feature_tab()
with tabs[3]:
    xai_tab()
with tabs[4]:
    artifacts_tab()
