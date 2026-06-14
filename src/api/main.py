from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "models" / "model.joblib"
METADATA_PATH = ROOT / "artifacts" / "models" / "metadata.json"
FEATURE_IMPORTANCE_PATH = ROOT / "artifacts" / "reports" / "feature_importance.json"
DEMO_SAMPLES_PATH = ROOT / "artifacts" / "reports" / "demo_samples.json"
INDEX_HTML_PATH = Path(__file__).resolve().parent / "static" / "index.html"


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(..., description="공정/센서 feature 이름과 값")


class PredictionResponse(BaseModel):
    prediction: int
    label_name: str
    probability: dict[str, float]
    top_features: list[dict[str, Any]]
    model_version: str
    data_version: str


def load_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_model_bundle() -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model artifact not found. Run src.models.train_binary first.")
    model = joblib.load(MODEL_PATH)
    metadata = load_json(METADATA_PATH)
    feature_importance = load_json(FEATURE_IMPORTANCE_PATH)
    return model, metadata, feature_importance


app = FastAPI(
    title="Diecasting Binary Defect Classification API",
    version="1.0.0",
    description="KAMP 다이캐스팅 공정/센서 데이터 기반 정상/불량 이진 분류 API",
)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if not INDEX_HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="UI page not found.")
    return HTMLResponse(INDEX_HTML_PATH.read_text(encoding="utf-8"))


@app.get("/samples")
def samples() -> dict[str, Any]:
    if not DEMO_SAMPLES_PATH.exists():
        raise HTTPException(status_code=503, detail="demo_samples.json not found.")
    return load_json(DEMO_SAMPLES_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    status = "ok" if MODEL_PATH.exists() and METADATA_PATH.exists() else "model_missing"
    return {"status": status}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    try:
        metadata = load_json(METADATA_PATH)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        model, metadata, feature_importance = load_model_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    expected_features = metadata["features"]
    missing = [feature for feature in expected_features if feature not in request.features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing required features.",
                "missing_features": missing,
            },
        )

    row = pd.DataFrame([{feature: request.features[feature] for feature in expected_features}])
    proba = model.predict_proba(row)[0]
    decision_threshold = float(metadata.get("decision_threshold", 0.5))
    prediction = int(float(proba[1]) >= decision_threshold)
    labels = metadata["target"]
    label_name = labels[str(prediction)]

    top_features = []
    for item in feature_importance[:5]:
        feature_name = item["feature"]
        top_features.append(
            {
                "feature": feature_name,
                "importance": item["importance"],
                "input_value": request.features.get(feature_name),
            }
        )

    return PredictionResponse(
        prediction=prediction,
        label_name=label_name,
        probability={"normal": float(proba[0]), "defect": float(proba[1])},
        top_features=top_features,
        model_version=metadata["model_version"],
        data_version=metadata["data_version"],
    )
