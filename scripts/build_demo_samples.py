"""Build a small set of demo samples drawn from the real test split.

Selects high-confidence, correctly classified normal and defect rows from
`data/processed/test.csv` so the FastAPI demo UI can offer realistic examples
with their true labels. Output: `artifacts/reports/demo_samples.json`.

Run from the project root:

    python -m scripts.build_demo_samples
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "models" / "model.joblib"
METADATA_PATH = ROOT / "artifacts" / "models" / "metadata.json"
TEST_CSV = ROOT / "data" / "processed" / "test.csv"
OUTPUT_PATH = ROOT / "artifacts" / "reports" / "demo_samples.json"

LABEL_COLUMN = "defect_label"
N_PER_CLASS = 3


def main() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    features = metadata["features"]
    threshold = float(metadata.get("decision_threshold", 0.5))
    labels = metadata["target"]

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(TEST_CSV).reset_index(drop=True)

    defect_proba = model.predict_proba(df[features])[:, 1]
    df = df.assign(
        _defect_proba=defect_proba,
        _prediction=(defect_proba >= threshold).astype(int),
    )
    correct = df[df["_prediction"] == df[LABEL_COLUMN]]

    samples = []
    # Most confident correctly classified normals (lowest defect probability).
    normals = correct[correct[LABEL_COLUMN] == 0].nsmallest(N_PER_CLASS, "_defect_proba")
    # Most confident correctly classified defects (highest defect probability).
    defects = correct[correct[LABEL_COLUMN] == 1].nlargest(N_PER_CLASS, "_defect_proba")

    for rank, (idx, row) in enumerate(normals.iterrows(), start=1):
        samples.append(_to_sample(f"정상 샘플 {rank}", idx, row, features, labels))
    for rank, (idx, row) in enumerate(defects.iterrows(), start=1):
        samples.append(_to_sample(f"불량 샘플 {rank}", idx, row, features, labels))

    payload = {
        "data_version": metadata.get("data_version"),
        "decision_threshold": threshold,
        "source": "data/processed/test.csv",
        "note": "test split에서 모델이 올바르게 분류한 고신뢰 샘플",
        "samples": samples,
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(samples)} demo samples to {OUTPUT_PATH.relative_to(ROOT)}")


def _to_sample(name, idx, row, features, labels):
    true_int = int(row[LABEL_COLUMN])
    return {
        "id": name,
        "true_label_int": true_int,
        "true_label": labels[str(true_int)],
        "test_row_index": int(idx),
        "expected_defect_proba": round(float(row["_defect_proba"]), 4),
        "features": {feature: float(row[feature]) for feature in features},
    }


if __name__ == "__main__":
    main()
