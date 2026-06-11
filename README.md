# Diecasting Binary Defect Classification MLOps

공개 제조 AI 데이터셋인 KAMP 주조 품질보증 AI 데이터셋을 활용해 다이캐스팅 제품의 **정상/불량 이진 분류 모델**을 만들고, MLflow 실험 추적, FastAPI serving, 간단 UI, Docker 실행 구조까지 연결하는 기말 프로젝트입니다.

## One Sentence Problem Definition

우리는 공정 엔지니어를 위해 KAMP 다이캐스팅 공정·센서 데이터를 사용하여 제품이 정상인지 불량인지 예측하고, XAI로 주요 판단 근거를 제공한다. 성공은 validation F1/ROC-AUC, MLflow 실험 기록, FastAPI+간단 UI 데모, Docker 실행 구조로 확인한다.

## Team Roles

| 역할 | 담당자 | 주요 책임 |
| :--- | :--- | :--- |
| Data | 김병근 | 데이터 출처, EDA, binary label, split, DVC 전략 |
| Modeling | Zhang Xin | RF baseline, 후보 모델 비교, champion 선정 |
| MLOps/Serving | 심재광 | GitHub 구조, MLflow, FastAPI, Docker |
| 발표/문서화 | 심재광 | 보고서, 발표자료, demo script |

## Project Structure

```text
diecasting-mlops/
  README.md
  configs/params.yaml
  data/raw/
  data/processed/
  notebooks/
  src/data/prepare_data.py
  src/models/train_binary.py
  src/api/main.py
  src/ui/app.py
  artifacts/models/
  artifacts/plots/
  artifacts/reports/
  dvc.yaml
  Dockerfile
  requirements.txt
  requirements-api.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## One-command Local Run on Windows

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1
```

Open this repository in PyCharm:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\open_pycharm.ps1
```

PyCharm shared run configurations are included under `.run/`:

- `FastAPI`
- `MLflow UI`
- `Streamlit Dashboard`
- `Zhang Baseline XAI`

This starts the local demo services:

- MLflow UI: `http://127.0.0.1:5000`
- FastAPI Swagger: `http://127.0.0.1:8000/docs`
- Streamlit UI: `http://127.0.0.1:8501`

To stop services:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stop_local.ps1
```

To rerun the full local data/model/XAI pipeline, first place the KAMP raw CSV at
`data/raw/DieCasting_Quality_Raw_Data_product1.csv`, then run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1 -RunPipeline
```

The dataset is KAMP `DATASET_SEQ=55`, "주조 품질보증 AI 데이터셋":
`https://www.kamp-ai.kr/aidataDetail?DATASET_SEQ=55`

## 1. Prepare Data

Raw data는 `data/raw/DieCasting_Quality_Raw_Data_product1.csv`에 둡니다. 결함 컬럼 중 하나라도 1이면 `defect_label=1`, 모두 0이면 `defect_label=0`으로 생성합니다. 결함 컬럼은 feature에서 제거합니다.

```bash
python -m src.data.prepare_data
```

생성 산출물:

- `data/processed/diecasting_product1_binary.csv`
- `data/processed/train.csv`
- `data/processed/valid.csv`
- `data/processed/test.csv`
- `artifacts/reports/data_profile.json`

## 2. Train Baseline with MLflow

```bash
python -m src.models.train_binary
```

## 2-1. Zhang Xin Baseline Comparison and SHAP XAI

Zhang Xin's modeling task compares multiple baseline candidates on the same
train/validation/test split, selects a tuning candidate by validation F1, and
generates SHAP-based global/local explanations.

```bash
python -m src.models.compare_baselines_xai
```

Main outputs:

- `artifacts/reports/baseline_metric_table.md`
- `artifacts/reports/baseline_comparison.csv`
- `artifacts/reports/candidate_handoff_note.md`
- `docs/zhang_baseline_xai_report.md`
- `artifacts/reports/xai_feature_interpretation.md`
- `artifacts/reports/shap_local_explanation.md`
- `artifacts/plots/baseline_validation_metrics.png`
- `artifacts/plots/baseline_confusion_matrices.png`
- `artifacts/plots/shap_summary_bar.png`
- `artifacts/plots/shap_beeswarm.png`
- `artifacts/plots/shap_waterfall_defect_sample.png`
- `artifacts/models/baseline_candidate.joblib`

MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

최신 MLflow에서는 local file store 대신 SQLite backend를 사용합니다. `mlflow.db`가 local tracking store 역할을 합니다.

주요 산출물:

- `artifacts/models/model.joblib`
- `artifacts/models/metadata.json`
- `artifacts/reports/metrics.json`
- `artifacts/reports/feature_importance.json`
- `artifacts/reports/sample_inputs.json`
- `artifacts/plots/confusion_matrix.png`
- `artifacts/plots/roc_curve.png`
- `artifacts/plots/feature_importance.png`

## 3. Run FastAPI

```bash
uvicorn src.api.main:app --reload --port 8000
```

Endpoints:

- `GET /health`
- `GET /model-info`
- `POST /predict`

예시 요청:

```bash
curl -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d @artifacts/reports/normal_request.json
```

발표용 예시 요청은 `artifacts/reports/normal_request.json`, `artifacts/reports/defect_request.json`에 저장됩니다.

## 4. Run Streamlit UI

FastAPI 서버를 먼저 실행한 뒤 별도 터미널에서 실행합니다.

```bash
streamlit run src/ui/app.py
```

## 5. Docker Demo

학습 산출물(`artifacts/models`, `artifacts/reports`)이 생성된 뒤 Docker image를 빌드합니다.
Dockerfile은 serving demo를 빠르게 만들기 위해 `requirements-api.txt`만 설치합니다.

```bash
docker build -t diecasting-api .
docker run -p 8000:8000 diecasting-api
```

컨테이너 실행 후 확인:

```bash
curl http://localhost:8000/health
```

## MLOps Components

| 영역 | 적용 방식 |
| :--- | :--- |
| Git/GitHub | 코드, config, README, Dockerfile 버전관리 |
| Data Version Control | raw/processed/split 데이터 버전 전략 문서화, DVC 적용 가능 구조 |
| MLflow | params, metrics, artifacts, champion tag 기록 |
| XAI/Error Analysis | feature importance와 top feature 기반 설명 |
| Serving | FastAPI `/health`, `/model-info`, `/predict` |
| Web UI | Streamlit sample prediction demo |
| Docker | FastAPI app 컨테이너 실행 구조 |

## DVC Pipeline

`dvc.yaml`에는 `prepare_data`, `train_binary` 두 stage가 정의되어 있습니다.

```bash
dvc init
dvc repro
```

자세한 데이터 버전 전략은 `docs/data_versioning.md`를 참고합니다.

## Final Presentation Demo Path

1. 문제정의와 binary target 설명
2. EDA/class distribution과 leakage 방지 설명
3. MLflow UI 또는 metrics table로 baseline 결과 제시
4. feature importance plot으로 XAI 설명
5. FastAPI Swagger 또는 curl로 `/predict` 호출
6. Streamlit UI에서 정상/불량 샘플 예측
7. Dockerfile과 `docker run` 명령으로 컨테이너 실행 구조 제시
