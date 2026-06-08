# Diecasting Binary Defect Classification MLOps

KAMP 주조 품질보증 AI 데이터셋을 활용해 다이캐스팅 제품의 **정상/불량 이진 분류 모델**을 만들고, 데이터 준비부터 MLflow 실험 추적, XAI, FastAPI serving, Streamlit UI, Docker 실행 구조까지 연결한 end-to-end MLOps 프로젝트입니다.

이 프로젝트의 목표는 단순히 모델 성능표 하나를 만드는 것이 아니라, 제조 공정 데이터를 실제 서비스 가능한 형태로 정리하고 재현 가능한 MLOps 흐름으로 보여주는 것입니다.

## One Sentence Problem Definition

**우리는 공정 엔지니어를 위해 KAMP 다이캐스팅 공정·센서 데이터를 사용하여 제품이 정상인지 불량인지 예측하고, XAI로 주요 판단 근거를 제공한다. 성공은 validation F1/ROC-AUC, MLflow 실험 기록, FastAPI+간단 UI 데모, Docker 실행 구조로 확인한다.**

## Project Summary

| 항목 | 내용 |
| :--- | :--- |
| 대상 사용자 | 다이캐스팅 공정 엔지니어 |
| 사용자 문제 | 생산 shot의 공정/센서 조건을 보고 정상/불량 가능성을 빠르게 판단 |
| 입력 데이터 | KAMP Product 1 다이캐스팅 공정·센서 변수 |
| 모델 출력 | `normal` 또는 `defect` 이진 분류 결과와 확률 |
| 설명 출력 | 예측에 영향을 크게 준 주요 feature와 입력값 |
| 성공 기준 | F1/ROC-AUC 성능, MLflow 기록, API/UI demo, Docker 구조 |
| Baseline model | `RandomForestClassifier` |
| Primary metric | F1-score |
| Secondary metrics | ROC-AUC, recall, precision, accuracy, confusion matrix |

## MVP Scope

| 포함 | 제외 또는 Future Work |
| :--- | :--- |
| 정상/불량 binary classification | 주요 불량/치명 불량 3-class 분류 |
| Product 1 데이터만 사용 | Product 2까지 포함한 통합 모델 |
| RandomForest baseline | 복잡한 딥러닝 모델 |
| MLflow local tracking | 운영용 MLflow server/registry 고도화 |
| FastAPI `/predict` | 실시간 공장 설비 연동 |
| Streamlit 간단 UI | 대시보드형 웹앱 고도화 |
| Dockerfile 작성 | 클라우드 배포, CI/CD |
| Feature importance 기반 XAI | 완전한 SHAP 운영 대시보드 |

최종 발표에서는 큰 제품보다 **작더라도 실제로 동작하는 end-to-end demo path**가 더 중요하므로, 본 프로젝트는 재현 가능한 작은 MLOps 시스템 구현에 초점을 둡니다.

## Repository Structure

```text
diecasting-mlops/
  README.md
  project_brief.md
  configs/
    params.yaml
  data/
    raw/
    processed/
  src/
    data/
      prepare_data.py
    models/
      train_binary.py
    api/
      main.py
    ui/
      app.py
  artifacts/
    models/
    plots/
    reports/
  docs/
    data_versioning.md
    final_report_outline.md
    presentation_outline.md
  dvc.yaml
  Dockerfile
  requirements.txt
  requirements-api.txt
  mlflow.db
```

> Note: `data/raw`, `data/processed`, `mlruns`, cache files, virtual environments, and logs are excluded by `.gitignore`. Raw data should be placed locally or managed with DVC/remote storage.

## Team Roles

| 역할 | 담당자 | 주요 책임 |
| :--- | :--- | :--- |
| Data | 김병근 | 데이터 출처, EDA, binary label, split, DVC 전략 |
| Modeling | Zhang Xin | RF baseline, 후보 모델 비교, XAI 초안, champion 후보 추천 |
| MLOps/Serving | 심재광 | GitHub 구조, MLflow, tuning handoff, FastAPI, Streamlit, Docker |
| 발표/문서화 | 심재광 | 보고서, 발표자료, demo script, 최종 README |

## Data Strategy

### Dataset

| 항목 | 내용 |
| :--- | :--- |
| 데이터셋 | KAMP 주조 품질보증 AI 데이터셋 |
| 사용 범위 | Product 1 |
| raw path | `data/raw/DieCasting_Quality_Raw_Data_product1.csv` |
| processed path | `data/processed/diecasting_product1_binary.csv` |
| 전체 row 수 | 4,207 |
| feature 수 | target 포함 29 columns, 학습 feature는 target 제외 28개 |

### Binary Label Rule

원천 데이터에는 Cavity 1/2의 여러 결함 컬럼이 포함됩니다. 본 프로젝트에서는 정상/불량만 분류하므로 다음 규칙으로 target을 만듭니다.

```text
if any defect column == 1:
    defect_label = 1  # defect
else:
    defect_label = 0  # normal
```

label 생성에 사용한 결함 컬럼은 feature에서 반드시 제거합니다. 결함 컬럼이 모델 입력에 남아 있으면 모델이 공정/센서 조건을 학습하는 것이 아니라 정답 정보를 그대로 보게 되는 **label leakage**가 발생하기 때문입니다.

### Removed Defect Columns

`Short_Shot_1`, `Bubble_1`, `Exfoliation_1`, `Blow_Hole_1`, `Stain_1`, `Dent_1`, `Deformation_1`, `Contamination_1`, `Impurity_1`, `Crack_1`, `Scratch_1`, `Buring_Mark_1`, `Inclusions_1`, `Short_Shot_2`, `Bubble_2`, `Exfoliation_2`, `Blow_Hole_2`, `Stain_2`, `Dent_2`, `Deformation_2`, `Contamination_2`, `Impurity_2`, `Crack_2`, `Scratch_2`, `Buring_Mark_2`, `Inclusions_2`

### Class Distribution

| Class | 의미 | Count |
| :--- | :--- | ---: |
| 0 | 정상 | 3,468 |
| 1 | 불량 | 739 |

정상 데이터가 훨씬 많으므로 accuracy만 보면 안 됩니다. 불량 탐지 성능을 확인하기 위해 F1, recall, ROC-AUC를 함께 봅니다.

### Split Strategy

현재 구현은 stratified split을 사용합니다. 기본 비율은 `train/validation/test = 70/15/15`입니다.

| Split | 정상 | 불량 |
| :--- | ---: | ---: |
| Train | 2,427 | 517 |
| Validation | 520 | 111 |
| Test | 521 | 111 |

향후 shot 시간, batch, lot 정보가 명확히 확인되면 group/time-aware split으로 개선할 수 있습니다.

## Baseline Model And Metrics

| 항목 | 내용 |
| :--- | :--- |
| Baseline model | `RandomForestClassifier` |
| Target | `defect_label` binary classification |
| Primary metric | F1-score |
| Secondary metrics | ROC-AUC, recall, precision, accuracy, confusion matrix |
| 모델 파일 | `artifacts/models/model.joblib` |
| Metadata | `artifacts/models/metadata.json` |
| Metrics | `artifacts/reports/metrics.json` |

### Current Baseline Result

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.781 | 0.427 | 0.712 | 0.534 | 0.828 |
| Test | 0.796 | 0.444 | 0.649 | 0.527 | 0.839 |

### Interpretation

- ROC-AUC는 0.83 수준으로, 공정/센서 변수에 정상/불량을 구분할 신호가 어느 정도 있습니다.
- F1은 0.53 수준으로 높지 않습니다. 정상 데이터가 많고 불량 데이터가 적은 class imbalance 영향이 있습니다.
- 불량 recall은 validation 0.712, test 0.649입니다. 불량을 놓치는 것이 위험하므로 향후 threshold tuning 또는 class weight 조정이 필요합니다.
- Precision은 낮은 편입니다. 공정 운영에서는 false alarm 비용과 defect miss 비용 중 어느 쪽을 더 중요하게 볼지 논의가 필요합니다.

## MLOps Architecture

```text
Raw CSV
  -> Binary label creation
  -> Leakage column removal
  -> Train/Valid/Test split
  -> DVC pipeline
  -> Baseline model comparison
  -> Post-baseline tuning
  -> MLflow tracking and artifact logging
  -> Model artifact
  -> Feature importance / XAI
  -> FastAPI /predict
  -> Streamlit UI
  -> Dockerfile
```

## MLOps Components

| 영역 | 적용 방식 |
| :--- | :--- |
| Git/GitHub | 코드, config, README, Dockerfile 버전관리 |
| Data Version Control | raw/processed/split 데이터 버전 전략 문서화, DVC 적용 가능 구조 |
| MLflow | params, metrics, artifacts, candidate/champion tag 기록 목표 |
| XAI/Error Analysis | feature importance와 top feature 기반 설명 |
| Serving | FastAPI `/health`, `/model-info`, `/predict` |
| Web UI | Streamlit sample prediction demo |
| Docker | FastAPI app 컨테이너 실행 구조 |

## DVC Pipeline

`dvc.yaml`에는 `prepare_data`, `train_binary` 두 stage가 정의되어 있습니다.

| Stage | Command | 역할 |
| :--- | :--- | :--- |
| `prepare_data` | `python -m src.data.prepare_data` | raw data에서 binary dataset과 split 생성 |
| `train_binary` | `python -m src.models.train_binary` | model, metrics, plots, request sample 생성 |

```bash
dvc init
dvc repro
```

DVC 설명 포인트:

- 데이터 버전이 바뀌면 모델 결과도 달라집니다.
- `dvc.yaml`로 데이터 준비와 학습 의존성을 명시했습니다.
- 실제 remote storage까지 완벽히 운영하지 않더라도 데이터 lineage를 보여주는 구조를 갖췄습니다.

자세한 데이터 버전 전략은 `docs/data_versioning.md`를 참고합니다.

## MLflow Tracking

현재 `sqlite:///mlflow.db`를 local tracking store로 사용합니다. 현재는 1차 `rf_binary_baseline` run이 기록되어 있습니다.

| 기록 항목 | 내용 |
| :--- | :--- |
| Experiment | `diecasting_binary_defect_classification` |
| 현재 Run | `rf_binary_baseline` |
| 추가 예정 Run | `decision_tree_baseline`, `xgboost_baseline`, `lightgbm_baseline`, `tuned_*` 등 |
| Params | model type, hyperparameters, data version, target column |
| Metrics | validation/test accuracy, precision, recall, F1, ROC-AUC |
| Artifacts | model, metadata, metrics, confusion matrix, ROC curve, feature importance |
| Tags | `stage=baseline`, `stage=tuned`, `stage=champion`, `serving_candidate=true` |

MLflow UI 실행:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

발표에서 보여줄 증거:

- MLflow UI run 화면
- baseline/tuning metric 비교표
- artifact 목록
- champion 또는 candidate_for_serving tag

## Serving API

FastAPI는 모델이 노트북 안에만 있는 것이 아니라 외부에서 호출 가능한 서비스 형태임을 보여주는 장치입니다.

| Endpoint | Method | 설명 |
| :--- | :--- | :--- |
| `/health` | GET | API 상태 확인 |
| `/model-info` | GET | model version, data version, feature list, metric summary 확인 |
| `/predict` | POST | 공정/센서 feature 입력 후 정상/불량 예측 |

`/predict` 응답에는 다음이 포함됩니다.

- `prediction`: 0 또는 1
- `label_name`: `normal` 또는 `defect`
- `probability`: 정상/불량 확률
- `top_features`: 주요 feature importance와 입력값
- `model_version`
- `data_version`

## Streamlit UI

Streamlit UI는 발표자가 API 요청을 직접 타이핑하지 않고, 샘플을 선택해 예측 결과를 보여주기 위한 간단한 서비스 화면입니다.

UI의 역할:

1. 정상/불량 sample 선택
2. API request JSON 확인
3. `/predict` 호출
4. 정상/불량 확률 표시
5. 주요 판단 변수 표시

## Setup

프로젝트 루트:

```bash
cd project/diecasting-mlops
```

환경 설치:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Prepare Data

Raw data는 `data/raw/DieCasting_Quality_Raw_Data_product1.csv`에 둡니다.

```bash
python -m src.data.prepare_data
```

생성 산출물:

- `data/processed/diecasting_product1_binary.csv`
- `data/processed/train.csv`
- `data/processed/valid.csv`
- `data/processed/test.csv`
- `artifacts/reports/data_profile.json`

## 2. Train Baseline With MLflow

```bash
python -m src.models.train_binary
```

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

Health check:

```bash
curl http://localhost:8000/health
```

Prediction example:

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

기본 URL:

```text
http://127.0.0.1:8501
```

## 5. Docker Demo

Docker는 production 배포가 아니라 **수업 제출용 MLOps 실행 구조 증거**로 사용합니다. 현재 Dockerfile은 FastAPI serving app을 실행하는 구조입니다.

학습 산출물(`artifacts/models`, `artifacts/reports`)이 생성된 뒤 Docker image를 빌드합니다. Dockerfile은 serving demo를 빠르게 만들기 위해 `requirements-api.txt`만 설치합니다.

```bash
docker build -t diecasting-api .
docker run -p 8000:8000 diecasting-api
```

컨테이너 실행 후 확인:

```bash
curl http://localhost:8000/health
```

> 현재 작업 PC에 Docker CLI가 없으면 실제 build 검증은 Docker Desktop이 설치된 환경에서 진행해야 합니다.

## Fixed Demo Samples

| Sample | Path | 예상 결과 |
| :--- | :--- | :--- |
| 정상 샘플 | `artifacts/reports/normal_request.json` | `normal` |
| 불량 샘플 | `artifacts/reports/defect_request.json` | `defect` |

현재 고정 샘플은 모델이 의도한 방향으로 예측하도록 선택되어 있습니다. 발표 직전에는 반드시 다시 확인합니다.

## Final Presentation Demo Path

1. 한 문장 문제정의 제시
2. 데이터 출처와 binary label 생성 규칙 설명
3. class distribution과 leakage 방지 설명
4. baseline metric 표 제시
5. MLflow UI에서 run, metric, artifact 확인
6. feature importance plot으로 XAI 설명
7. FastAPI `/health`, `/model-info`, `/predict` 호출
8. Streamlit UI에서 정상/불량 sample 예측
9. Dockerfile과 `docker build/run` 명령 설명
10. 한계와 Future Work 제시

## Role Handoff

### 김병근: Data

김병근 담당자는 **DataOps와 데이터 버전관리 준비**를 맡습니다. 데이터가 어디서 왔고, 어떤 기준으로 label을 만들었고, 어떤 split으로 평가할지 설명할 책임이 있습니다.

| 우선순위 | 작업 | 완료 기준 |
| :--- | :--- | :--- |
| 1 | 데이터 출처와 Product 1 선택 이유 정리 | 보고서 Data 섹션 초안 |
| 2 | 정상/불량 class 분포 plot 생성 | 발표용 class distribution 이미지 |
| 3 | 주요 feature 5~10개 분포 확인 | EDA notebook 또는 plot |
| 4 | 결측치/이상치 처리 여부 정리 | 데이터 품질 표 |
| 5 | leakage 방지 설명 작성 | 결함 컬럼 제거 문장과 표 |
| 6 | DVC 데이터 버전 전략 보완 | `docs/data_versioning.md` 업데이트 |

발표 문장:

> 원천 데이터에는 공정·센서 변수와 결함 여부 컬럼이 함께 들어 있습니다. 우리는 결함 컬럼을 이용해 정상/불량 label을 만든 뒤, 해당 결함 컬럼은 feature에서 제거해 label leakage를 방지했습니다.

### Zhang Xin: Modeling

Zhang Xin 담당자는 **최적화 전 baseline 모델 비교와 XAI 초안**을 맡습니다. 목표는 최종 튜닝 모델을 완성하는 것이 아니라, 여러 기본 모델을 같은 split과 metric으로 비교하여 어떤 모델을 tuning 후보로 넘길지 판단하는 것입니다.

| 우선순위 | 작업 | 완료 기준 |
| :--- | :--- | :--- |
| 1 | 현재 RF baseline 결과를 표와 문장으로 정리 | 보고서 Baseline 섹션 |
| 2 | Decision Tree와 RandomForest 비교 | tree ensemble baseline table |
| 3 | XGBoost와 LightGBM 기본 모델 비교 | boosting baseline table |
| 4 | confusion matrix와 classification report 해석 | 정상/불량 오류 유형 설명 |
| 5 | feature importance와 SHAP plot 생성 | XAI plot, top feature table |
| 6 | tuning 후보 모델 1~2개 추천 | 심재광에게 넘길 handoff note |

후보 실험:

| 후보 | 목적 |
| :--- | :--- |
| Logistic Regression | 가장 단순한 baseline 비교 |
| Decision Tree | 단일 tree 기준 모델과 해석성 확인 |
| RandomForest | 현재 baseline 및 XAI-friendly 모델 |
| XGBoost | boosting 계열 성능 비교 |
| LightGBM | 빠른 boosting baseline 비교 |
| CatBoost | 가능하면 추가 후보로 검토 |

발표 문장:

> Accuracy만 보면 정상 class가 많아 과대평가될 수 있기 때문에, 불량 class를 고려한 F1과 ROC-AUC를 중심으로 평가했습니다. 또한 불량을 놓치는 비용이 크므로 recall도 함께 확인했습니다.

### 심재광: MLOps/Serving

심재광은 **학습 이후 tuning, MLflow 기반 실험관리, MLOps/Serving 통합**을 맡습니다. Zhang Xin이 baseline 후보 비교와 XAI 초안을 만들면, 심재광은 그 결과를 받아 threshold/hyperparameter tuning을 수행하고, MLflow에 실험을 기록하며, 최종 serving candidate를 API와 UI로 연결합니다.

| 우선순위 | 작업 | 완료 기준 |
| :--- | :--- | :--- |
| 1 | Zhang이 넘긴 baseline 후보 중 tuning 대상 확정 | tuning 대상 모델명과 이유 |
| 2 | threshold tuning 또는 hyperparameter tuning 수행 | tuning 전/후 metric table |
| 3 | MLflow에 baseline/tuning 결과 기록 | MLflow UI screenshot |
| 4 | serving candidate 또는 champion tag 정리 | MLflow tag 증거 |
| 5 | 최종 model artifact를 FastAPI와 연결 | `/predict` 동작 |
| 6 | API Swagger 화면 확인 | `/docs` 화면 캡처 |
| 7 | Docker Desktop 환경에서 build/run 검증 | `/health` 동작 캡처 |
| 8 | README 실행 명령 최종 점검 | 새 환경에서 실행 가능 |
| 9 | API/UI demo script 작성 | 5분 발표 리허설 가능 |

발표 문장:

> week13 실습 구조를 따라 MLflow에 parameter, metric, artifact를 기록했고, week14 실습 구조를 따라 모델 artifact를 FastAPI endpoint로 감쌌습니다. Dockerfile도 포함해 컨테이너 실행 구조를 만들었습니다.

## Operation Risks

| 리스크 | 설명 | 현재 대응 | 추가 보완 |
| :--- | :--- | :--- | :--- |
| Label leakage | 결함 컬럼이 feature에 남으면 정답 유출 | 결함 컬럼 제거 | Data 담당자가 컬럼 목록 재확인 |
| Class imbalance | 정상 3,468 / 불량 739로 불균형 | F1, recall, ROC-AUC 사용 | threshold tuning |
| False negative | 불량을 정상으로 예측하는 경우 | recall 확인 | defect recall 중심 threshold 선택 |
| False positive | 정상을 불량으로 예측하는 경우 | precision 확인 | 운영 비용 관점 논의 |
| Drift | 공정 조건 분포가 변하면 성능 저하 | Future Work로 PSI/KS 제안 | drift simulation 가능 시 추가 |
| Demo failure | 발표 중 서버/API 오류 | 고정 샘플 준비 | 발표 전 로컬 리허설 |
| Docker 검증 | Docker CLI 없는 환경에서는 build 불가 | Dockerfile 작성 | Docker Desktop 환경에서 검증 필요 |

## Final Deliverables Checklist

| 제출물 | 담당 | 상태 | 해야 할 일 |
| :--- | :--- | :--- | :--- |
| 최종 보고서 PDF | 심재광, 전체 검토 | outline 있음 | Word/PDF 작성 |
| 발표자료 PDF | 심재광 | outline 있음 | slide 제작 |
| GitHub Repository | 심재광 | public repo 생성 완료 | README/문서 최종화 |
| 데이터/EDA 자료 | 김병근 | data profile 있음 | EDA plot 보강 |
| Baseline 모델 비교 | Zhang Xin | RF baseline 있음 | Decision Tree/RF/XGBoost/LightGBM 비교, XAI 초안 |
| Tuning/모델 최적화 | 심재광 | RF baseline 있음 | Zhang 결과 기반 threshold/hyperparameter tuning |
| MLflow evidence | 심재광 | `mlflow.db` 있음 | baseline/tuning run 비교, UI screenshot |
| FastAPI demo | 심재광 | 구현/검증 완료 | 발표 리허설 |
| Web UI demo | 심재광 | 구현/검증 완료 | 발표 리허설 |
| Docker evidence | 심재광 | Dockerfile 있음 | Docker 환경에서 build/run 확인 |
| 데모 영상 | 선택 | 미정 | 시간이 있으면 1080p 녹화 |

## Next Actions

### 김병근

1. `data_profile.json` 기반으로 데이터 설명 표 작성
2. class distribution plot 생성
3. 결함 컬럼 제거와 leakage 방지 설명 작성
4. DVC/data versioning 설명 검토

### Zhang Xin

1. RF baseline 결과 해석 작성
2. Decision Tree, RandomForest, XGBoost, LightGBM 기본 모델 비교
3. 모델별 F1, ROC-AUC, recall, precision, confusion matrix 정리
4. Feature importance와 SHAP 기반 XAI 초안 작성
5. tuning 후보 모델 1~2개를 심재광에게 전달

### 심재광

1. Zhang이 넘긴 후보 모델을 기준으로 threshold/hyperparameter tuning
2. tuning 전/후 결과를 MLflow에 기록
3. README 최종 실행 검증
4. MLflow/FastAPI/Streamlit screenshot 확보
5. Docker build 가능한 환경에서 검증
6. 보고서와 발표자료 초안 작성

## Final Message

이 프로젝트는 “다이캐스팅 불량 예측 모델을 하나 만들었다”가 아니라, 다음 메시지를 보여주는 것이 목표입니다.

> 제조 공정 데이터를 정상/불량 예측 문제로 정의하고, 데이터 버전관리·실험 추적·모델 artifact·XAI·API serving·간단 UI·Docker 실행 구조까지 연결하여 작은 MLOps 기반 AI 서비스를 구현했다.

발표와 보고서는 이 메시지를 중심으로 정리합니다.
