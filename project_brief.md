# 다이캐스팅 정상/불량 예측 MLOps 프로젝트 Brief

이 문서는 팀원 인수인계용 프로젝트 기준 문서이다.  
목표는 “모델 성능표 하나”가 아니라, **데이터 준비부터 모델 학습, 실험 추적, XAI, API/간단 UI, Docker 실행 구조까지 이어지는 작은 end-to-end MLOps 시스템**을 완성하는 것이다.

## 1. 한 문장 문제정의

**우리는 공정 엔지니어를 위해 KAMP 다이캐스팅 공정·센서 데이터를 사용하여 제품이 정상인지 불량인지 예측하고, XAI로 주요 판단 근거를 제공한다. 성공은 validation F1/ROC-AUC, MLflow 실험 기록, FastAPI+간단 UI 데모, Docker 실행 구조로 확인한다.**

### 프로젝트를 이렇게 정의한 이유

| 요소 | 본 프로젝트의 답 |
| :--- | :--- |
| 대상 사용자 | 다이캐스팅 공정 엔지니어 |
| 사용자 문제 | 생산 shot의 공정/센서 조건을 보고 정상/불량 가능성을 빠르게 판단하고 싶다 |
| 입력 데이터 | KAMP Product 1 다이캐스팅 공정·센서 변수 |
| 모델 출력 | `normal` 또는 `defect` 이진 분류 결과와 확률 |
| 설명 출력 | 예측에 영향을 크게 준 주요 feature와 입력값 |
| 성공 기준 | F1/ROC-AUC 성능, MLflow 기록, API/UI 데모, Docker 구조 |

## 2. 최종 목표와 MVP 범위

### 최종 목표

공정 엔지니어가 샘플 공정 데이터를 입력하면 모델이 정상/불량을 예측하고, 주요 판단 변수까지 보여주는 작은 AI 서비스를 만든다. 이 과정에서 Git/DVC/MLflow/FastAPI/Docker를 연결해 MLOps 수업에서 배운 전 과정을 프로젝트 형태로 보여준다.

### MVP 범위

| 포함 | 제외 또는 Future Work |
| :--- | :--- |
| 정상/불량 binary classification | 주요 불량/치명 불량 3-class 분류 |
| Product 1 데이터만 사용 | Product 2까지 포함한 통합 모델 |
| AutoML-lite 후보 비교와 Logistic champion | 복잡한 딥러닝 모델 |
| MLflow local tracking | 운영용 MLflow server/registry 고도화 |
| FastAPI `/predict` | 실시간 공장 설비 연동 |
| Streamlit 간단 UI | 대시보드형 웹앱 고도화 |
| Dockerfile 작성 | 클라우드 배포, CI/CD |
| feature importance 기반 XAI | 완전한 SHAP 운영 대시보드 |

범위를 이렇게 줄이는 이유는 week15 자료의 핵심 메시지와 같다. 최종 발표에서는 큰 제품보다 **작더라도 실제로 동작하는 end-to-end demo path**가 더 중요하다.

## 3. Week15 팀별 제출 체크 답변

week15 master notebook의 “오늘 끝나기 전 팀별 제출 체크” 항목에 대한 우리 팀의 현재 답변이다.

| 체크 항목 | 우리 팀 답변 | 현재 상태 |
| :--- | :--- | :--- |
| 한 문장 문제정의 | 공정 엔지니어를 위해 KAMP 공정·센서 데이터로 정상/불량을 예측하고 XAI 근거를 제공한다 | 확정 |
| 데이터/label/split 전략 | Product 1 raw data에서 결함 컬럼으로 `defect_label` 생성, 결함 컬럼 제거, stratified 70/15/15 split | 1차 구현 완료 |
| baseline 모델과 metric | 5개 후보 AutoML-lite leaderboard, primary metric은 F1, 보조 metric은 ROC-AUC/recall/precision/실행 시간 | 완료 |
| MLOps 구조 | Git/GitHub repo 구조, DVC pipeline, MLflow tracking, FastAPI API, Streamlit UI, Dockerfile | 구조 구현 완료 |
| 최종 발표 demo path | sample input -> FastAPI `/predict` -> normal/defect probability -> top features -> Streamlit 화면 -> Dockerfile 설명 | 1차 구현 완료 |

## 4. 데이터, Label, Split 전략

### 데이터 출처

| 항목 | 내용 |
| :--- | :--- |
| 데이터셋 | KAMP 주조 품질보증 AI 데이터셋 |
| 사용 범위 | Product 1 |
| raw path | `data/raw/DieCasting_Quality_Raw_Data_product1.csv` |
| processed path | `data/processed/diecasting_product1_binary.csv` |
| 전체 row 수 | 원천 4,207 / exact dedup 후 2,515 |
| feature 수 | target 포함 29 columns, 학습 feature는 target 제외 28개 |

### Binary label 생성 규칙

원천 데이터에는 Cavity 1/2의 여러 결함 컬럼이 포함된다. 본 프로젝트에서는 정상/불량만 분류하므로 다음 규칙으로 target을 만든다.

```text
if any defect column == 1:
    defect_label = 1  # 불량
else:
    defect_label = 0  # 정상
```

label 생성에 사용한 결함 컬럼은 feature에서 반드시 제거한다.  
이유는 결함 컬럼이 모델 입력에 남아 있으면 모델이 공정/센서 조건을 학습하는 것이 아니라 정답 정보를 그대로 보게 되는 **label leakage**가 발생하기 때문이다.

### 제거된 결함 컬럼

`Short_Shot_1`, `Bubble_1`, `Exfoliation_1`, `Blow_Hole_1`, `Stain_1`, `Dent_1`, `Deformation_1`, `Contamination_1`, `Impurity_1`, `Crack_1`, `Scratch_1`, `Buring_Mark_1`, `Inclusions_1`, `Short_Shot_2`, `Bubble_2`, `Exfoliation_2`, `Blow_Hole_2`, `Stain_2`, `Dent_2`, `Deformation_2`, `Contamination_2`, `Impurity_2`, `Crack_2`, `Scratch_2`, `Buring_Mark_2`, `Inclusions_2`

### 현재 class 분포

| Class | 의미 | Count |
| :--- | :--- | ---: |
| 0 | 정상 | 1,960 |
| 1 | 불량 | 555 |

정상 데이터가 훨씬 많으므로 accuracy만 보면 안 된다. 불량 탐지 성능을 확인하기 위해 F1, recall, ROC-AUC를 함께 본다.

### Split 전략

현재 구현은 stratified split을 사용한다.

| Split | 정상 | 불량 |
| :--- | ---: | ---: |
| Train | 1,370 | 389 |
| Validation | 295 | 83 |
| Test | 295 | 83 |

기본 비율은 `train/validation/test = 70/15/15`이다.  
완전히 동일한 processed row 1,692개를 split 전에 제거했으며 split 간 exact row overlap은 모두 0건이다. 향후 shot 시간, batch, lot 정보가 명확히 확인되면 group/time-aware split으로 개선할 수 있다.

## 5. Baseline 모델과 현재 Metric

### 현재 serving champion

| 항목 | 내용 |
| :--- | :--- |
| Champion model | `StandardScaler + LogisticRegression` |
| Target | `defect_label` binary classification |
| Primary metric | F1-score |
| Secondary metrics | ROC-AUC, recall, precision, accuracy, confusion matrix |
| 모델 파일 | `artifacts/models/model.joblib` |
| Metadata | `artifacts/models/metadata.json` |
| Metrics | `artifacts/reports/metrics.json` |

### 현재 champion 결과

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.677 | 0.391 | 0.843 | 0.534 | 0.748 |
| Test | 0.651 | 0.360 | 0.759 | 0.488 | 0.758 |

### 결과 해석

- exact dedup 후 성능은 기존 중복 포함 평가보다 낮아졌지만 데이터 누수 가능성을 제거한 신뢰 가능한 결과다.
- Logistic baseline validation F1 0.523에서 CV와 threshold tuning 후 0.534로 개선되었다.
- 최종 threshold는 0.49이며 불량 recall은 validation 0.843, test 0.759이다.
- Precision은 낮은 편이다. 즉, 불량이라고 예측했지만 실제 정상인 경우가 존재한다. 공정 운영에서는 false alarm 비용과 defect miss 비용 중 어느 쪽을 더 중요하게 볼지 논의가 필요하다.

### 역할 경계: baseline 비교와 tuning을 분리한다

이번 프로젝트에서 모델링 작업은 두 단계로 나눈다.

| 단계 | 담당 | 목적 | 범위 |
| :--- | :--- | :--- | :--- |
| Baseline 후보 비교 | Zhang Xin | 어떤 모델군이 이 데이터에 적합한지 확인 | Logistic Regression, Decision Tree, RandomForest, XGBoost/LightGBM 등 기본 설정 비교 |
| 학습 이후 tuning/실험관리 | 심재광 | 선택된 후보를 MLOps 관점에서 추적·최적화 | threshold 조정, hyperparameter tuning, MLflow run 관리, champion tag |

따라서 Zhang Xin은 “최종 튜닝 완료 모델”을 책임지는 것이 아니라, **최적화 전 baseline 모델 후보들의 성능 비교와 XAI 초안을 만드는 역할**에 집중한다. 이후 심재광이 tuning, MLflow tracking, serving candidate 선정으로 이어받는다.

### Zhang Xin이 이어서 구체화할 일

| 우선순위 | 작업 | 산출물 |
| :--- | :--- | :--- |
| 1 | 현재 RF baseline metric 재확인 및 결과표 정리 | `baseline_metric_table.md` 또는 보고서 표 |
| 2 | Decision Tree, RandomForest, XGBoost, LightGBM 등 기본 설정 모델 비교 | baseline comparison table |
| 3 | 각 모델의 F1, ROC-AUC, recall, precision, confusion matrix 정리 | baseline metric report |
| 4 | baseline 후보 중 tuning 대상으로 넘길 모델 1~2개 추천 | candidate handoff note |
| 5 | feature importance/SHAP 기반 XAI 초안 작성 | XAI plot, feature interpretation table |

### 심재광이 이어받을 tuning/MLOps 작업

| 우선순위 | 작업 | 산출물 |
| :--- | :--- | :--- |
| 1 | Zhang이 넘긴 후보 모델의 threshold 또는 hyperparameter tuning | tuning result table |
| 2 | tuning 전/후 결과를 MLflow run으로 기록 | MLflow comparison evidence |
| 3 | champion 또는 candidate_for_serving tag 설정 | MLflow tag screenshot |
| 4 | 최종 model artifact를 FastAPI serving과 연결 | updated model metadata |
| 5 | 발표자료에서 “baseline -> tuning -> serving” 흐름 정리 | MLOps pipeline slide |

## 6. MLOps 구조와 목표

본 프로젝트의 MLOps 목표는 다음 네 가지다.

1. 데이터와 label 생성 과정을 다시 실행할 수 있어야 한다.
2. 모델 실험의 parameter, metric, artifact가 남아야 한다.
3. 선택된 모델이 API로 호출 가능해야 한다.
4. 발표에서 Docker 기반 실행 구조를 보여줄 수 있어야 한다.

### 전체 구조

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

### 담당자별 참고 강의자료

| 담당 | 참고 자료 | 프로젝트에서 가져올 내용 |
| :--- | :--- | :--- |
| 김병근 | `Week-2-Data-Ops-EDA-and-Preprocessing.ipynb` | 데이터 로드, 결측치/이상치 확인, target 분포, feature 분포, EDA 리포트 작성 방식 |
| 김병근 | `week03 - mlops infra with code and data versioning.ipynb` | Git/DVC 구조, 데이터 파일을 Git에 직접 넣지 않는 이유, DVC stage/metadata 설명 |
| Zhang Xin | `week05 - classic ml tree ensemble and random forest.ipynb` | Decision Tree, RandomForest, feature importance, confusion matrix, classification report |
| Zhang Xin | `week06 - boosting evolution xgboost and lightgbm.ipynb` | XGBoost/LightGBM baseline, early stopping 개념, boosting 계열 비교 |
| Zhang Xin | `week07 - advanced ML II - SOTA boosting.pdf` | SOTA boosting 계열 모델을 후보군으로 고려하는 이유와 tuning 전 비교 관점 |
| Zhang Xin | `week10 - deep learning II NLP and transformer.pdf` | 복잡한 딥러닝/Transformer 계열은 본 tabular 문제의 MVP 범위 밖이라는 판단 근거 정리 |
| Zhang Xin | `week12 - xai feature importance and shap.ipynb` | Feature importance, SHAP bar/summary/waterfall, local explanation 문장 작성 |
| 심재광 | `week13 - mlops core experiment tracking.ipynb` | MLflow experiment/run, params/metrics/artifacts, candidate/champion tag |
| 심재광 | `week14 - model serving and deployment.ipynb` | FastAPI `/health`, `/predict`, model artifact loading, Dockerfile/README 제출 구조 |

### Repository 구조

```text
diecasting-mlops/
  README.md
  configs/params.yaml
  data/
    raw/
    processed/
  src/
    data/prepare_data.py
    models/train_binary.py
    api/main.py
    ui/app.py
  artifacts/
    models/
    plots/
    reports/
  docs/
    project_brief.md
    data_versioning.md
    final_report_outline.md
    presentation_outline.md
  dvc.yaml
  Dockerfile
  requirements.txt
  requirements-api.txt
```

### Git/GitHub 목표

| 목표 | 내용 |
| :--- | :--- |
| Public GitHub Repository | 최종 제출용 public repo 생성 |
| README | 문제정의, 실행 방법, API demo, Docker 명령 포함 |
| Commit 단위 | 데이터 준비, baseline, MLflow, API, UI, Docker, docs 단위로 나누기 |
| 주의 | raw/processed 대용량 데이터는 Git에 직접 넣지 않고 DVC 또는 별도 안내로 관리 |

### DVC 목표

`dvc.yaml`에는 네 stage가 정의되어 있다.

| Stage | Command | 역할 |
| :--- | :--- | :--- |
| `prepare_data` | `python -m src.data.prepare_data` | raw data에서 binary dataset과 split 생성 |
| `train_binary` | `python -m src.models.train_binary` | RF baseline artifact 생성 |
| `compare_baselines_xai` | `python -m src.models.compare_baselines_xai` | AutoML-lite leaderboard와 SHAP 생성 |
| `tune_logistic` | `python -m src.models.tune_logistic` | Logistic tuning, MLflow champion, serving artifact 생성 |

교수님께 설명할 포인트:

- 데이터 버전이 바뀌면 모델 결과도 달라진다.
- `dvc.yaml`로 데이터 준비와 학습 의존성을 명시했다.
- 실제 remote storage까지 완벽히 운영하지 않더라도 데이터 lineage를 보여주는 구조를 갖췄다.

### MLflow 목표

현재 `sqlite:///mlflow.db`를 local tracking store로 사용한다.  
baseline 비교 run과 `logistic_tuned_champion` run이 기록되어 있다. serving artifact metadata의 run ID와 MLflow champion run ID가 일치하며 이전 champion은 superseded 처리한다.

| 기록 항목 | 내용 |
| :--- | :--- |
| Experiment | `diecasting_binary_defect_classification` |
| Baseline Runs | `logistic_regression_baseline`, `decision_tree_baseline`, `random_forest_baseline`, `xgboost_baseline`, `lightgbm_baseline` |
| Champion Run | `logistic_tuned_champion` |
| Params | model type, hyperparameters, data version, target column |
| Metrics | validation/test accuracy, precision, recall, F1, ROC-AUC |
| Artifacts | model, metadata, metrics, confusion matrix, ROC curve, feature importance |
| Tags | `stage=baseline`, `stage=tuned`, `stage=champion`, `serving_candidate=true` |

발표에서 보여줄 증거:

- MLflow UI run 화면
- baseline/tuning metric 비교표
- artifact 목록
- champion 또는 candidate_for_serving tag

### Serving/API 목표

FastAPI는 모델이 노트북 안에만 있는 것이 아니라 외부에서 호출 가능한 서비스 형태임을 보여주는 장치다.

| Endpoint | Method | 설명 |
| :--- | :--- | :--- |
| `/health` | GET | API 상태 확인 |
| `/model-info` | GET | model version, data version, feature list, metric summary 확인 |
| `/predict` | POST | 공정/센서 feature 입력 후 정상/불량 예측 |

`/predict` 응답에는 다음이 포함된다.

- `prediction`: 0 또는 1
- `label_name`: `normal` 또는 `defect`
- `probability`: 정상/불량 확률
- `top_features`: 주요 feature importance와 입력값
- `model_version`
- `data_version`

### Web UI 목표

Streamlit UI는 발표자가 API 요청을 직접 타이핑하지 않고, 샘플을 선택해 예측 결과를 보여주기 위한 간단한 서비스 화면이다.

UI의 역할:

1. 정상/불량 sample 선택
2. API request JSON 확인
3. `/predict` 호출
4. 정상/불량 확률 표시
5. 주요 판단 변수 표시

### Docker 목표

Docker는 production 배포가 아니라 **수업 제출용 MLOps 실행 구조 증거**로 사용한다.

현재 Dockerfile은 FastAPI serving app을 실행하는 구조다.

```bash
docker build -t diecasting-api .
docker run -p 8000:8000 diecasting-api
```

주의: 현재 작업 PC에는 Docker CLI가 설치되어 있지 않아 실제 build 검증은 아직 못 했다.  
하지만 Dockerfile, `requirements-api.txt`, 실행 명령은 준비되어 있으므로 Docker Desktop이 설치된 환경에서 검증하면 된다.

## 7. 팀원별 역할과 인수인계

## 7.1 김병근: Data 담당

### 책임

김병근 담당자는 **DataOps와 데이터 버전관리 준비**를 맡는다. 데이터가 어디서 왔고, 어떤 기준으로 label을 만들었고, 어떤 split으로 평가할지 설명할 책임이 있다. 모델 코드를 많이 짜는 것보다 데이터의 신뢰성, EDA 근거, leakage 방지, DVC 적용 논리를 명확히 설명하는 것이 핵심이다.

### 참고할 수업 자료

| 자료 | 활용 포인트 |
| :--- | :--- |
| `Week-2-Data-Ops-EDA-and-Preprocessing.ipynb` | 데이터 로드, target 분포, 결측치, 수치형 분포, 이상치, 전처리 설계 |
| `week03 - mlops infra with code and data versioning.ipynb` | Git/DVC 역할 구분, raw/processed 데이터 버전관리, DVC stage 설명 |

### 현재 준비된 내용

| 항목 | 현재 상태 |
| :--- | :--- |
| raw data | `data/raw/DieCasting_Quality_Raw_Data_product1.csv` |
| processed data | `data/processed/diecasting_product1_binary.csv` |
| data profile | `artifacts/reports/data_profile.json` |
| label rule | 결함 컬럼 중 하나라도 1이면 불량 |
| split | stratified 70/15/15 |
| EDA | class 분포, 핵심 feature 8개 분포/boxplot, correlation heatmap 생성 완료 |
| data quality | 결측 0건, 상수 컬럼과 IQR 이상치 후보를 `data_profile.json`에 기록 |

### 해야 할 일

| 우선순위 | 작업 | 완료 기준 |
| :--- | :--- | :--- |
| 1 | 데이터 출처와 Product 1 선택 이유 정리 | 보고서 Data 섹션 초안 |
| 2 | 정상/불량 class 분포 plot 생성 | 완료: `artifacts/plots/eda_class_distribution.png` |
| 3 | 주요 feature 5~10개 분포 확인 | 완료: 통계 기준 상위 8개 분포/boxplot |
| 4 | 결측치/이상치 처리 여부 정리 | 완료: 결측 0건, IQR 이상치는 보고만 수행 |
| 5 | leakage 방지 설명 작성 | “결함 컬럼 제거” 문장과 표 |
| 6 | DVC 데이터 버전 전략 보완 | `docs/data_versioning.md` 업데이트 |

### 구체적으로 확인할 질문

| 질문 | 답해야 하는 내용 |
| :--- | :--- |
| 데이터는 어디서 왔는가? | KAMP 주조 품질보증 AI 데이터셋, Product 1 사용 |
| target은 어떻게 만들었는가? | 결함 컬럼 중 하나라도 1이면 defect, 모두 0이면 normal |
| 어떤 컬럼을 feature에서 제거했는가? | label 생성에 사용한 Cavity 1/2 결함 컬럼 전체 |
| class imbalance가 있는가? | dedup 후 정상 1,960 / 불량 555 |
| split은 어떻게 했는가? | stratified 70/15/15, train/valid/test class 분포 제시 |
| DVC로 무엇을 관리할 것인가? | raw, processed, split dataset, `dvc.yaml` pipeline |

### 발표에서 말할 문장

“원천 데이터에는 공정·센서 변수와 결함 여부 컬럼이 함께 들어 있습니다. 우리는 결함 컬럼을 이용해 정상/불량 label을 만든 뒤, 해당 결함 컬럼은 feature에서 제거해 label leakage를 방지했습니다.”

## 7.2 Zhang Xin: Modeling 담당

### 책임

Zhang Xin 담당자는 **최적화 전 baseline 모델 비교와 XAI 초안**을 맡는다. 목표는 최종 튜닝 모델을 완성하는 것이 아니라, 여러 기본 모델을 같은 split과 metric으로 비교하여 “어떤 모델을 tuning 후보로 넘길지” 판단하는 것이다. 또한 baseline 모델이 어떤 feature를 보고 판단하는지 feature importance와 SHAP으로 설명할 수 있게 만든다.

### 참고할 수업 자료

| 자료 | 활용 포인트 |
| :--- | :--- |
| `week05 - classic ml tree ensemble and random forest.ipynb` | Decision Tree, RandomForest, feature importance, confusion matrix |
| `week06 - boosting evolution xgboost and lightgbm.ipynb` | XGBoost, LightGBM, boosting baseline, early stopping 개념 |
| `week07 - advanced ML II - SOTA boosting.pdf` | CatBoost/boosting 계열을 후보로 고려하는 이유 |
| `week10 - deep learning II NLP and transformer.pdf` | 복잡한 딥러닝/Transformer를 이번 tabular MVP에서 제외하는 근거 |
| `week12 - xai feature importance and shap.ipynb` | SHAP bar/summary/waterfall plot, local explanation 작성 |

### 현재 준비된 내용

| 항목 | 현재 상태 |
| :--- | :--- |
| baseline leader | LogisticRegression |
| validation F1 | 0.523 |
| validation ROC-AUC | 0.743 |
| test F1 | 0.494 |
| test ROC-AUC | 0.758 |
| model artifact | `artifacts/models/model.joblib` |
| metrics artifact | `artifacts/reports/metrics.json` |

### 해야 할 일

| 우선순위 | 작업 | 완료 기준 |
| :--- | :--- | :--- |
| 1 | 현재 RF baseline 결과를 표와 문장으로 정리 | 보고서 Baseline 섹션 |
| 2 | Decision Tree와 RandomForest 비교 | tree ensemble baseline table |
| 3 | XGBoost와 LightGBM 기본 모델 비교 | boosting baseline table |
| 4 | confusion matrix와 classification report 해석 | 정상/불량 오류 유형 설명 |
| 5 | feature importance와 SHAP plot 생성 | XAI plot, top feature table |
| 6 | tuning 후보 모델 1~2개 추천 | 심재광에게 넘길 handoff note |

### 후보 실험 예시

| 후보 | 목적 |
| :--- | :--- |
| Logistic Regression | 가장 단순한 baseline 비교 |
| Decision Tree | 단일 tree 기준 모델과 해석성 확인 |
| RandomForest | 현재 baseline 및 XAI-friendly 모델 |
| XGBoost | boosting 계열 성능 비교 |
| LightGBM | 빠른 boosting baseline 비교 |
| CatBoost | 가능하면 추가 후보로 검토 |

### Zhang Xin의 범위 밖

| 범위 밖 작업 | 담당 |
| :--- | :--- |
| threshold tuning | 심재광 |
| Optuna/GridSearch/RandomizedSearch 기반 최적화 | 심재광 |
| MLflow run 구조 정리와 champion tag | 심재광 |
| FastAPI serving 연결 | 심재광 |

### 발표에서 말할 문장

“Accuracy만 보면 정상 class가 많아 과대평가될 수 있기 때문에, 불량 class를 고려한 F1과 ROC-AUC를 중심으로 평가했습니다. 또한 불량을 놓치는 비용이 크므로 recall도 함께 확인했습니다.”

## 7.3 심재광: MLOps/Serving 담당

### 책임

심재광은 **학습 이후 tuning, MLflow 기반 실험관리, MLOps/Serving 통합**을 맡는다. Zhang Xin이 baseline 후보 비교와 XAI 초안을 만들면, 심재광은 그 결과를 받아 threshold/hyperparameter tuning을 수행하고, MLflow에 실험을 기록하며, 최종 serving candidate를 API와 UI로 연결한다.

### 참고할 수업 자료

| 자료 | 활용 포인트 |
| :--- | :--- |
| `week13 - mlops core experiment tracking.ipynb` | MLflow experiment/run, params, metrics, artifacts, candidate/champion tag |
| `week14 - model serving and deployment.ipynb` | FastAPI endpoint, model artifact loading, Swagger/curl test, Dockerfile |

### 현재 준비된 내용

| 항목 | 현재 상태 |
| :--- | :--- |
| repo skeleton | 구현 완료 |
| MLflow | SQLite backend `mlflow.db` 생성 |
| API | FastAPI `/health`, `/model-info`, `/predict` 구현 |
| UI | Streamlit sample prediction 구현 |
| Docker | Dockerfile, `requirements-api.txt` 작성 |
| DVC | `dvc.yaml`, `.dvcignore`, data versioning doc 작성 |

### 해야 할 일

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

### 발표에서 말할 문장

“week13 실습 구조를 따라 MLflow에 parameter, metric, artifact를 기록했고, week14 실습 구조를 따라 모델 artifact를 FastAPI endpoint로 감쌌습니다. Dockerfile도 포함해 컨테이너 실행 구조를 만들었습니다.”

## 7.4 심재광: 발표/문서화 담당

### 책임

발표/문서화 역할은 마지막에 PPT만 만드는 것이 아니다. 문제정의, 데이터, baseline, 실험 추적, XAI, API demo가 하나의 이야기로 이어지도록 정리해야 한다.

### 해야 할 일

| 우선순위 | 작업 | 완료 기준 |
| :--- | :--- | :--- |
| 1 | 최종 보고서 목차 확정 | `docs/final_report_outline.md` 기반 |
| 2 | 발표자료 흐름 확정 | `docs/presentation_outline.md` 기반 |
| 3 | 팀원 산출물 취합 | EDA plot, metrics table, MLflow screenshot |
| 4 | 데모 영상 가능 여부 판단 | 시간이 되면 선택 제출 |
| 5 | 발표 리허설 | API/UI/Docker 설명 5분 내 완료 |

## 8. 현재 실행 방법

프로젝트 루트:

```bash
cd project/diecasting-mlops
```

### 환경 설치

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 데이터 준비

```bash
python -m src.data.prepare_data
```

### Baseline 학습 및 MLflow 기록

```bash
python -m src.models.train_binary
```

### MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

### FastAPI 실행

```bash
uvicorn src.api.main:app --reload --port 8000
```

확인:

```bash
curl http://localhost:8000/health
```

예측:

```bash
curl -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d @artifacts/reports/normal_request.json
```

### Streamlit UI 실행

FastAPI 서버를 먼저 실행한 뒤:

```bash
streamlit run src/ui/app.py
```

기본 URL:

```text
http://127.0.0.1:8501
```

### Docker 실행

Docker Desktop이 설치된 환경에서:

```bash
docker build -t diecasting-api .
docker run -p 8000:8000 diecasting-api
```

## 9. 최종 발표 Demo Path

발표 당일에는 아래 순서로 보여주는 것이 가장 안정적이다.

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

### 고정 데모 샘플

| Sample | Path | 예상 결과 |
| :--- | :--- | :--- |
| 정상 샘플 | `artifacts/reports/normal_request.json` | `normal` |
| 불량 샘플 | `artifacts/reports/defect_request.json` | `defect` |

현재 고정 샘플은 모델이 의도한 방향으로 예측하도록 선택되어 있다. 발표 직전에는 반드시 다시 확인한다.

## 10. 운영 리스크와 대응

| 리스크 | 설명 | 현재 대응 | 추가 보완 |
| :--- | :--- | :--- | :--- |
| Label leakage | 결함 컬럼이 feature에 남으면 정답 유출 | 결함 컬럼 제거 | Data 담당자가 컬럼 목록 재확인 |
| Class imbalance | dedup 후 정상 1,960 / 불량 555로 불균형 | class weight, F1, recall, ROC-AUC, threshold 0.49 | 운영 비용 기반 threshold 재검토 |
| False negative | 불량을 정상으로 예측하는 경우 | recall 확인 | defect recall 중심 threshold 선택 |
| False positive | 정상을 불량으로 예측하는 경우 | precision 확인 | 운영 비용 관점 논의 |
| Drift | 공정 조건 분포가 변하면 성능 저하 | Future Work로 PSI/KS 제안 | drift simulation 가능 시 추가 |
| Demo failure | 발표 중 서버/API 오류 | 고정 샘플 준비 | 발표 전 로컬 리허설 |
| Docker 검증 | 현재 PC에 Docker CLI 없음 | Dockerfile 작성 | Docker Desktop 환경에서 검증 필요 |

## 11. 최종 산출물 체크리스트

| 제출물 | 담당 | 상태 | 해야 할 일 |
| :--- | :--- | :--- | :--- |
| 최종 보고서 PDF | 심재광, 전체 검토 | outline 있음 | Word/PDF 작성 |
| 발표자료 PDF | 심재광 | outline 있음 | slide 제작 |
| GitHub Repository | 심재광 | local repo 구조 있음 | public repo 생성/push |
| 데이터/EDA 자료 | 김병근 | data profile 및 EDA plot 완료 | 발표 자료에 이미지 배치 |
| Baseline 모델 비교 | Zhang Xin | 5개 후보 leaderboard/XAI 완료 | 보고서 반영 |
| Tuning/모델 최적화 | 심재광 | Logistic CV/threshold tuning 완료 | 발표 근거 반영 |
| MLflow evidence | 심재광 | champion run과 screenshot 완료 | 발표자료 삽입 |
| FastAPI demo | 심재광 | 구현/검증 완료 | 발표 리허설 |
| Web UI demo | 심재광 | 구현/검증 완료 | 발표 리허설 |
| Docker evidence | 심재광 | Dockerfile 있음 | Docker 환경에서 build/run 확인 |
| 데모 영상 | 선택 | 미정 | 시간이 있으면 1080p 녹화 |

## 12. 다음 액션

### 김병근

1. `data_profile.json` 기반으로 데이터 설명 표 작성
2. 생성된 class distribution plot을 발표 자료에 배치
3. 결함 컬럼 제거와 leakage 방지 설명 작성
4. DVC/data versioning 설명 검토

### Zhang Xin

1. RF baseline 결과 해석 작성
2. Decision Tree, RandomForest, XGBoost, LightGBM 기본 모델 비교
3. 모델별 F1, ROC-AUC, recall, precision, confusion matrix 정리
4. Feature importance와 SHAP 기반 XAI 초안 작성
5. tuning 후보 모델 1~2개를 심재광에게 전달

### 심재광

1. GitHub public repository 생성
2. Zhang이 넘긴 후보 모델을 기준으로 threshold/hyperparameter tuning
3. tuning 전/후 결과를 MLflow에 기록
4. README 최종 실행 검증
5. MLflow/FastAPI/Streamlit screenshot 확보
6. Docker build 가능한 환경에서 검증
7. 보고서와 발표자료 초안 작성

## 13. 이 프로젝트의 최종 메시지

이 프로젝트는 “다이캐스팅 불량 예측 모델을 하나 만들었다”가 아니라, 다음 메시지를 보여주는 것이 목표다.

> 제조 공정 데이터를 정상/불량 예측 문제로 정의하고, 데이터 버전관리·실험 추적·모델 artifact·XAI·API serving·간단 UI·Docker 실행 구조까지 연결하여 작은 MLOps 기반 AI 서비스를 구현했다.

발표와 보고서는 이 메시지를 중심으로 정리한다.
