# Baseline and SHAP XAI Report

## 1. 보고서 목적

본 문서는 다이캐스팅 정상/불량 이진 분류 프로젝트에서 baseline 모델링과
SHAP 기반 XAI가 어떤 역할을 하는지 정리한다.

Baseline의 목적은 여러 기본 모델을 동일한 데이터 split과 동일한 metric으로
비교하여 후속 tuning 대상으로 넘길 후보 모델을 선정하는 것이다. XAI의 목적은
선정된 모델이 어떤 feature를 근거로 normal/defect를 판단했는지 설명하는 것이다.

## 2. Baseline 구현 내용

Baseline comparison은 다음 모델 후보를 같은 train/validation/test split에서
비교하도록 구성했다.

- Logistic Regression
- Decision Tree
- RandomForest
- XGBoost
- LightGBM

평가 metric은 accuracy, precision, recall, F1, ROC-AUC와 fit/predict 시간을 함께 사용한다.
정상 class가 더 많은 class imbalance 문제이므로, 모델 선정은 validation F1을
우선 기준으로 하고 validation ROC-AUC를 tie-breaker로 사용한다.

exact dedup split에서 AutoML-lite leaderboard 1위는 Logistic Regression이다.

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.672 | 0.384 | 0.819 | 0.523 | 0.743 |
| Test | 0.659 | 0.366 | 0.759 | 0.494 | 0.758 |

후속 MLOps 단계에서는 이 후보를 5-fold CV와 threshold tuning으로 최적화하고
MLflow champion 및 serving artifact로 연결한다.

## 3. SHAP/XAI 구현 내용

SHAP은 모든 후보 모델에 각각 적용하지 않는다. 먼저 baseline 후보 모델들을
비교한 뒤, validation F1 기준으로 선택된 best candidate model 1개에 대해
자동으로 SHAP 분석을 수행한다.

구현된 XAI 산출물은 다음 두 가지 설명을 포함한다.

- Global explanation: 전체 test sample 기준으로 평균적으로 중요한 feature를
  보여준다.
- Local explanation: 특정 sample 하나가 왜 defect 또는 normal로 예측되었는지
  feature별 기여도를 보여준다.

SHAP 값은 다음처럼 해석한다.

- positive SHAP value: 예측을 defect 방향으로 이동시킨다.
- negative SHAP value: 예측을 normal 방향으로 이동시킨다.
- 절댓값이 클수록 해당 feature가 예측에 더 크게 기여했다.

## 4. 주요 산출물

Baseline 산출물:

- `artifacts/reports/baseline_metric_table.md`
- `artifacts/reports/baseline_comparison.csv`
- `artifacts/reports/candidate_handoff_note.md`
- `artifacts/models/baseline_candidate.joblib`
- `artifacts/models/baseline_candidate_metadata.json`

XAI 산출물:

- `artifacts/reports/xai_feature_interpretation.md`
- `artifacts/reports/shap_local_explanation.md`
- `artifacts/plots/shap_summary_bar.png`
- `artifacts/plots/shap_beeswarm.png`
- `artifacts/plots/shap_waterfall_defect_sample.png`

## 5. 재현 방법

원본 CSV를 다음 경로에 배치한 뒤 pipeline을 실행하면 baseline comparison과
SHAP 산출물을 재생성할 수 있다.

```text
data/raw/DieCasting_Quality_Raw_Data_product1.csv
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1 -RunPipeline
```
