# Baseline and SHAP XAI Report

## 1. 목적

본 문서는 다이캐스팅 정상/불량 이진 분류 프로젝트에서 baseline 모델 비교와
SHAP 기반 XAI 산출 방식을 정리한 보고서이다.

프로젝트의 핵심 목표는 공정/센서 데이터를 입력받아 제품이 `normal`인지
`defect`인지 예측하고, 모델이 어떤 feature를 근거로 판단했는지 설명하는
것이다.

## 2. 현재 repository baseline

현재 repository에는 RandomForest 기반 serving artifact가 포함되어 있으며,
검증/테스트 성능은 다음과 같다.

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.781 | 0.427 | 0.712 | 0.534 | 0.828 |
| Test | 0.796 | 0.444 | 0.649 | 0.527 | 0.839 |

정상 데이터가 불량 데이터보다 많기 때문에 accuracy만으로 모델을 판단하지
않는다. 본 프로젝트에서는 불량 탐지 품질을 보기 위해 F1을 primary metric으로
사용하고, ROC-AUC, recall, precision을 함께 확인한다.

## 3. Baseline comparison 방법

Baseline comparison은 여러 후보 모델을 같은 train/validation/test split에서
비교하여 tuning 대상으로 넘길 모델을 선정하는 단계이다.

비교 대상 모델은 다음과 같다.

- Logistic Regression
- Decision Tree
- RandomForest
- XGBoost
- LightGBM

모델 선택 규칙은 다음과 같다.

1. validation F1이 가장 높은 모델을 우선 선택한다.
2. F1이 비슷하거나 같은 경우 validation ROC-AUC를 tie-breaker로 사용한다.
3. 선택된 모델은 `baseline_candidate.joblib`로 저장하고, 이후 tuning/serving
   후보로 넘긴다.

## 4. SHAP/XAI 적용 방식

SHAP 분석은 모든 후보 모델에 각각 수행하는 방식이 아니다.

먼저 baseline 후보 모델들을 학습/평가한 뒤, validation F1 기준으로 선택된
best candidate model 1개에 대해 SHAP 분석을 자동으로 수행한다.

산출되는 XAI 결과는 두 종류이다.

- Global explanation: 전체 test sample에서 어떤 feature가 평균적으로 중요한지
  설명한다.
- Local explanation: 특정 sample 하나가 왜 defect 또는 normal로 예측되었는지
  feature별 기여도를 설명한다.

SHAP 값 해석 기준은 다음과 같다.

- positive SHAP value: 예측을 defect 방향으로 밀어준다.
- negative SHAP value: 예측을 normal 방향으로 밀어준다.
- 절댓값이 클수록 해당 feature가 예측에 미친 영향이 크다.

## 5. 재현 실행 방법

원본 CSV 파일을 아래 경로에 배치한 뒤 실행한다.

```bash
data/raw/DieCasting_Quality_Raw_Data_product1.csv
```

개별 실행 명령은 다음과 같다.

```bash
python -m src.data.prepare_data
python -m src.models.compare_baselines_xai
```

Windows local demo 전체 실행은 다음 명령을 사용한다.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1 -RunPipeline
```

현재 local repository에는 원본 CSV가 포함되어 있지 않으므로, committed
RandomForest artifact 기준의 demo는 실행 가능하지만 최종 baseline comparison
결과와 SHAP plot은 raw CSV를 추가한 뒤 재생성해야 한다.

## 6. 주요 산출물

Baseline comparison 산출물:

- `artifacts/reports/baseline_metric_table.md`
- `artifacts/reports/baseline_comparison.csv`
- `artifacts/reports/candidate_handoff_note.md`
- `artifacts/models/baseline_candidate.joblib`
- `artifacts/models/baseline_candidate_metadata.json`

SHAP/XAI 산출물:

- `artifacts/reports/xai_feature_interpretation.md`
- `artifacts/reports/shap_local_explanation.md`
- `artifacts/plots/shap_summary_bar.png`
- `artifacts/plots/shap_beeswarm.png`
- `artifacts/plots/shap_waterfall_defect_sample.png`

## 7. 발표 시 핵심 설명

- 본 프로젝트는 다이캐스팅 제품의 정상/불량을 예측하는 binary classification
  문제이다.
- 결함 컬럼은 label 생성에만 사용하고 feature에서는 제거하여 label leakage를
  방지했다.
- Accuracy보다 F1, recall, precision, ROC-AUC를 함께 보는 이유는 class imbalance
  때문이다.
- SHAP은 후보 모델 비교 이후 선택된 best candidate model에 자동 적용된다.
- 최종 serving 전에는 선택된 candidate에 대해 threshold tuning 또는
  hyperparameter tuning을 추가로 수행하는 것이 적절하다.
