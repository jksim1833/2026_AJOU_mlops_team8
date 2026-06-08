# Notebook Plan

최종 보고서와 발표자료에는 아래 순서의 노트북 결과를 반영한다.

| Notebook | 목적 | 담당 |
| :--- | :--- | :--- |
| `01_eda.ipynb` | 데이터 출처, class 분포, 결측/이상치, 주요 변수 분포 | 김병근 |
| `02_baseline_binary.ipynb` | RandomForest baseline, F1/ROC-AUC, confusion matrix | Zhang Xin |
| `03_mlflow_experiments.ipynb` | MLflow run 비교, champion 선정 | Zhang Xin, 심재광 |
| `04_xai.ipynb` | feature importance/SHAP 기반 모델 해석 | 심재광 |

현재 실행 가능한 핵심 코드는 `src/data/prepare_data.py`와 `src/models/train_binary.py`에 있다.
