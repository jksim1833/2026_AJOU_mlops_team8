# Data Version Control Plan

본 프로젝트는 DVC를 완전한 remote storage까지 운영하지 않더라도, 데이터 lineage가 보이는 구조를 우선 적용한다.

## Data Versions

| Version | Path | Description |
| :--- | :--- | :--- |
| `raw_product1_v1` | `data/raw/DieCasting_Quality_Raw_Data_product1.csv` | KAMP Product 1 원천 공정/센서/결함 데이터 |
| `binary_product1_v2_dedup` | `data/processed/diecasting_product1_binary.csv` | 결함 컬럼 제거 후 완전히 동일한 row를 제거한 이진 라벨 데이터 |
| `split_product1_v2_dedup` | `data/processed/train.csv`, `valid.csv`, `test.csv` | exact row overlap이 없는 stratified 70/15/15 split |

## EDA Lineage

- EDA는 별도 데이터 복사본을 만들지 않고 `binary_product1_v2_dedup` 전체
  processed dataset에서 생성한다.
- `prepare_data` stage가 `data_profile.json`과 class distribution, feature
  distribution, boxplot, correlation heatmap을 함께 생성한다.
- 핵심 feature는 target과 상수 컬럼을 제외한 뒤 클래스 간 표준화 평균
  차이 상위 8개로 매 실행 결정한다.
- IQR 1.5 기준 이상치는 제거하지 않고 데이터 품질 후보로만 기록한다.
- 따라서 processed 데이터 버전이 동일하면 EDA 통계와 plot도 동일하게
  재생성된다.

## Leakage Policy

- 결함 컬럼은 `defect_label` 생성에만 사용한다.
- label 생성 후 `Short_Shot`, `Bubble`, `Crack` 등 모든 결함 컬럼은 feature에서 제거한다.
- scaler, feature selection, model fitting은 train split 내부 기준으로만 수행한다.
- 완전히 동일한 processed row는 split 전에 한 번만 남긴다.
- `data_profile.json`의 split 간 exact row overlap은 모두 0이어야 한다.

## DVC Commands

```bash
dvc init
dvc repro
dvc status
```

remote storage를 붙이는 경우:

```bash
dvc remote add -d localremote ../dvc-storage
dvc push
```

## Presentation Evidence

- `dvc.yaml`: 데이터 준비, baseline, leaderboard/XAI, champion tuning stage 정의
- `artifacts/reports/data_profile.json`: row/column/class/split과 EDA 품질 통계 기록
- `artifacts/plots/eda_*.png`: class, feature 분포, boxplot, 상관관계 증거
- `configs/params.yaml`: split seed, model params, artifact path 기록
