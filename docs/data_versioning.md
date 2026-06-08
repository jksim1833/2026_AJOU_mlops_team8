# Data Version Control Plan

본 프로젝트는 DVC를 완전한 remote storage까지 운영하지 않더라도, 데이터 lineage가 보이는 구조를 우선 적용한다.

## Data Versions

| Version | Path | Description |
| :--- | :--- | :--- |
| `raw_product1_v1` | `data/raw/DieCasting_Quality_Raw_Data_product1.csv` | KAMP Product 1 원천 공정/센서/결함 데이터 |
| `binary_product1_v1` | `data/processed/diecasting_product1_binary.csv` | 결함 컬럼 기반 정상/불량 이진 라벨 데이터 |
| `split_product1_v1` | `data/processed/train.csv`, `valid.csv`, `test.csv` | stratified 70/15/15 split |

## Leakage Policy

- 결함 컬럼은 `defect_label` 생성에만 사용한다.
- label 생성 후 `Short_Shot`, `Bubble`, `Crack` 등 모든 결함 컬럼은 feature에서 제거한다.
- scaler, feature selection, model fitting은 train split 내부 기준으로만 수행한다.

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

- `dvc.yaml`: 데이터 준비와 학습 stage 정의
- `artifacts/reports/data_profile.json`: 생성 데이터 row/column/class/split 기록
- `configs/params.yaml`: split seed, model params, artifact path 기록
