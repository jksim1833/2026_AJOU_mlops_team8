# Data and DVC Summary

## 1. 담당 범위

본 문서는 다이캐스팅 품질 예측 프로젝트에서 데이터 준비 및 DVC 기반 데이터 버전 관리 작업을 정리한다.

Data 담당 범위는 다음과 같다.

* raw 데이터 구조 확인
* defect_label 생성 기준 정리
* label leakage 방지
* train/valid/test split 생성
* processed 데이터 생성
* DVC pipeline 재현 확인
* 데이터 버전 관리 방식 정리

## 2. 사용 데이터

project_brief.md 기준 raw 데이터는 다음 파일이다.

```text
data/raw/DieCasting_Quality_Raw_Data_product1.csv
```

데이터 구조는 다음과 같다.

```text
원천 row 수: 4207
exact dedup 후 row 수: 2515
제거된 완전 동일 row 수: 1692
전체 column 수: 29
target column: defect_label
```

## 3. defect_label 생성 기준

원본 데이터에는 제품 결함 여부를 나타내는 결함 컬럼들이 포함되어 있다.

defect_label 생성 기준은 다음과 같다.

```text
결함 컬럼 중 하나라도 1이면 defect_label = 1
모든 결함 컬럼이 0이면 defect_label = 0
```

defect_label 생성에 사용된 결함 컬럼은 feature에서 제거한다.

이유는 해당 결함 컬럼을 feature로 그대로 사용하면 모델이 정답을 직접 보고 학습하는 label leakage가 발생할 수 있기 때문이다.

## 4. 전체 class 분포

project_brief 기준 전체 데이터의 class 분포는 다음과 같다.

```text
normal(0): 1960
defect(1): 555
```

정상 데이터가 불량 데이터보다 많은 class imbalance 구조이다.

## 5. Train / Valid / Test split

데이터는 stratified split 방식으로 분리한다.

분리 후 class 분포는 다음과 같다.

```text
train:
  normal(0): 1370
  defect(1): 389

valid:
  normal(0): 295
  defect(1): 83

test:
  normal(0): 295
  defect(1): 83
```

valid/test는 모델 평가용 데이터이므로 원본 class 비율을 유지한다. 세 split 사이의 완전히 동일한 row overlap은 모두 0건이다.

## 6. Oversampling 적용 여부

현재 project_brief 기준 pipeline에서는 oversampling을 적용하지 않는다.

class imbalance는 class weight, F1/recall/ROC-AUC, validation threshold tuning으로 관리한다.

추후 oversampling 실험이 필요하면 train 데이터에만 선택적으로 적용해야 한다.

valid/test에는 oversampling을 적용하면 안 된다.

## 7. 생성되는 processed 데이터

prepare_data stage 실행 후 생성되는 파일은 다음과 같다.

```text
data/processed/diecasting_product1_binary.csv
data/processed/train.csv
data/processed/valid.csv
data/processed/test.csv
artifacts/reports/data_profile.json
```

각 파일의 의미는 다음과 같다.

| 파일                             | 설명                                          |
| ------------------------------ | ------------------------------------------- |
| diecasting_product1_binary.csv | defect_label이 생성되고 leakage 컬럼이 제거된 전체 데이터   |
| train.csv                      | 학습용 데이터                                     |
| valid.csv                      | 검증용 데이터                                     |
| test.csv                       | 테스트용 데이터                                    |
| data_profile.json              | 데이터 row 수, class 분포, 제거된 결함 컬럼, split 분포 요약 |

## 8. DVC 실행 방법

데이터 준비 단계만 재현하려면 다음 명령을 사용한다.

```bash
dvc repro prepare_data
```

전체 pipeline을 재현하려면 다음 명령을 사용한다.

```bash
dvc repro
```

DVC 상태 확인은 다음 명령으로 수행한다.

```bash
dvc status
```

정상 상태에서는 다음과 유사한 메시지가 출력된다.

```text
Data and pipelines are up to date.
```

## 9. Git과 DVC 관리 기준

다음 파일들은 Git이 직접 추적하지 않고 DVC 또는 로컬 실행 결과로 관리한다.

```text
data/raw/*.csv
data/processed/*.csv
artifacts/models/*
artifacts/plots/*
artifacts/reports/*
artifacts/examples/*
mlflow.db
mlruns/
mlartifacts/
```

Git에는 코드, DVC metadata, dvc.yaml, dvc.lock, 문서 파일을 중심으로 올린다.

## 10. 새 데이터가 들어왔을 때 처리 절차

추후 새 raw 데이터가 추가되면 다음 순서로 처리한다.

```text
1. 새 raw CSV를 data/raw에 배치
2. 컬럼 구조와 결함 컬럼 존재 여부 확인
3. prepare_data.py 실행 가능 여부 확인
4. dvc repro prepare_data 실행
5. data_profile.json에서 class 분포와 split 분포 확인
6. dvc status 확인
7. 필요한 경우 DVC metadata 갱신
8. Git commit 생성
9. PR 생성
```

## 11. 요약

본 데이터 파이프라인은 project_brief.md 기준 raw 데이터를 사용하여 defect_label을 생성하고, label leakage를 방지하기 위해 결함 컬럼을 feature에서 제거한다.

이후 stratified split으로 train/valid/test를 생성하며, DVC를 통해 데이터 준비 과정을 재현 가능하게 관리한다.
