# 최종 보고서 Outline

## 1. 프로젝트 개요

- 한 문장 문제정의
- 대상 사용자: 공정 엔지니어
- 입력: KAMP Product 1 공정/센서 데이터
- 출력: 정상/불량 예측과 주요 판단 변수

## 2. 데이터 이해와 전처리

- 데이터 출처와 변수 구성
- binary label 생성 규칙
- 결함 컬럼 제거를 통한 leakage 방지
- class distribution과 train/valid/test split

## 3. MLOps 구조

- Git/GitHub repository 구조
- DVC 데이터 버전관리 계획
- MLflow experiment tracking 구조
- Docker 기반 serving 실행 구조

## 4. Baseline 및 모델 실험

- AutoML-lite baseline leaderboard와 실행 시간
- Logistic Regression hyperparameter/threshold tuning
- 주요 metric: F1, ROC-AUC, recall, precision, accuracy
- MLflow run 비교와 champion 선정 근거

## 5. XAI 및 오류 분석

- feature importance plot
- 정상/불량 예측 샘플별 주요 변수
- 모델 한계와 개선 방향

## 6. Serving 및 Demo

- FastAPI endpoint: `/health`, `/model-info`, `/predict`
- Streamlit UI demo
- Docker build/run 명령

## 7. 결론 및 Future Work

- 이번 프로젝트에서 구현한 end-to-end MLOps 흐름
- drift monitoring, retraining, AutoGluon/H2O 보조 검증 확장 계획
