# 발표자료 Outline

## Slide 1. Title

다이캐스팅 정상/불량 예측을 위한 MLOps 기반 AI 서비스

## Slide 2. Problem

공정 엔지니어가 공정/센서 데이터를 기반으로 제품이 정상인지 불량인지 빠르게 판단하고, 판단 근거를 확인해야 한다.

## Slide 3. Data

- KAMP 주조 품질보증 AI 데이터셋
- Product 1, 4,207 rows
- 정상 3,468 / 불량 739
- 결함 컬럼으로 binary label 생성 후 feature에서 제거

## Slide 4. Pipeline

Raw Data -> Binary Label -> DVC -> RF Baseline -> MLflow -> XAI -> FastAPI -> Streamlit/Docker

## Slide 5. Baseline Result

- Validation F1: 0.534
- Validation ROC-AUC: 0.828
- Test F1: 0.527
- Test ROC-AUC: 0.839

## Slide 6. MLflow Tracking

- params, metrics, artifacts 기록
- champion tag
- model/metadata/plot 저장

## Slide 7. XAI

- Top features: Factory_Humidity, Factory_Temp, Cycle_Time, Spray_Time, Coolant_Pressure
- 정상/불량 샘플별 입력값과 feature importance 비교

## Slide 8. API Demo

- `GET /health`
- `GET /model-info`
- `POST /predict`
- 정상 샘플과 불량 샘플 요청 결과

## Slide 9. Web UI + Docker

- Streamlit UI에서 sample 선택 후 API 호출
- Dockerfile과 `docker build`, `docker run` 명령으로 컨테이너 실행 구조 제시

## Slide 10. Risks & Next Steps

- class imbalance
- feature drift
- threshold tuning
- retraining pipeline
- model monitoring
