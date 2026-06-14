# Diecasting MLOps Handover

## 시작 지점

- 저장소: `https://github.com/jksim1833/2026_AJOU_mlops_team8.git`
- 작업 브랜치: `sim`
- 프로젝트 경로: `project/diecasting-mlops`
- 기준 문서: `project_brief.md`
- Python 환경: `.venv`, Python 3.10
- 데이터 버전: `binary_product1_v2_dedup`
- champion: `LogisticRegression`, model version `logistic_champion_v1`

새 thread에서는 먼저 다음을 실행해 현재 상태를 확인한다.

```powershell
git switch sim
git pull origin sim
git status
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m dvc status
```

## 완료된 구현

1. KAMP Product 1 원본에서 결함 컬럼을 이용해 binary label을 만들고 해당
   결함 컬럼을 feature에서 제거했다.
2. 완전히 동일한 processed row 1,692개를 제거하고 stratified 70/15/15
   split을 생성했다. split 간 exact overlap은 0건이다.
3. DVC pipeline은 다음 네 단계로 구성되어 있다.
   - `prepare_data`
   - `train_binary`
   - `compare_baselines_xai`
   - `tune_logistic`
4. Logistic Regression, Decision Tree, RandomForest, XGBoost, LightGBM을
   동일 split에서 비교하는 AutoML-lite leaderboard와 SHAP XAI가 구현됐다.
5. Logistic Regression을 5-fold CV, deterministic 30회 search로 튜닝하고
   validation probability에서 threshold를 선택했다.
6. MLflow champion run, FastAPI, Streamlit, 샘플 요청과 발표용 evidence가
   준비되어 있다.
7. Week 2 EDA 패턴을 `prepare_data`에 통합했다.
   - class distribution
   - 상위 8개 feature histogram/boxplot
   - correlation heatmap
   - 결측치, 상수 feature, IQR 이상치 후보
   - Streamlit `Data EDA` 탭

## 현재 핵심 결과

- processed rows: 2,515
- normal: 1,960
- defect: 555
- missing values: 0
- constant features: 8
- EDA 선택 feature:
  - `Factory_Humidity`
  - `Factory_Temp`
  - `Spray_Time`
  - `Casting_Pressure`
  - `Cylinder_Pressure`
  - `Pressure_Rise_Time`
  - `Spray_2_Time`
  - `Air_Pressure`
- decision threshold: `0.49`
- MLflow run ID: `c902f8c099e941e6ba2d3dc62a4cf3b1`
- validation: F1 `0.5344`, recall `0.8434`, ROC-AUC `0.7476`
- test: F1 `0.4884`, recall `0.7590`, ROC-AUC `0.7581`

## 검증 상태

- 전체 test: `12 passed`
- `python -m src.data.prepare_data`: 성공
- `dvc repro`: 성공
- `dvc status`: `Data and pipelines are up to date`
- processed CSV SHA-256은 EDA 추가 전후 동일하다.
- Streamlit `Data EDA` 탭 렌더링과 브라우저 console error 없음이 확인됐다.
- evidence:
  - `docs/evidence/mlflow_champion.png`
  - `docs/evidence/fastapi_swagger.png`
  - `docs/evidence/streamlit_champion_metrics.png`

## 로컬 실행

전체 pipeline을 재실행하고 서비스를 시작하려면:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1 -RunPipeline
```

이미 artifact가 있으면:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1 -SkipInstall
```

- MLflow: `http://127.0.0.1:5000`
- FastAPI Swagger: `http://127.0.0.1:8000/docs`
- Streamlit: `http://127.0.0.1:8501`

종료:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stop_local.ps1
```

## 남은 작업

1. Docker Desktop이 있는 환경에서 다음을 검증하고 `/health`, `/model-info`,
   정상/불량 `/predict` 증거를 저장한다.

   ```powershell
   docker build -t diecasting-api .
   docker run --rm -p 8000:8000 --name diecasting-api diecasting-api
   curl.exe http://localhost:8000/health
   curl.exe http://localhost:8000/model-info
   curl.exe -X POST http://localhost:8000/predict `
     -H "Content-Type: application/json" `
     --data-binary "@artifacts/reports/normal_request.json"
   ```

2. 최종 보고서 PDF와 발표자료 PDF를 작성한다.
3. EDA, leaderboard, tuning 전후 성능, MLflow champion, Swagger, Streamlit
   이미지를 발표자료에 배치한다.
4. 발표 직전에 normal/defect 고정 sample과 API/UI 흐름을 다시 리허설한다.
5. 선택 사항으로 데모 영상을 녹화한다.

## 중요 주의사항

- 현재 PC에는 Docker CLI가 설치되어 있지 않아 실제 Docker build/run은
  검증하지 못했다.
- `.gitignore`가 `data/raw`, `data/processed`, `artifacts`와 `mlflow.db`를
  제외한다.
- 현재 DVC remote도 설정되어 있지 않다. 따라서 새 PC에서 Git clone만
  하면 raw data, processed data, champion model artifact가 복원되지 않는다.
- 새 환경 재현을 제출 요건으로 삼는다면 다음 중 하나를 먼저 결정해야 한다.
  - DVC remote를 설정하고 `dvc push` 수행
  - release asset 또는 별도 storage로 serving artifact 제공
  - 허용 범위를 확인한 뒤 필요한 champion artifact만 Git LFS 등으로 관리
- Docker image는 build context의 `artifacts/models/model.joblib`,
  `artifacts/models/metadata.json`,
  `artifacts/reports/feature_importance.json`을 필요로 한다. 이 파일들이
  로컬에 없는 상태에서 build하면 실행 가능한 API image가 되지 않는다.
- 기존 champion 결과는 test를 한 번만 평가한 결과다. 보고서 작성 때문에
  test를 반복적으로 기준 삼아 추가 튜닝하지 않는다.

## 다음 thread 권장 첫 작업

Docker를 제외하고 코드 구현은 완료된 상태다. 다음 thread에서는
`project_brief.md`와 이 문서를 읽고, 먼저 제출물 재현성 전략(DVC remote
또는 artifact 배포 방식)을 정한 뒤 Docker 검증, 보고서, 발표자료 순서로
진행한다.
