# Diecasting MLOps Handover

## 시작 지점

- 저장소: `https://github.com/jksim1833/2026_AJOU_mlops_team8.git`
- 작업 브랜치: `sim`
- Docker 적용 기준 커밋: `bc00f90` (`Validate Docker serving deployment`)
- 최신 인수인계 상태: `origin/sim` HEAD
- 프로젝트 경로: `project/diecasting-mlops`
- 기준 문서: `project_brief.md`
- Python 환경: `.venv`, Python 3.10
- 데이터 버전: `binary_product1_v2_dedup`
- champion: `LogisticRegression`, model version `logistic_champion_v1`
- Docker image: `diecasting-api:logistic-champion-v1`

새 thread에서는 먼저 다음을 실행해 현재 상태를 확인한다.

```powershell
git switch sim
git pull origin sim
git status
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m dvc status
.\.venv\Scripts\python.exe -m dvc status -c
docker ps --filter name=diecasting-api
```

GitHub default branch는 아직 `main`이지만 최신 통합 결과는 `sim`에 있다.
최종 제출 전에 `sim`을 `main`에 반영하고 default branch 내용을 확인해야
한다.

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
8. week03 강의의 local-folder 방식을 따라 DVC remote `localstorage`를
   구성하고 clean clone에서 `dvc pull` 복원을 검증했다.
9. week15의 실행 재현성 요구에 맞춰 champion FastAPI를 Docker image로
   만들고 실제 build/run/API 호출을 검증했다.

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
- DVC default remote: `localstorage`
- `dvc push`: 49개 객체 업로드 성공
- `dvc status -c`: cache와 remote 동기화 확인
- Docker image `diecasting-api:logistic-champion-v1`: build 성공
- Docker image size: 약 `146.8 MiB`
- Docker container: `healthy`
- Docker API: `/health`, `/model-info`, 정상/불량 `/predict` 모두 성공
- Docker normal sample: `normal`, normal probability `0.8937`
- Docker defect sample: `defect`, defect probability `0.6982`
- Docker container는 host volume 없이 image 자체 artifact로 실행된다.
- processed CSV SHA-256은 EDA 추가 전후 동일하다.
- Streamlit `Data EDA` 탭 렌더링과 브라우저 console error 없음이 확인됐다.
- evidence:
  - `docs/evidence/mlflow_champion.png`
  - `docs/evidence/fastapi_swagger.png`
  - `docs/evidence/streamlit_champion_metrics.png`
  - `docs/evidence/docker_fastapi_swagger.png`
  - `docs/evidence/docker_verification.md`

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

## Docker 적용 상태

### 프로젝트에서 Docker를 사용한 목적

week15 자료의 요구는 production cloud 배포 자체보다 실행 재현성과
안정적인 serving demo다. Docker는 API 코드, Python runtime, dependency,
champion model artifact를 하나의 image로 고정해 로컬 `.venv`와 독립적으로
동일한 추론 환경을 실행한다.

현재 image는 **학습용이 아니라 추론 전용**이다.

```text
DVC pull
  -> champion model artifact 복원
  -> Docker image build
  -> FastAPI container 실행
  -> /health, /model-info, /predict 제공
```

Image에 포함되는 항목:

- Python `3.10-slim`
- version이 고정된 pandas, numpy, scikit-learn, joblib, FastAPI
- `src/api`
- `artifacts/models/model.joblib`
- `artifacts/models/metadata.json`
- `artifacts/reports/feature_importance.json`

Image에 포함하지 않는 항목:

- raw/processed 학습 데이터
- DVC cache와 MLflow DB
- EDA, 학습, tuning 코드
- Streamlit UI
- 로컬 `.venv`

`model.joblib`에는 `StandardScaler + LogisticRegression` pipeline이
저장되어 있다. API는 입력 feature를 DataFrame으로 만든 뒤 모델 확률을
계산하고 metadata의 threshold `0.49`를 적용해 normal/defect를 반환한다.

### Docker 재현 명령

Docker build 전에 DVC artifact가 존재해야 한다.

```powershell
.\.venv\Scripts\python.exe -m dvc pull
docker build -t diecasting-api:logistic-champion-v1 .
docker run --rm -d -p 8000:8000 `
  --name diecasting-api `
  diecasting-api:logistic-champion-v1
```

확인:

```powershell
docker ps
docker logs diecasting-api
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/model-info
curl.exe -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  --data-binary "@artifacts/reports/normal_request.json"
curl.exe -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  --data-binary "@artifacts/reports/defect_request.json"
```

- Swagger: `http://127.0.0.1:8000/docs`
- 종료: `docker stop diecasting-api`
- 상세 증거: `docs/evidence/docker_verification.md`

작성 시점에는 `diecasting-api` container가 port `8000`에서 `healthy`
상태로 실행 중이다. 새 thread 시작 시 이미 실행 중이면 같은 이름으로
다시 `docker run`하지 말고 `docker ps`로 먼저 확인한다.

## 남은 작업

1. 최종 보고서 PDF를 작성한다.
2. 발표자료 PDF를 제작한다.
3. EDA, leaderboard, tuning 전후 성능, MLflow champion, Swagger,
   Streamlit, Docker evidence를 발표자료에 배치한다.
4. API/UI/Docker 고정 sample demo script를 작성하고 리허설한다.
5. 최신 `sim`을 GitHub `main`에 반영하고 public repository의 default
   branch와 제출 링크를 최종 확인한다.
6. 선택 사항으로 데모 영상을 녹화한다.

## 중요 주의사항

- Docker build/run은 현재 PC의 Docker Desktop Linux container 환경에서
  검증했다. cloud registry push와 원격 서버 배포는 프로젝트 범위 밖이다.
- 현재 Docker image는 추론 전용이다. 다른 PC로 image를 전달하면 Python
  환경 없이 FastAPI 추론은 가능하지만, 재학습/DVC/MLflow/Streamlit은
  실행할 수 없다.
- `.gitignore`가 `data/raw`, `data/processed`, `artifacts`와 `mlflow.db`를
  제외한다.
- week03 강의 방식에 따라 sibling 폴더
  `project/diecasting_dvc_remote_storage`를 DVC default remote
  `localstorage`로 설정했다.
- local remote는 GitHub에 포함되지 않는다. 다른 PC에서 Git clone만 하면
  복원되지 않으며, remote 폴더를 별도로 전달하거나 클라우드 remote로
  전환해야 한다. 공유 클라우드 remote는 프로젝트 범위에서 제외했다.
- Docker image는 build context의 `artifacts/models/model.joblib`,
  `artifacts/models/metadata.json`,
  `artifacts/reports/feature_importance.json`을 필요로 한다. 이 파일들이
  로컬에 없는 상태에서 build하면 실행 가능한 API image가 되지 않는다.
- 기존 champion 결과는 test를 한 번만 평가한 결과다. 보고서 작성 때문에
  test를 반복적으로 기준 삼아 추가 튜닝하지 않는다.

## 다음 thread 권장 첫 작업

코드 구현, local DVC remote, Docker serving 검증은 완료됐다. 다음
thread에서는 구현을 추가하기보다 제출물 완성에 집중한다.

권장 순서:

1. `project_brief.md`, 이 문서, `docs/final_report_outline.md`,
   `docs/presentation_outline.md`를 읽는다.
2. 현재 test/DVC/Docker 상태를 위 시작 명령으로 확인한다.
3. 최종 보고서 PDF를 작성한다.
4. 발표자료 PDF에 기존 evidence를 배치한다.
5. API/UI/Docker demo를 리허설한다.
6. `sim`을 `main`에 반영하고 GitHub 제출 상태를 점검한다.
