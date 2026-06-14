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
10. 최종 보고서(Word/PDF)와 발표자료(PPTX/PDF, 11장) 및 발표 대본을
    제작했다. 모든 수치는 `artifacts/reports/*.json`과
    `docs/evidence/docker_verification.md` 원본 기준으로 통일했다. 상세는
    아래 "제출물" 섹션 참고.

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

## 제출물 (보고서·발표자료)

### 산출물 파일

- 최종 보고서: `docs/diecasting_mlops_final_report.docx` / `.pdf` (한국어,
  약 23페이지). 7개 장 + 표지/목차/요약. §3 MLOps 구조는 Git/DVC/MLflow/
  Docker의 동작 원리·설계 의도·재현성 보장까지 상세화하고 §3.5
  "End-to-End 재현성 보장"을 신설했다.
- 발표자료: `docs/diecasting_mlops_presentation.pptx` (11장) / `.pdf`.
  보고서와 동일한 네이비 톤. PPTX에는 발표 노트를 넣지 않았다.
- 발표 대본: `docs/diecasting_mlops_presentation_script.md` (슬라이드별
  구어체 대본, 목표 팀당 약 8분, 예상 질문 대응 포함).
- 생성 스크립트: `scripts/build_report.js` (보고서),
  `scripts/build_slides.js` (발표자료). 수정 후 재실행만 하면 문서가
  다시 생성된다.

### 발표자료 슬라이드 구성 (11장)

1. 표지
2. 문제 정의와 목표
3. 데이터 이해와 전처리
4. 데이터 파이프라인 — DVC 4단계 실행 (★핵심, 3열 표로 단계·명령·산출물)
5. MLOps 4계층 & 재현성 (★핵심, Git/DVC/MLflow/Docker 2×2 카드)
6. 모델 실험 — 어떻게 비교했나 (★핵심, 실험 절차 5단계 + Val F1 차트)
7. 튜닝 & 실험 추적 (MLflow)
8. XAI — 모델 해석 (SHAP)
9. Serving — FastAPI & Streamlit
10. Docker 배포 & 재현성
11. 결론 & Future Work

slide 4·5·6이 이번 수업의 핵심이라 분량·시간을 가장 크게 배정했다.
원래 한 장이던 파이프라인 슬라이드를 4(DVC 4단계 실행)와 5(4계층·재현성)
두 장으로 분리해 과정을 상세히 설명한다.

### 문서 생성·변환 방법

- Node 전역 패키지 `docx`, `pptxgenjs`를 사용한다. 실행 시
  `$env:NODE_PATH = (npm root -g)`를 설정한 뒤 `node scripts/build_report.js`,
  `node scripts/build_slides.js`를 돌린다.
- 이 PC에는 LibreOffice가 없고 docx/pptx skill의 `soffice.py`는 Windows를
  지원하지 않는다(AF_UNIX 오류). 그래서 PDF 변환은 **Microsoft Word/
  PowerPoint COM 자동화**로 수행했다. Word는 PDF 저장 시 TOC(목차)도
  갱신한다. 변환 전 잔류 `WINWORD`/`POWERPNT` 프로세스를 종료해야 파일
  잠금(EBUSY)을 피한다.
- 보고서 docx 검증은 skill의 `validate.py`를 `PYTHONUTF8=1`로 실행해야
  한다(한국어 Windows 로케일 cp949에서 UTF-8 XML 디코딩 오류 방지).

### 알게 된 점 / 주의

- `docs/evidence`의 `mlflow_champion.png`, `fastapi_swagger.png`,
  `streamlit_champion_metrics.png`, `docker_fastapi_swagger.png`는 **확장자만
  `.png`이고 실제 내용은 JPEG**다. 두 빌드 스크립트는 파일 시그니처로
  포맷을 자동 감지해 임베드한다. `type`을 `png`로 강제하면 Word가 파일을
  열지 못하므로 변경하지 말 것.
- 표지/문서의 과목명·교수명·팀 번호·학번은 repo명에서 **추정한 값**이다
  (아주대학교 · MLOps · 8조). 최종 제출 전 실제 정보로 확정해야 한다.
- 보고서 본문 수치는 `metrics.json`/`baseline_comparison.json`/
  `data_profile.json`/`metadata.json`/`docker_verification.md` 기준으로
  통일했다. 결함 컬럼 제거 수는 26개, Docker image size는 약 154MB가
  최신 원본 기준이다.

## 남은 작업

1. (완료) 최종 보고서 Word/PDF 작성 — §3 MLOps 구조 상세화 포함.
2. (완료) 발표자료 PPTX/PDF(11장)와 발표 대본 작성, EDA·leaderboard·
   tuning·MLflow·Swagger·Streamlit·Docker evidence 배치.
3. 표지·문서의 과목명/교수명/팀·학번 등 메타데이터를 확정하고 보고서·
   발표자료 양쪽에 반영한다.
4. API/UI/Docker 고정 sample demo를 리허설한다(발표 대본의 시연 팁 참고).
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

코드 구현, local DVC remote, Docker serving 검증, 최종 보고서·발표자료·
발표 대본 초안까지 완료됐다. 다음 thread에서는 새 구현보다 **제출물
확정과 리허설, main 반영**에 집중한다.

권장 순서:

1. 이 문서의 "제출물" 섹션과 `project_brief.md`를 읽는다.
2. 현재 test/DVC/Docker 상태를 위 시작 명령으로 확인한다.
3. 표지·문서의 과목명/교수명/팀·학번 메타데이터를 확정하고, 필요한 값을
   `scripts/build_report.js`와 `scripts/build_slides.js`에서 고친 뒤 두
   스크립트를 다시 실행해 docx/pptx와 PDF를 재생성한다.
4. 발표 대본(`docs/diecasting_mlops_presentation_script.md`)으로 API/UI/
   Docker demo를 리허설한다.
5. `sim`을 `main`에 반영하고 GitHub 제출 상태(default branch·링크)를
   점검한다.
6. 선택 사항으로 데모 영상을 녹화한다.
