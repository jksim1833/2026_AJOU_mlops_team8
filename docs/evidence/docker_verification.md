# Docker Serving Verification

Verified on 2026-06-14 with Docker Desktop Linux containers.

## Purpose

The week15 workshop treats Docker as optional, but requires reproducible
execution. This image packages the FastAPI serving code, pinned Python
dependencies, and the DVC-restored champion artifacts. It does not retrain the
model inside the container.

## Image and Container

- Image: `diecasting-api:logistic-champion-v1`
- Base runtime: Python `3.10-slim`
- Image size: approximately 154 MB
- Initial build context transfer: approximately 202 KB after `.dockerignore`
- Container name: `diecasting-api`
- Port mapping: host `8000` to container `8000`
- Docker health status: `healthy`
- Bind mounts or volumes: none

The image contains only:

- `src/api`
- `artifacts/models/model.joblib`
- `artifacts/models/metadata.json`
- `artifacts/reports/feature_importance.json`
- pinned API dependencies

## API Results

| Request | Result |
| :--- | :--- |
| `GET /health` | `status=ok` |
| `GET /model-info` | model `logistic_champion_v1`, data `binary_product1_v2_dedup`, threshold `0.49` |
| normal `POST /predict` | `normal`, normal probability `0.8937` |
| defect `POST /predict` | `defect`, defect probability `0.6982` |

All requests returned HTTP 200. Container logs confirmed the four requests.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m dvc pull
docker build -t diecasting-api:logistic-champion-v1 .
docker run --rm -d -p 8000:8000 `
  --name diecasting-api `
  diecasting-api:logistic-champion-v1
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/model-info
curl.exe -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  --data-binary "@artifacts/reports/normal_request.json"
curl.exe -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  --data-binary "@artifacts/reports/defect_request.json"
docker stop diecasting-api
```
