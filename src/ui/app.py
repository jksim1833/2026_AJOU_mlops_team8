from __future__ import annotations

import json
from pathlib import Path

import requests
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_INPUTS_PATH = ROOT / "artifacts" / "reports" / "sample_inputs.json"
API_URL = "http://localhost:8000"


st.set_page_config(page_title="Diecasting Defect Demo", layout="wide")
st.title("다이캐스팅 정상/불량 예측 데모")

st.caption("공정/센서 feature를 FastAPI `/predict` endpoint로 보내 정상/불량 예측과 주요 feature를 확인합니다.")

if not SAMPLE_INPUTS_PATH.exists():
    st.error("sample_inputs.json이 없습니다. 먼저 데이터 준비와 모델 학습을 실행하세요.")
    st.stop()

samples = json.loads(SAMPLE_INPUTS_PATH.read_text(encoding="utf-8"))
sample_name = st.selectbox("샘플 입력 선택", list(samples.keys()))
payload = {"features": samples[sample_name]}

with st.expander("요청 JSON", expanded=False):
    st.json(payload)

if st.button("예측 실행", type="primary"):
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"API 호출 실패: {exc}")
        st.stop()

    result = response.json()
    label = "불량" if result["prediction"] == 1 else "정상"
    st.metric("예측 결과", label)
    col1, col2 = st.columns(2)
    col1.metric("정상 확률", f"{result['probability']['normal']:.3f}")
    col2.metric("불량 확률", f"{result['probability']['defect']:.3f}")

    st.subheader("주요 판단 변수")
    st.dataframe(result["top_features"], use_container_width=True)

    st.subheader("응답 JSON")
    st.json(result)
