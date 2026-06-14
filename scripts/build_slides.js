// Generate the 8-minute final presentation (Korean, navy theme) as .pptx
// Content distilled from the final report + code artifacts.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const PLOTS = path.join(ROOT, "artifacts", "plots");
const EVID = path.join(ROOT, "docs", "evidence");

// ---- palette / type ----
const NAVY = "1F4E79", BLUE = "2E75B6", ICE = "DCE7F2", ICE2 = "EEF3F8";
const AMBER = "E0A400", INK = "222222", MUTE = "666666", WHITE = "FFFFFF";
const KF = "맑은 고딕";
const W = 13.33, H = 7.5, M = 0.6;

// ---- image helpers (handle .png files that are actually JPEG) ----
function imgInfo(file) {
  const b = fs.readFileSync(file);
  if (b[0] === 0x89 && b[1] === 0x50) return { mime: "image/png", w: b.readUInt32BE(16), h: b.readUInt32BE(20), b };
  if (b[0] === 0xff && b[1] === 0xd8) {
    let i = 2;
    while (i < b.length) {
      if (b[i] !== 0xff) { i++; continue; }
      const m = b[i + 1];
      if (m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc)
        return { mime: "image/jpeg", h: b.readUInt16BE(i + 5), w: b.readUInt16BE(i + 7), b };
      i += 2 + b.readUInt16BE(i + 2);
    }
  }
  throw new Error("bad image " + file);
}
function imgData(file) { const { mime, b } = imgInfo(file); return mime + ";base64," + b.toString("base64"); }
// fit-and-center an image inside a box
function addPic(slide, file, bx, by, bw, bh) {
  const { w, h } = imgInfo(file);
  const r = Math.min(bw / w, bh / h);
  const dw = w * r, dh = h * r;
  slide.addImage({ data: imgData(file), x: bx + (bw - dw) / 2, y: by + (bh - dh) / 2, w: dw, h: dh });
}

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "AJOU MLOps Team 8";
pres.title = "다이캐스팅 정상/불량 예측 MLOps AI 서비스";

const shadow = () => ({ type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.18 });

// reusable: content-slide title with small amber motif square + page number
let pageNo = 0;
function contentTitle(slide, text, sub) {
  slide.background = { color: WHITE };
  slide.addShape(pres.shapes.RECTANGLE, { x: M, y: 0.45, w: 0.14, h: 0.52, fill: { color: AMBER } });
  slide.addText(text, { x: M + 0.28, y: 0.38, w: W - 2 * M - 0.3, h: 0.66, margin: 0, fontFace: KF, fontSize: 28, bold: true, color: NAVY, valign: "middle" });
  if (sub) slide.addText(sub, { x: M + 0.28, y: 1.0, w: W - 2 * M - 0.3, h: 0.35, margin: 0, fontFace: KF, fontSize: 13, color: MUTE, valign: "middle" });
  pageNo++;
  slide.addText(String(pageNo), { x: W - 0.9, y: H - 0.5, w: 0.5, h: 0.3, fontFace: KF, fontSize: 10, color: MUTE, align: "right" });
  slide.addText("다이캐스팅 MLOps · 8조", { x: M, y: H - 0.5, w: 4, h: 0.3, fontFace: KF, fontSize: 10, color: MUTE });
}
// stat callout card
function statCard(slide, x, y, w, h, big, label, fill = ICE2, bigColor = NAVY, labelColor = MUTE) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, rectRadius: 0.06, shadow: shadow() });
  slide.addText(big, { x, y: y + 0.12, w, h: h * 0.55, margin: 0, fontFace: KF, fontSize: 26, bold: true, color: bigColor, align: "center", valign: "middle" });
  slide.addText(label, { x: x + 0.1, y: y + h * 0.58, w: w - 0.2, h: h * 0.36, margin: 0, fontFace: KF, fontSize: 12, color: labelColor, align: "center", valign: "top" });
}

// =============================================================== Slide 1: Title
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.25, h: H, fill: { color: AMBER } });
  s.addText("MLOps 기말 프로젝트", { x: 1.1, y: 1.5, w: 11, h: 0.5, fontFace: KF, fontSize: 18, color: ICE });
  s.addText("다이캐스팅 정상/불량 예측을 위한\nMLOps 기반 AI 서비스", { x: 1.0, y: 2.05, w: 11.4, h: 1.9, fontFace: KF, fontSize: 40, bold: true, color: WHITE, lineSpacingMultiple: 1.05 });
  s.addText("End-to-End MLOps Pipeline for Die-Casting Defect Prediction", { x: 1.1, y: 4.05, w: 11, h: 0.5, fontFace: KF, fontSize: 16, italic: true, color: ICE });
  s.addShape(pres.shapes.LINE, { x: 1.1, y: 4.8, w: 5.5, h: 0, line: { color: AMBER, width: 2 } });
  s.addText([
    { text: "아주대학교 · MLOps · 8조", options: { breakLine: true, bold: true } },
    { text: "김병근 (Data) · Zhang Xin (Modeling/XAI) · 심재광 (MLOps/Serving)", options: { breakLine: true, fontSize: 14 } },
    { text: "2026-06-14", options: { fontSize: 13, color: ICE } },
  ], { x: 1.1, y: 5.0, w: 11, h: 1.4, fontFace: KF, fontSize: 16, color: WHITE, lineSpacingMultiple: 1.15 });
}

// =============================================================== Slide 2: Problem & Goal
{
  const s = pres.addSlide();
  contentTitle(s, "문제 정의와 목표");
  // one-line definition callout
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 1.35, w: W - 2 * M, h: 1.0, fill: { color: NAVY }, rectRadius: 0.06 });
  s.addText([
    { text: "공정·센서 데이터로 제품의 ", options: {} },
    { text: "정상/불량을 예측", options: { bold: true, color: AMBER } },
    { text: "하고, 그 ", options: {} },
    { text: "판단 근거(XAI)", options: { bold: true, color: AMBER } },
    { text: "를 함께 제시하는 AI 서비스", options: {} },
  ], { x: M + 0.3, y: 1.35, w: W - 2 * M - 0.6, h: 1.0, margin: 0, fontFace: KF, fontSize: 19, color: WHITE, valign: "middle" });
  // left: background problem
  s.addText("배경", { x: M, y: 2.7, w: 5.8, h: 0.4, fontFace: KF, fontSize: 17, bold: true, color: BLUE });
  s.addText([
    { text: "온도·압력·분사시간 등 다수 공정 변수가 품질에 영향", options: { bullet: true, breakLine: true } },
    { text: "변수 간 상호작용이 복잡 → 사람의 직관만으로 일관된 판단 곤란", options: { bullet: true, breakLine: true } },
    { text: "단순 성능 비교가 아닌 \"운영 가능한 AI 서비스\" 구축이 목표", options: { bullet: true } },
  ], { x: M, y: 3.1, w: 6.0, h: 2.6, fontFace: KF, fontSize: 15, color: INK, paraSpaceAfter: 10, lineSpacingMultiple: 1.05 });
  // right: I/O cards
  const rx = 7.0;
  s.addText("입·출력 / 대상", { x: rx, y: 2.7, w: 5.7, h: 0.4, fontFace: KF, fontSize: 17, bold: true, color: BLUE });
  const rows = [
    ["대상 사용자", "다이캐스팅 공정 엔지니어 / 품질관리자"],
    ["입력 (Input)", "28개 공정·센서 feature"],
    ["출력 (Output)", "정상/불량, 클래스 확률, 상위 기여 feature"],
    ["성공 기준", "F1/ROC-AUC · MLflow · API+UI · Docker"],
  ];
  let cy = 3.1;
  rows.forEach(([k, v]) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx, y: cy, w: 5.73, h: 0.62, fill: { color: ICE2 }, rectRadius: 0.05 });
    s.addText(k, { x: rx + 0.15, y: cy, w: 1.9, h: 0.62, margin: 0, fontFace: KF, fontSize: 13, bold: true, color: NAVY, valign: "middle" });
    s.addText(v, { x: rx + 2.05, y: cy, w: 3.5, h: 0.62, margin: 0, fontFace: KF, fontSize: 12.5, color: INK, valign: "middle" });
    cy += 0.72;
  });
}

// =============================================================== Slide 3: Data
{
  const s = pres.addSlide();
  contentTitle(s, "데이터 이해와 전처리", "KAMP 다이캐스팅 Product 1 · binary_product1_v2_dedup");
  // stat callouts
  statCard(s, M, 1.55, 2.7, 1.25, "4,207 → 2,515", "중복 1,692건 제거 후");
  statCard(s, M + 2.9, 1.55, 2.7, 1.25, "1,960 / 555", "정상 / 불량 (22.1%)");
  statCard(s, M + 5.8, 1.55, 2.7, 1.25, "70 / 15 / 15", "stratified split · overlap 0", ICE2, NAVY);
  // donut
  s.addChart(pres.charts.DOUGHNUT, [{ name: "클래스", labels: ["정상(normal)", "불량(defect)"], values: [1960, 555] }], {
    x: 9.3, y: 1.5, w: 3.4, h: 3.6, holeSize: 55, chartColors: [BLUE, AMBER],
    showPercent: true, showLegend: true, legendPos: "b", legendFontSize: 11, dataLabelColor: WHITE, dataLabelFontSize: 12, fontFace: KF,
  });
  // key points
  s.addText("전처리 핵심", { x: M, y: 3.0, w: 8.4, h: 0.4, fontFace: KF, fontSize: 17, bold: true, color: BLUE });
  s.addText([
    { text: "이진 라벨: 결함 컬럼 합 > 0 → defect(1), 아니면 normal(0)", options: { bullet: true, breakLine: true } },
    { text: "라벨 누출 방지: 결함 판정 26개 컬럼을 feature에서 제거", options: { bullet: true, breakLine: true } },
    { text: "중복 제거 후 stratified 분할, 3분할 간 완전중복 0건 검증", options: { bullet: true, breakLine: true } },
    { text: "결측치 0 · 상수 feature 8개 · EDA로 SMD 상위 8개 분석", options: { bullet: true, breakLine: true } },
    { text: "최상위 분리 변수: Factory_Humidity (SMD 0.715, 불량과 음의 상관)", options: { bullet: true } },
  ], { x: M, y: 3.4, w: 8.5, h: 3.3, fontFace: KF, fontSize: 14.5, color: INK, paraSpaceAfter: 9, lineSpacingMultiple: 1.04 });
}

// =============================================================== Slide 4a: DVC pipeline detail
{
  const s = pres.addSlide();
  contentTitle(s, "데이터 파이프라인 — DVC 4단계 실행", "dvc repro 한 줄로 순서 실행 · 입력 해시가 바뀐 단계와 그 이후만 다시 실행(증분)");
  const x0 = M, cardW = W - 2 * M;
  const cName = 4.0, cProc = 4.7, cOut = cardW - cName - cProc; // column widths inside card
  // column headers
  s.addText("단계 · 실행 명령", { x: x0 + 0.7, y: 1.42, w: cName, h: 0.3, margin: 0, fontFace: KF, fontSize: 12, bold: true, color: BLUE });
  s.addText("처리 내용", { x: x0 + 0.7 + cName, y: 1.42, w: cProc, h: 0.3, margin: 0, fontFace: KF, fontSize: 12, bold: true, color: BLUE });
  s.addText("산출물 (outs)", { x: x0 + 0.7 + cName + cProc, y: 1.42, w: cOut, h: 0.3, margin: 0, fontFace: KF, fontSize: 12, bold: true, color: BLUE });
  const stages = [
    ["prepare_data", "src.data.prepare_data", "라벨 생성 → 결함 26컬럼 제거(누출 방지)\n→ 중복 1,692건 제거 → stratified 70/15/15", "train/valid/test.csv\ndata_profile.json · EDA plots"],
    ["train_binary", "src.models.train_binary", "RandomForest baseline 학습(balanced)\nvalidation·test 평가 + MLflow 로깅", "rf_baseline.joblib\n지표·플롯 · MLflow run"],
    ["compare_baselines_xai", "src.models.compare_baselines_xai", "5개 후보 동일 split 비교 → Val F1 순위화\n최적 후보에 SHAP 전역·지역 해석", "baseline_comparison.json\nSHAP 산출물 · handoff note"],
    ["tune_logistic", "src.models.tune_logistic", "5-fold CV 30회 탐색 → threshold sweep(0.49)\n→ champion 등록·태깅", "model.joblib · metadata.json\nmetrics.json"],
  ];
  const ry0 = 1.8, rh = 1.16, rgap = 0.12;
  stages.forEach(([name, cmd, proc, out], i) => {
    const y = ry0 + i * (rh + rgap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x0, y, w: cardW, h: rh, fill: { color: i % 2 ? ICE2 : WHITE }, line: { color: "C9D6E3", width: 1 }, rectRadius: 0.05, shadow: shadow() });
    s.addShape(pres.shapes.OVAL, { x: x0 + 0.16, y: y + (rh - 0.46) / 2, w: 0.46, h: 0.46, fill: { color: NAVY } });
    s.addText(String(i + 1), { x: x0 + 0.16, y: y + (rh - 0.46) / 2, w: 0.46, h: 0.46, margin: 0, fontFace: KF, fontSize: 16, bold: true, color: WHITE, align: "center", valign: "middle" });
    // name + command
    s.addText([
      { text: name, options: { fontFace: "Consolas", fontSize: 12, bold: true, color: NAVY, breakLine: true } },
      { text: "python -m " + cmd, options: { fontFace: "Consolas", fontSize: 8.5, color: MUTE } },
    ], { x: x0 + 0.7, y, w: cName, h: rh, margin: 0, valign: "middle", lineSpacingMultiple: 1.1 });
    s.addShape(pres.shapes.LINE, { x: x0 + 0.7 + cName - 0.1, y: y + 0.12, w: 0, h: rh - 0.24, line: { color: "DDE5EE", width: 1 } });
    s.addText(proc, { x: x0 + 0.7 + cName, y, w: cProc - 0.15, h: rh, margin: 0, fontFace: KF, fontSize: 10.5, color: INK, valign: "middle", lineSpacingMultiple: 1.05 });
    s.addShape(pres.shapes.LINE, { x: x0 + 0.7 + cName + cProc - 0.1, y: y + 0.12, w: 0, h: rh - 0.24, line: { color: "DDE5EE", width: 1 } });
    s.addText(out, { x: x0 + 0.7 + cName + cProc, y, w: cOut - 0.2, h: rh, margin: 0, fontFace: KF, fontSize: 9.5, color: BLUE, valign: "middle", lineSpacingMultiple: 1.05 });
  });
  s.addText("각 단계의 산출물이 다음 단계의 입력이 되며, dvc.lock 해시로 정합성·증분 실행을 보장", { x: M, y: 6.7, w: W - 2 * M, h: 0.35, margin: 0, fontFace: KF, fontSize: 11.5, italic: true, color: MUTE, align: "center" });
}

// =============================================================== Slide 4b: 4 layers & reproducibility
{
  const s = pres.addSlide();
  contentTitle(s, "MLOps 4계층 & 재현성", "파이프라인을 떠받치는 도구의 역할과, 누구나 동일 결과를 얻는 방법");
  const layers = [
    ["Git / GitHub", "코드·설정 버전관리", "작업 브랜치 sim → main 통합. .gitignore로 데이터·모델·DB는 제외 → 코드와 데이터의 책임을 분리"],
    ["DVC", "데이터·파이프라인 버전관리", "content-addressed 캐시 + 4-stage DAG. dvc.lock 해시로 바뀐 단계만 증분 재실행, 어디서나 dvc pull로 복원"],
    ["MLflow", "실험 추적 · 모델 거버넌스", "params·metrics·artifacts를 run 단위로 기록. champion 태그로 서빙 대상 단일화(이전 champion은 superseded)"],
    ["Docker", "환경 동결 서빙", "runtime·의존성·champion 아티팩트를 추론 전용 image(~154MB)로 고정 → .venv 없이도 동일 추론"],
  ];
  const gw = (W - 2 * M - 0.3) / 2, gh = 1.85, gx = [M, M + gw + 0.3], gy = [1.55, 1.55 + gh + 0.25];
  layers.forEach(([k, role, why], i) => {
    const x = gx[i % 2], y = gy[Math.floor(i / 2)];
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: gw, h: gh, fill: { color: WHITE }, line: { color: "C9D6E3", width: 1 }, rectRadius: 0.06, shadow: shadow() });
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.12, h: gh, fill: { color: AMBER } });
    s.addText(k, { x: x + 0.3, y: y + 0.18, w: gw - 0.5, h: 0.45, margin: 0, fontFace: KF, fontSize: 18, bold: true, color: NAVY });
    s.addText(role, { x: x + 0.3, y: y + 0.66, w: gw - 0.5, h: 0.35, margin: 0, fontFace: KF, fontSize: 13, bold: true, color: BLUE });
    s.addText(why, { x: x + 0.3, y: y + 1.04, w: gw - 0.55, h: 0.72, margin: 0, fontFace: KF, fontSize: 12, color: INK, valign: "top", lineSpacingMultiple: 1.05 });
  });
  // reproduce callout
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 5.85, w: W - 2 * M, h: 0.95, fill: { color: NAVY }, rectRadius: 0.05, shadow: shadow() });
  s.addText([
    { text: "재현 절차    ", options: { bold: true, color: AMBER, fontSize: 15 } },
    { text: "git checkout", options: { fontFace: "Consolas", bold: true } },
    { text: "  →  ", options: { color: AMBER } },
    { text: "dvc pull", options: { fontFace: "Consolas", bold: true } },
    { text: "  →  ", options: { color: AMBER } },
    { text: "docker run", options: { fontFace: "Consolas", bold: true } },
  ], { x: M + 0.35, y: 5.95, w: W - 2 * M - 0.7, h: 0.45, margin: 0, fontFace: KF, fontSize: 15, color: WHITE, valign: "middle" });
  s.addText("코드(Git) + 데이터(DVC) + 모델 선택(MLflow) + 환경(Docker)이 맞물려 임의 시점 모델을 동일하게 재현", { x: M + 0.35, y: 6.38, w: W - 2 * M - 0.7, h: 0.35, margin: 0, fontFace: KF, fontSize: 12, color: ICE, valign: "middle" });
}

// =============================================================== Slide 5: Model experiment (process-emphasized)
{
  const s = pres.addSlide();
  contentTitle(s, "모델 실험 — 어떻게 비교했나", "동일 split에서 5개 후보를 같은 절차로 학습·평가하고 validation F1로 순위화 (AutoML-lite)");
  // experiment procedure: 5 numbered steps
  const steps = ["동일 split 고정", "5개 후보 학습", "동일 지표 평가", "Val F1 순위화", "Champion 선정"];
  const sw = 2.05, sgap = 0.22, sy = 1.5, sh = 0.82;
  steps.forEach((t, i) => {
    const x = M + i * (sw + sgap);
    const last = i === steps.length - 1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: sy, w: sw, h: sh, fill: { color: last ? NAVY : ICE }, rectRadius: 0.05, shadow: shadow() });
    s.addText([
      { text: String(i + 1) + "  ", options: { bold: true, color: last ? AMBER : BLUE, fontSize: 14 } },
      { text: t, options: { bold: true, color: last ? WHITE : NAVY, fontSize: 12 } },
    ], { x: x + 0.12, y: sy, w: sw - 0.24, h: sh, margin: 0, fontFace: KF, align: "center", valign: "middle" });
    if (!last) s.addText("›", { x: x + sw - 0.02, y: sy, w: sgap + 0.04, h: sh, margin: 0, fontFace: KF, fontSize: 18, bold: true, color: AMBER, align: "center", valign: "middle" });
  });
  // bar chart Val F1 (the ranking result)
  s.addText("결과: validation F1 순위", { x: M, y: 2.55, w: 6.3, h: 0.36, fontFace: KF, fontSize: 14, bold: true, color: BLUE });
  s.addChart(pres.charts.BAR, [{ name: "Val F1", labels: ["Logistic", "D.Tree", "XGBoost", "R.Forest", "LightGBM"], values: [0.523, 0.486, 0.467, 0.467, 0.389] }], {
    x: M, y: 2.95, w: 6.4, h: 3.5, barDir: "col", chartColors: [NAVY],
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: INK, dataLabelFontSize: 11, dataLabelFormatCode: "0.000",
    catAxisLabelColor: MUTE, catAxisLabelFontSize: 11, valAxisHidden: true, valGridLine: { style: "none" },
    showLegend: false, showTitle: false, fontFace: KF, valAxisMinVal: 0, valAxisMaxVal: 0.6,
  });
  s.addText("1위 ↓", { x: M + 0.35, y: 3.05, w: 1.2, h: 0.35, fontFace: KF, fontSize: 12, bold: true, color: AMBER });
  // right: champion + rationale
  const rx = 7.15;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx, y: 2.55, w: 5.58, h: 1.5, fill: { color: NAVY }, rectRadius: 0.06, shadow: shadow() });
  s.addText("Champion: Logistic Regression", { x: rx + 0.25, y: 2.7, w: 5.1, h: 0.45, margin: 0, fontFace: KF, fontSize: 16, bold: true, color: WHITE });
  s.addText([
    { text: "Val F1 ", options: {} }, { text: "0.523", options: { bold: true, color: AMBER } },
    { text: "  ·  Recall ", options: {} }, { text: "0.819", options: { bold: true, color: AMBER } },
    { text: "  ·  ROC-AUC ", options: {} }, { text: "0.743", options: { bold: true, color: AMBER } },
  ], { x: rx + 0.25, y: 3.25, w: 5.1, h: 0.6, margin: 0, fontFace: KF, fontSize: 14, color: WHITE, valign: "middle" });
  s.addText("선정 근거 (다음 단계 tune_logistic으로 핸드오프)", { x: rx, y: 4.3, w: 5.58, h: 0.4, fontFace: KF, fontSize: 14, bold: true, color: BLUE });
  s.addText([
    { text: "불량 탐지에 중요한 recall이 가장 높음", options: { bullet: true, breakLine: true } },
    { text: "학습·추론 < 0.01s 로 가장 빠름", options: { bullet: true, breakLine: true } },
    { text: "선형 모델 → 계수 기반 해석 용이", options: { bullet: true, breakLine: true } },
    { text: "트리·부스팅 계열은 recall/F1 열세", options: { bullet: true } },
  ], { x: rx, y: 4.7, w: 5.58, h: 1.8, fontFace: KF, fontSize: 13.5, color: INK, paraSpaceAfter: 8 });
}

// =============================================================== Slide 6: Tuning & MLflow
{
  const s = pres.addSlide();
  contentTitle(s, "튜닝 & 실험 추적 (MLflow)");
  // before/after table
  s.addText("5-fold CV 30회 탐색 → threshold 0.49 선택", { x: M, y: 1.45, w: 7, h: 0.4, fontFace: KF, fontSize: 15, bold: true, color: BLUE });
  const head = (t) => ({ text: t, options: { fill: { color: NAVY }, color: WHITE, bold: true, align: "center", fontFace: KF, fontSize: 12 } });
  const cc = (t, b) => ({ text: t, options: { align: "center", fontFace: KF, fontSize: 12, color: b ? NAVY : INK, bold: !!b } });
  const rowL = (t) => ({ text: t, options: { fontFace: KF, fontSize: 12, color: INK } });
  s.addTable([
    [head("구분"), head("F1"), head("ROC-AUC"), head("Recall"), head("Precision")],
    [rowL("Baseline val @0.50"), cc("0.523"), cc("0.743"), cc("0.819"), cc("0.384")],
    [rowL("Tuned val @0.49"), cc("0.534", 1), cc("0.748", 1), cc("0.843", 1), cc("0.391", 1)],
    [rowL("Champion test @0.49"), cc("0.488"), cc("0.758"), cc("0.759"), cc("0.360")],
  ], { x: M, y: 1.9, w: 6.8, colW: [2.2, 1.05, 1.3, 1.05, 1.2], rowH: [0.5, 0.5, 0.5, 0.5], border: { pt: 0.5, color: "C9D6E3" }, fill: { color: WHITE }, valign: "middle", autoPage: false,
    rowFill: ["FFFFFF", "FFFFFF", ICE, "FFFFFF"] });
  // best params + champion run
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 4.25, w: 7.0, h: 1.7, fill: { color: ICE2 }, rectRadius: 0.05 });
  s.addText("Best Params · Champion", { x: M + 0.2, y: 4.35, w: 6.6, h: 0.4, margin: 0, fontFace: KF, fontSize: 14, bold: true, color: NAVY });
  s.addText([
    { text: "C=0.0562 · penalty=l2 · solver=saga · class_weight=balanced", options: { breakLine: true } },
    { text: "MLflow run: c902f8c099e941e6ba2d3dc62a4cf3b1", options: { breakLine: true, fontSize: 12, color: MUTE } },
    { text: "params·metrics·artifacts 기록 + champion 태그 (이전 champion은 superseded)", options: { fontSize: 12 } },
  ], { x: M + 0.2, y: 4.75, w: 6.6, h: 1.1, margin: 0, fontFace: KF, fontSize: 13, color: INK, lineSpacingMultiple: 1.05 });
  // mlflow screenshot
  addPic(s, path.join(EVID, "mlflow_champion.png"), 7.95, 1.55, 4.78, 4.5);
  s.addText("MLflow champion run", { x: 7.95, y: 6.05, w: 4.78, h: 0.3, fontFace: KF, fontSize: 11, italic: true, color: MUTE, align: "center" });
}

// =============================================================== Slide 7: XAI
{
  const s = pres.addSlide();
  contentTitle(s, "XAI — 모델 해석 (SHAP)", "전역(global) + 지역(local) 해석으로 판단 근거 설명");
  addPic(s, path.join(PLOTS, "shap_summary_bar.png"), M, 1.6, 5.9, 3.5);
  s.addText("그림: 평균 |SHAP| 상위 feature", { x: M, y: 5.1, w: 5.9, h: 0.3, fontFace: KF, fontSize: 11, italic: true, color: MUTE, align: "center" });
  // right text
  const rx = 6.9;
  s.addText("전역 해석", { x: rx, y: 1.55, w: 5.8, h: 0.4, fontFace: KF, fontSize: 16, bold: true, color: BLUE });
  s.addText([
    { text: "Factory_Humidity — 가장 큰 기여 (높을수록 정상)", options: { bullet: true, breakLine: true } },
    { text: "Spray_1_Time — 높을수록 불량", options: { bullet: true, breakLine: true } },
    { text: "SHAP·permutation·EDA(SMD) 결과가 상호 일치", options: { bullet: true } },
  ], { x: rx, y: 1.95, w: 5.9, h: 1.7, fontFace: KF, fontSize: 14, color: INK, paraSpaceAfter: 8 });
  s.addText("지역 해석 (test sample #57)", { x: rx, y: 3.6, w: 5.8, h: 0.4, fontFace: KF, fontSize: 16, bold: true, color: BLUE });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx, y: 4.0, w: 5.9, h: 1.85, fill: { color: ICE2 }, rectRadius: 0.05 });
  s.addText([
    { text: "실제=불량, 예측=불량, defect 확률 0.801", options: { breakLine: true, bold: true, color: NAVY } },
    { text: "Factory_Humidity=46.4 → SHAP +1.39 (불량 강하게 기여)", options: { bullet: true, breakLine: true } },
    { text: "낮은 공장 습도가 핵심 동인 → 전역 해석과 일치", options: { bullet: true, breakLine: true } },
    { text: "변수별 기여도로 \"왜 불량인가\"를 엔지니어에게 제시", options: { bullet: true } },
  ], { x: rx + 0.2, y: 4.1, w: 5.5, h: 1.65, margin: 0, fontFace: KF, fontSize: 13, color: INK, paraSpaceAfter: 6, lineSpacingMultiple: 1.03 });
}

// =============================================================== Slide 8: Serving API & UI
{
  const s = pres.addSlide();
  contentTitle(s, "Serving — FastAPI & Streamlit");
  // endpoints
  s.addText("REST API (FastAPI)", { x: M, y: 1.45, w: 6, h: 0.4, fontFace: KF, fontSize: 16, bold: true, color: BLUE });
  const ep = [["GET /health", "모델 로드 상태 확인"], ["GET /model-info", "버전·threshold·feature 목록"], ["POST /predict", "예측·확률·상위 기여 feature"]];
  let ey = 1.9;
  ep.forEach(([k, v]) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: ey, w: 6.0, h: 0.62, fill: { color: ICE2 }, rectRadius: 0.05 });
    s.addText(k, { x: M + 0.15, y: ey, w: 2.5, h: 0.62, margin: 0, fontFace: "Consolas", fontSize: 13, bold: true, color: NAVY, valign: "middle" });
    s.addText(v, { x: M + 2.7, y: ey, w: 3.2, h: 0.62, margin: 0, fontFace: KF, fontSize: 12.5, color: INK, valign: "middle" });
    ey += 0.72;
  });
  // response note card
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 4.2, w: 6.0, h: 1.05, fill: { color: ICE2 }, rectRadius: 0.05 });
  s.addText([
    { text: "응답 구성: ", options: { bold: true, color: NAVY, breakLine: true } },
    { text: "예측 라벨 · 클래스 확률 · 상위 5개 기여 feature · 모델/데이터 버전", options: {} },
    { text: " → 추적성과 설명가능성 동시 제공", options: { color: MUTE } },
  ], { x: M + 0.2, y: 4.2, w: 5.6, h: 1.05, margin: 0, fontFace: KF, fontSize: 13, color: INK, valign: "middle", lineSpacingMultiple: 1.05 });
  // streamlit card
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 5.4, w: 6.0, h: 1.2, fill: { color: NAVY }, rectRadius: 0.05, shadow: shadow() });
  s.addText("Streamlit UI · 6개 탭", { x: M + 0.2, y: 5.5, w: 5.6, h: 0.4, margin: 0, fontFace: KF, fontSize: 14, bold: true, color: WHITE });
  s.addText("Predict · Metrics · Data EDA · Feature Importance · Baseline/XAI · Artifacts", { x: M + 0.2, y: 5.95, w: 5.6, h: 0.55, margin: 0, fontFace: KF, fontSize: 12.5, color: ICE, valign: "top" });
  // screenshots
  addPic(s, path.join(EVID, "fastapi_swagger.png"), 6.95, 1.5, 5.78, 2.6);
  addPic(s, path.join(EVID, "streamlit_champion_metrics.png"), 6.95, 4.2, 5.78, 2.6);
}

// =============================================================== Slide 9: Docker & Reproducibility
{
  const s = pres.addSlide();
  contentTitle(s, "Docker 배포 & 재현성");
  // stat cards
  statCard(s, M, 1.55, 2.55, 1.2, "~154 MB", "추론 전용 image");
  statCard(s, M + 2.7, 1.55, 2.55, 1.2, "healthy", "volume 없이 실행");
  // predict results
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 2.95, w: 5.25, h: 1.5, fill: { color: NAVY }, rectRadius: 0.06, shadow: shadow() });
  s.addText("컨테이너 /predict 검증", { x: M + 0.2, y: 3.05, w: 4.9, h: 0.4, margin: 0, fontFace: KF, fontSize: 14, bold: true, color: WHITE });
  s.addText([
    { text: "정상 샘플 → normal (확률 0.8937)", options: { bullet: true, breakLine: true, color: WHITE } },
    { text: "불량 샘플 → defect (확률 0.6982)", options: { bullet: true, breakLine: true, color: WHITE } },
    { text: "/health·/model-info 포함 4건 모두 HTTP 200", options: { bullet: true, color: ICE } },
  ], { x: M + 0.2, y: 3.5, w: 4.9, h: 0.95, margin: 0, fontFace: KF, fontSize: 12.5, paraSpaceAfter: 4 });
  // reproduce steps
  s.addText("3단계 재현", { x: M, y: 4.7, w: 5.25, h: 0.4, fontFace: KF, fontSize: 15, bold: true, color: BLUE });
  s.addText([
    { text: "git checkout", options: { bullet: { type: "number" }, breakLine: true } },
    { text: "dvc pull", options: { bullet: { type: "number" }, breakLine: true } },
    { text: "docker build & run", options: { bullet: { type: "number" } } },
  ], { x: M + 0.1, y: 5.1, w: 5.0, h: 1.4, fontFace: "Consolas", fontSize: 14, color: NAVY, bold: true, paraSpaceAfter: 8 });
  // docker swagger screenshot
  addPic(s, path.join(EVID, "docker_fastapi_swagger.png"), 6.4, 1.5, 6.33, 5.0);
  s.addText("Docker 컨테이너 FastAPI Swagger", { x: 6.4, y: 6.5, w: 6.33, h: 0.3, fontFace: KF, fontSize: 11, italic: true, color: MUTE, align: "center" });
}

// =============================================================== Slide 10: Conclusion
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.25, h: H, fill: { color: AMBER } });
  s.addText("결론 & Future Work", { x: 1.0, y: 0.7, w: 11.5, h: 0.9, fontFace: KF, fontSize: 32, bold: true, color: WHITE });
  // result cards
  statCard(s, 1.0, 1.95, 3.5, 1.2, "F1 0.488", "Test (recall 0.759)", "2A5C8A", WHITE, ICE);
  statCard(s, 4.7, 1.95, 3.5, 1.2, "ROC-AUC 0.758", "Test 기준", "2A5C8A", WHITE, ICE);
  statCard(s, 8.4, 1.95, 3.9, 1.2, "9 / 9", "평가 요구항목 충족", "2A5C8A", WHITE, ICE);
  s.addText("핵심 성과", { x: 1.0, y: 3.45, w: 11, h: 0.4, fontFace: KF, fontSize: 18, bold: true, color: AMBER });
  s.addText([
    { text: "데이터 버전관리 → 실험 추적 → 튜닝 → XAI → 서빙 → 컨테이너 배포 전 과정 구현·검증", options: { bullet: true, breakLine: true } },
    { text: "champion 모델은 불량 탐지(recall) 중심의 운영 가능한 baseline 제공", options: { bullet: true, breakLine: true } },
    { text: "git checkout + dvc pull + docker run 으로 환경 독립적 재현 보장", options: { bullet: true } },
  ], { x: 1.0, y: 3.85, w: 11.4, h: 1.7, fontFace: KF, fontSize: 15, color: WHITE, paraSpaceAfter: 8, lineSpacingMultiple: 1.05 });
  s.addText("Future Work", { x: 1.0, y: 5.55, w: 11, h: 0.4, fontFace: KF, fontSize: 18, bold: true, color: AMBER });
  s.addText([
    { text: "데이터/예측 드리프트 모니터링 · 자동 재학습 트리거", options: { bullet: true, breakLine: true } },
    { text: "AutoGluon/H2O 본격 AutoML · 클라우드 registry/배포 · 멀티클래스 결함 분류", options: { bullet: true } },
  ], { x: 1.0, y: 5.95, w: 11.4, h: 1.2, fontFace: KF, fontSize: 15, color: WHITE, paraSpaceAfter: 8, lineSpacingMultiple: 1.05 });
}

const out = path.join(ROOT, "docs", "diecasting_mlops_presentation.pptx");
pres.writeFile({ fileName: out }).then(f => console.log("WROTE", out));
