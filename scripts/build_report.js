// Generate the final MLOps project report (Korean) as a .docx file.
// Numbers are taken verbatim from artifacts/reports/*.json and docs/evidence.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, ImageRun, TableOfContents,
  HeadingLevel, BorderStyle, WidthType, ShadingType, VerticalAlign,
  PageNumber, PageBreak, ExternalHyperlink,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const PLOTS = path.join(ROOT, "artifacts", "plots");
const EVID = path.join(ROOT, "docs", "evidence");

// ---- helpers -------------------------------------------------------------
const CONTENT_W = 9360; // US Letter, 1" margins (DXA)

// Detect real image format + dimensions (some .png files are actually JPEG).
function imgInfo(file) {
  const b = fs.readFileSync(file);
  if (b[0] === 0x89 && b[1] === 0x50) {
    // PNG: IHDR width @16, height @20 (big-endian uint32)
    return { type: "png", data: b, w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
  }
  if (b[0] === 0xff && b[1] === 0xd8) {
    // JPEG: scan SOFn markers for dimensions
    let i = 2;
    while (i < b.length) {
      if (b[i] !== 0xff) { i++; continue; }
      const marker = b[i + 1];
      if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
        return { type: "jpg", data: b, h: b.readUInt16BE(i + 5), w: b.readUInt16BE(i + 7) };
      }
      i += 2 + b.readUInt16BE(i + 2);
    }
  }
  throw new Error("Unsupported image format: " + file);
}

function imageRun(file, maxWpx) {
  const { type, data, w, h } = imgInfo(file);
  const width = Math.min(maxWpx, w);
  const height = Math.round((h / w) * width);
  return new ImageRun({
    type, data, transformation: { width, height },
    altText: { title: path.basename(file), description: path.basename(file), name: path.basename(file) },
  });
}

// image scaled to a max display width in pixels, preserving aspect ratio
function image(file, maxWpx) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
    children: [imageRun(file, maxWpx)],
  });
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [new TextRun({ text, italics: true, size: 18, color: "555555" })],
  });
}

function h1(text) { return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] }); }
function h2(text) { return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] }); }
function h3(text) { return new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(text)] }); }

// paragraph from array of {text, bold, italics} runs OR a plain string
function p(content, opts = {}) {
  const runs = Array.isArray(content)
    ? content.map(r => new TextRun(r))
    : [new TextRun(content)];
  return new Paragraph({ spacing: { after: 120, line: 300 }, children: runs, ...opts });
}

function bullet(text, level = 0) {
  const runs = Array.isArray(text) ? text.map(r => new TextRun(r)) : [new TextRun(text)];
  return new Paragraph({ numbering: { reference: "bullets", level }, spacing: { after: 60, line: 290 }, children: runs });
}

const BORDER = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const BORDERS = { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER };
const HEAD_FILL = "1F4E79";
const ALT_FILL = "EEF3F8";

function cell(text, { width, head = false, alt = false, alignRight = false, bold = false } = {}) {
  const runs = (Array.isArray(text) ? text : [text]).map(t =>
    new TextRun({ text: String(t), bold: head || bold, color: head ? "FFFFFF" : "000000", size: 19 }));
  return new TableCell({
    borders: BORDERS,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: head ? HEAD_FILL : (alt ? ALT_FILL : "FFFFFF"), type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: alignRight ? AlignmentType.RIGHT : AlignmentType.LEFT, children: runs })],
  });
}

// build a table; widths sum to CONTENT_W; rows = array of arrays; first row is header
function table(widths, rows, { rightCols = [] } = {}) {
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map((cells, ri) => new TableRow({
      tableHeader: ri === 0,
      children: cells.map((c, ci) => cell(c, {
        width: widths[ci],
        head: ri === 0,
        alt: ri > 0 && ri % 2 === 0,
        alignRight: ri > 0 && rightCols.includes(ci),
      })),
    })),
  });
}

const GH = "https://github.com/jksim1833/2026_AJOU_mlops_team8";

// ---- document ------------------------------------------------------------
const children = [];

// Cover page
children.push(
  new Paragraph({ spacing: { before: 2400, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "MLOps 기말 프로젝트 최종 보고서", size: 28, color: "555555" })] }),
  new Paragraph({ spacing: { before: 400, after: 200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "다이캐스팅 정상/불량 예측을 위한", bold: true, size: 44 })] }),
  new Paragraph({ spacing: { after: 600 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "MLOps 기반 AI 서비스", bold: true, size: 44 })] }),
  new Paragraph({ spacing: { before: 200, after: 80 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "End-to-End MLOps Pipeline for Die-Casting Defect Prediction", italics: true, size: 24, color: "555555" })] }),
  new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "1F4E79", space: 1 } }, spacing: { before: 600, after: 400 }, children: [] }),
  new Paragraph({ spacing: { after: 120 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "아주대학교 · MLOps · 8조", size: 24 })] }),
  new Paragraph({ spacing: { after: 120 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "팀원: 김병근 (Data) · Zhang Xin (Modeling/XAI) · 심재광 (MLOps/Serving/발표)", size: 22 })] }),
  new Paragraph({ spacing: { after: 120 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "제출일: 2026-06-14", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [new ExternalHyperlink({ children: [new TextRun({ text: GH, style: "Hyperlink", size: 20 })], link: GH })] }),
  new Paragraph({ children: [new PageBreak()] }),
);

// TOC
children.push(
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("목차")] }),
  new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }),
  new Paragraph({ children: [new PageBreak()] }),
);

// Executive summary table
children.push(h1("요약 (Executive Summary)"));
children.push(p("본 프로젝트는 한국지능정보사회진흥원 KAMP 다이캐스팅(die-casting) 공정·센서 데이터를 사용해 제품의 정상/불량(normal/defect)을 예측하는 이진 분류 AI 서비스를, 데이터 버전관리부터 실험 추적·모델 서빙·컨테이너 배포까지 MLOps 전 과정으로 구현한 결과를 정리한다. 단순 모델 성능 비교가 아니라 '하나의 AI 서비스를 개발·운영하는 관점'에서 재현 가능한 파이프라인을 구축하는 데 초점을 두었다."));
children.push(table(
  [3120, 6240],
  [
    ["항목", "핵심 결과"],
    ["문제 정의", "공정 센서값으로 제품 불량 여부를 예측하고 근거(XAI)를 제시"],
    ["데이터", "KAMP Product 1, 중복 제거 후 2,515건 (normal 1,960 / defect 555)"],
    ["Champion 모델", "StandardScaler + Logistic Regression (logistic_champion_v1)"],
    ["성능 (Test)", "F1 0.488 · Recall 0.759 · ROC-AUC 0.758 (threshold 0.49)"],
    ["MLOps 스택", "Git/GitHub · DVC · MLflow · FastAPI · Streamlit · Docker"],
    ["재현성", "DVC 4-stage pipeline, clean clone에서 dvc pull 복원 검증"],
  ],
));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ===== 1. 개요 =====
children.push(h1("1. 프로젝트 개요"));
children.push(h2("1.1 문제 정의"));
children.push(p([
  { text: "한 문장 정의: ", bold: true },
  { text: "다이캐스팅 공정/센서 데이터로부터 제품의 정상/불량을 예측하고, 그 판단 근거를 XAI로 함께 제시하는 AI 서비스." },
]));
children.push(p("다이캐스팅은 고온 용탕을 고압으로 금형에 주입하는 공정으로, 온도·압력·분사 시간 등 다수의 공정 변수가 제품 품질에 영향을 준다. 현장의 공정 엔지니어는 이러한 센서값을 보고 불량 가능성을 판단해야 하지만, 변수 간 상호작용이 복잡해 사람의 직관만으로는 일관된 판단이 어렵다. 본 서비스는 공정값을 입력하면 정상/불량 예측과 주요 기여 변수를 반환하여 의사결정을 보조한다."));
children.push(h2("1.2 대상 사용자 · 입출력"));
children.push(table(
  [2200, 7160],
  [
    ["구분", "내용"],
    ["대상 사용자", "다이캐스팅 공정 엔지니어 / 품질 관리자"],
    ["입력 (Input)", "28개 공정·센서 feature (속도, 압력, 분사 시간, 온도, 습도 등)"],
    ["출력 (Output)", "정상/불량 예측, 클래스별 확률, 상위 기여 feature, 모델·데이터 버전"],
    ["성공 기준", "validation F1/ROC-AUC, MLflow 실험 기록, FastAPI+UI demo, Docker 실행 구조"],
  ],
));
children.push(h2("1.3 MVP 범위"));
children.push(p("불량 종류를 세분하지 않고 정상/불량 이진 분류로 단순화하여, 제한된 기간 안에 end-to-end MLOps 흐름을 완성하는 것을 MVP 목표로 삼았다. 멀티클래스 결함 분류, 실시간 스트리밍 추론, 클라우드 배포는 future work로 분리했다."));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ===== 2. 데이터 =====
children.push(h1("2. 데이터 이해와 전처리"));
children.push(h2("2.1 데이터 소스와 라벨 생성"));
children.push(p("원본 데이터는 KAMP 다이캐스팅 Product 1 데이터셋(DieCasting_Quality_Raw_Data_product1.csv)이다. 원본에는 공정/센서 측정값과 함께 결함 유형별 판정 컬럼(Short_Shot, Bubble, Blow_Hole, Crack 등)이 포함되어 있다."));
children.push(p([
  { text: "이진 라벨 규칙: ", bold: true },
  { text: "결함 컬럼들의 합이 0보다 크면 defect(1), 모두 0이면 normal(0)으로 defect_label을 생성했다. 라벨 생성 직후 26개의 결함 판정 컬럼을 feature에서 제거하여 label leakage(정답 누출)를 차단했다." },
]));
children.push(p([
  { text: "제거한 결함 컬럼(26개): ", bold: true },
  { text: "Short_Shot, Bubble, Exfoliation, Blow_Hole, Stain, Dent, Deformation, Contamination, Impurity, Crack, Scratch, Buring_Mark, Inclusions의 _1 / _2 버전." },
]));
children.push(h2("2.2 중복 제거 · 클래스 분포"));
children.push(table(
  [4680, 2340, 2340],
  [
    ["항목", "값", "비고"],
    ["원본 행 수", "4,207", "raw"],
    ["완전 중복 제거", "-1,692", "동일 processed row"],
    ["최종 행 수", "2,515", "dedup 후"],
    ["feature/컬럼 수", "28 / 29", "target 포함 29"],
    ["normal (0)", "1,960", "77.93%"],
    ["defect (1)", "555", "22.07%"],
    ["결측치", "0", "median imputation 결과 0"],
    ["상수 feature", "8", "Min/Max 류 (분산 0)"],
  ],
  { rightCols: [1] },
));
children.push(p("결측치는 numeric 변환 후 feature별 중앙값으로 대치했으나 실제 대치 건수는 0이었다. Air/Coolant/Factory의 _Min/_Max 8개 컬럼은 분산이 0인 상수 feature로 확인되었다(모델 입력에는 유지하되 해석 시 제외)."));
children.push(h2("2.3 데이터 분할 (Train/Valid/Test)"));
children.push(p("defect_label 기준 stratified split을 적용해 클래스 비율을 모든 분할에서 유지했다(test 15%, validation 15% of remaining, random_state=42). 세 분할 간 완전 동일 행 중복(exact overlap)은 0건으로 검증되어 data leakage가 없음을 확인했다."));
children.push(table(
  [3120, 2080, 2080, 2080],
  [
    ["분할", "전체", "normal", "defect"],
    ["Train", "1,759", "1,370", "389"],
    ["Validation", "378", "295", "83"],
    ["Test", "378", "295", "83"],
  ],
  { rightCols: [1, 2, 3] },
));
children.push(h2("2.4 EDA 및 Feature 분석"));
children.push(p("표준화 평균차(SMD = |μ_defect − μ_normal| / σ_pop) 기준으로 클래스 분리력이 큰 상위 8개 feature를 선정해 분포·상관을 분석했다."));
children.push(table(
  [4680, 2340, 2340],
  [
    ["Feature", "SMD", "target 상관"],
    ["Factory_Humidity", "0.715", "-0.297"],
    ["Factory_Temp", "0.581", "+0.241"],
    ["Spray_Time", "0.449", "-0.186"],
    ["Casting_Pressure", "0.408", "-0.169"],
    ["Cylinder_Pressure", "0.407", "-0.169"],
    ["Pressure_Rise_Time", "0.302", "+0.125"],
    ["Spray_2_Time", "0.275", "+0.114"],
    ["Air_Pressure", "0.254", "+0.105"],
  ],
  { rightCols: [1, 2] },
));
children.push(p("Factory_Humidity가 가장 강한 분리력을 보이며 불량과 음의 상관(습도가 낮을수록 불량 경향)을 나타냈다. Factory_Humidity와 Factory_Temp는 -0.95로 강한 음의 상관, Casting_Pressure와 Cylinder_Pressure는 +0.999로 거의 동일한 정보를 담아 다중공선성이 존재함을 확인했다."));
children.push(image(path.join(PLOTS, "eda_class_distribution.png"), 360));
children.push(caption("그림 2-1. 클래스 분포 (normal 77.9% / defect 22.1%) — class imbalance 존재"));
children.push(image(path.join(PLOTS, "eda_correlation_heatmap.png"), 460));
children.push(caption("그림 2-2. 선정 feature 간 상관 히트맵 — 압력 변수 간 강한 상관(다중공선성)"));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ===== 3. MLOps 구조 =====
children.push(h1("3. MLOps 구조"));
children.push(p("본 프로젝트의 MLOps 설계 원칙은 “같은 입력에서 항상 같은 결과가 나오고, 그 과정과 산출물이 모두 추적되며, 모델을 코드·환경과 함께 통째로 재현·배포할 수 있어야 한다”는 것이다. 이를 위해 (1) 코드 버전관리(Git), (2) 데이터·파이프라인 버전관리(DVC), (3) 실험 추적·모델 거버넌스(MLflow), (4) 환경 고정 서빙(Docker)의 네 계층을 결합했다. 네 도구는 독립적으로 동작하지 않고, 아래와 같이 하나의 재현 가능한 흐름으로 맞물린다."));
children.push(p([
  { text: "전체 데이터 흐름: ", bold: true },
  { text: "raw CSV → (DVC) prepare_data → train/valid/test → train_binary·compare_baselines_xai → tune_logistic → champion model.joblib → (MLflow) 실험 기록·champion 태깅 → (Docker) image build → FastAPI/Streamlit 서빙. 각 화살표의 입력·출력 해시와 파라미터가 버전관리되어, 임의의 시점 결과를 동일하게 복원할 수 있다." },
]));
children.push(table(
  [2400, 4200, 2760],
  [
    ["계층", "구성", "핵심 역할"],
    ["Source Control", "Git / GitHub (sim → main)", "코드·설정 이력, 협업"],
    ["Data Versioning", "DVC 4-stage pipeline + local remote", "데이터·산출물 추적, 재실행"],
    ["Experiment Tracking", "MLflow (sqlite backend + artifact store)", "실험 비교, champion 거버넌스"],
    ["Serving", "FastAPI + Streamlit + Docker", "환경 고정, 추론 제공"],
  ],
));

children.push(h2("3.1 Source Code Version Control (Git/GitHub)"));
children.push(p([
  { text: "원격 저장소 " },
  { text: "2026_AJOU_mlops_team8", bold: true },
  { text: "에서 통합 브랜치 sim에 작업을 누적하고 최종 제출 시 default branch인 main에 반영하는 단순 브랜치 전략을 사용한다. 팀 작업은 의미 단위 커밋(데이터 준비, baseline, 튜닝, Docker 검증 등)으로 분리해 변경 이력을 추적 가능하게 유지했다." },
]));
children.push(p([
  { text: "버전관리 경계: ", bold: true },
  { text: ".gitignore로 data/raw, data/processed, artifacts/, mlflow.db 등 대용량 데이터와 생성물은 Git에서 제외하고, 코드(src/)·설정(configs/params.yaml, dvc.yaml)·DVC 메타파일(*.dvc, dvc.lock)만 Git으로 관리한다. 이렇게 “코드는 Git, 데이터/모델은 DVC”로 책임을 분리하면 저장소가 가벼워지고, 동시에 어떤 코드 커밋이 어떤 데이터·모델 버전과 짝을 이루는지 dvc.lock의 해시를 통해 1:1로 추적된다." },
]));
children.push(p([
  { text: "단일 진실 원천(Single Source of Truth): ", bold: true },
  { text: "split 비율, random_state(42), 튜닝 search 횟수·CV fold·threshold 탐색 범위, MLflow experiment 이름 등 모든 실행 파라미터를 configs/params.yaml 한 곳에 모았다. 코드에 숫자를 흩뿌리지 않고 설정으로 외부화함으로써, 파라미터 변경이 곧 DVC 의존성 변경으로 감지되어 재현성과 실험 관리가 동시에 보장된다." },
]));

children.push(h2("3.2 Data Version Control (DVC)"));
children.push(p([
  { text: "동작 원리: ", bold: true },
  { text: "DVC는 데이터·모델 파일 자체를 Git에 넣는 대신, 파일 내용을 해시(MD5)로 식별하는 content-addressed 캐시(.dvc/cache)에 저장하고, Git에는 그 해시를 가리키는 작은 메타파일만 커밋한다. 따라서 수백 MB의 데이터가 바뀌어도 Git diff는 해시 한 줄만 바뀌며, 특정 커밋으로 checkout하면 그 시점의 정확한 데이터·모델이 복원된다." },
]));
children.push(p([
  { text: "파이프라인(DAG)과 증분 실행: ", bold: true },
  { text: "dvc.yaml에 4개 stage를 의존성(deps)과 산출물(outs)로 선언하면, DVC는 이를 방향성 비순환 그래프(DAG)로 해석한다. dvc repro 실행 시 각 stage의 입력 코드·데이터·params 해시를 dvc.lock에 기록된 값과 비교해, 변경이 감지된 stage와 그 하위만 다시 실행한다(증분 실행). 즉 전처리만 바뀌면 전체가 아니라 영향받는 단계만 재계산되어 빠르고 일관된 재현이 가능하다." },
]));
children.push(table(
  [2600, 3380, 3380],
  [
    ["Stage", "역할", "주요 산출물(outs)"],
    ["prepare_data", "라벨 생성·중복 제거·stratified 분할·EDA", "processed CSV, train/valid/test, data_profile.json, EDA plots"],
    ["train_binary", "RandomForest baseline 학습 + MLflow 로깅", "rf_baseline.joblib, 지표/플롯"],
    ["compare_baselines_xai", "5종 후보 leaderboard + SHAP 해석", "baseline_comparison.json, SHAP 산출물"],
    ["tune_logistic", "5-fold CV 튜닝·threshold 선택·champion 등록", "model.joblib, metadata.json, metrics.json"],
  ],
));
children.push(p([
  { text: "재현성 검증: ", bold: true },
  { text: "현재 dvc status는 “Data and pipelines are up to date”를 반환해 커밋된 코드·데이터·산출물이 정합함을 보장한다. dvc.lock에는 각 stage 입출력의 해시가 고정되어 있어, 다른 시점·다른 PC에서도 동일 파이프라인 결과를 재현할 수 있다." },
]));
children.push(p([
  { text: "원격 저장소(remote): ", bold: true },
  { text: "week03 강의의 local-folder 방식을 따라 sibling 폴더를 DVC default remote(localstorage)로 설정했다. dvc push로 캐시 객체 49개를 remote에 업로드하고, clean clone 상태에서 dvc pull로 데이터·모델이 그대로 복원되는 것을 검증했다(dvc status -c로 cache–remote 동기화 확인). 공유 클라우드 remote(S3/GCS 등)는 별도 인프라가 필요해 본 MVP 범위에서는 제외했고, 향후 과제로 둔다." },
]));

children.push(h2("3.3 Experiment Tracking & Model Governance (MLflow)"));
children.push(p([
  { text: "추적 아키텍처: ", bold: true },
  { text: "MLflow는 backend store(메타데이터)와 artifact store(파일)로 구성된다. 본 프로젝트는 추가 서버 없이 로컬에서 운영하기 위해 backend를 SQLite(sqlite:///mlflow.db)로, artifact는 로컬 파일시스템으로 두었다. 경량 SQLite 백엔드는 모델 단계 전이(stage transition)와 태깅을 지원하므로, 단일 PC 환경에서 실험 비교와 간단한 모델 거버넌스를 모두 충족한다." },
]));
children.push(p([
  { text: "Run 단위 기록: ", bold: true },
  { text: "모든 학습/튜닝은 하나의 MLflow run으로 기록된다. run마다 (1) 하이퍼파라미터(params), (2) validation/test의 F1·ROC-AUC·recall·precision 등 지표(metrics), (3) 아티팩트(직렬화 모델, confusion matrix·ROC·SHAP 플롯, normal/defect 예시 요청)를 함께 로깅한다. 덕분에 baseline·튜닝 run을 같은 화면에서 정량 비교하고, 결과를 코드 변경과 연결해 추적할 수 있다." },
]));
children.push(p([
  { text: "Champion 거버넌스(승격 라이프사이클): ", bold: true },
  { text: "tune_logistic은 최종 모델 run에 champion 및 serving_candidate=true 태그를 부여한다. 새 champion이 등록되면 동일 experiment의 이전 champion run은 자동으로 superseded로 강등되어, 서빙 후보가 항상 정확히 하나만 유지된다. 이는 모델 레지스트리의 “Production 단계 단일화” 패턴을 경량으로 구현한 것으로, 서빙 측(FastAPI/Docker)은 champion 아티팩트만 신뢰하면 되도록 책임을 분리한다." },
]));
children.push(p([
  { text: "현재 champion run ID는 ", },
  { text: "c902f8c099e941e6ba2d3dc62a4cf3b1", bold: true },
  { text: " 이며, 모델·데이터 버전과 threshold(0.49)가 metadata.json에 함께 고정되어 서빙 시점까지 일관되게 전달된다." },
]));
children.push(image(path.join(EVID, "mlflow_champion.png"), 520));
children.push(caption("그림 3-1. MLflow champion run — 파라미터·지표·태그(run ID c902f8c0...)"));

children.push(h2("3.4 컨테이너화 (Docker)"));
children.push(p([
  { text: "설계 의도: ", bold: true },
  { text: "week15 자료의 핵심 요구는 production 클라우드 배포 자체가 아니라 실행 재현성과 안정적인 serving demo다. 이에 champion 추론 환경을 API 코드·Python runtime·고정 버전 의존성·champion 아티팩트로 묶어 단일 image로 고정했다. 로컬 .venv가 없어도, 다른 PC에서도 동일한 추론 결과를 보장하는 것이 목표다." },
]));
children.push(p([
  { text: "추론 전용 최소 image: ", bold: true },
  { text: "이미지는 의도적으로 추론에 필요한 것만 포함한다. 포함: src/api, model.joblib, metadata.json, feature_importance.json, 고정 API 의존성(requirements-api.txt). 제외: raw/processed 학습 데이터, DVC 캐시·MLflow DB, 학습·튜닝·EDA 코드, Streamlit UI. .dockerignore로 빌드 컨텍스트를 최소화해(약 202KB) image 크기를 약 154MB로 유지했고, HEALTHCHECK로 컨테이너 상태를 모니터링한다. 빌드·실행·API 검증의 상세 결과는 6장에서 다룬다." },
]));

children.push(h2("3.5 End-to-End 재현성 보장"));
children.push(p("네 계층은 다음과 같이 맞물려 “코드 한 커밋 → 동일한 데이터 → 동일한 모델 → 동일한 서빙 환경”을 보장한다."));
children.push(bullet([{ text: "코드+설정: ", bold: true }, { text: "Git 커밋 + configs/params.yaml이 실행 로직과 파라미터를 고정한다." }]));
children.push(bullet([{ text: "데이터+파이프라인: ", bold: true }, { text: "dvc.lock 해시가 입력 데이터와 각 stage 산출물을 고정하고, dvc pull로 어디서나 복원한다." }]));
children.push(bullet([{ text: "실험+모델 선택: ", bold: true }, { text: "MLflow run이 지표·아티팩트를 기록하고 champion 태그가 서빙 대상을 단일화한다." }]));
children.push(bullet([{ text: "실행 환경: ", bold: true }, { text: "Docker image가 runtime·의존성·champion 아티팩트를 동결해 추론 결과를 환경 독립적으로 재현한다." }]));
children.push(p("결과적으로 임의 시점의 모델을 “git checkout + dvc pull + docker run” 세 단계로 재현할 수 있으며, 이것이 단순 성능 비교를 넘어선 본 프로젝트의 MLOps 기여다."));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ===== 4. 모델 실험 =====
children.push(h1("4. 모델 실험 및 성능 비교"));
children.push(h2("4.1 AutoML-lite Leaderboard"));
children.push(p("동일한 split 위에서 5개 후보 모델을 학습·평가하고 validation F1(동점 시 ROC-AUC)을 기준으로 순위화하는 AutoML-lite leaderboard를 구현했다. 모든 모델은 class_weight='balanced'로 불균형을 보정했다."));
children.push(table(
  [2520, 1368, 1368, 1368, 1368, 1368],
  [
    ["Model", "Val F1", "Val AUC", "Val Recall", "Test F1", "Test AUC"],
    ["Logistic Regression", "0.523", "0.743", "0.819", "0.494", "0.758"],
    ["Decision Tree", "0.486", "0.749", "0.819", "0.467", "0.690"],
    ["XGBoost", "0.467", "0.731", "0.602", "0.424", "0.719"],
    ["Random Forest", "0.467", "0.746", "0.590", "0.487", "0.745"],
    ["LightGBM", "0.389", "0.715", "0.410", "0.365", "0.688"],
  ],
  { rightCols: [1, 2, 3, 4, 5] },
));
children.push(p([
  { text: "Champion 선정 근거: ", bold: true },
  { text: "Logistic Regression이 validation F1 0.523으로 최고였다. 불량 탐지에서 중요한 recall(0.819)이 높고, 학습·추론이 가장 빠르며(<0.01s), 선형 모델이라 계수 기반 해석이 용이하다. 트리/부스팅 계열은 recall이 낮거나(RF/XGB ~0.59~0.60) F1이 떨어져, 본 데이터·목표에는 단순 선형 모델이 가장 적합했다." },
]));
children.push(image(path.join(PLOTS, "baseline_validation_metrics.png"), 480));
children.push(caption("그림 4-1. 5개 baseline 모델 validation 지표 비교"));
children.push(h2("4.2 Logistic Regression 튜닝"));
children.push(p("선정된 Logistic Regression을 StratifiedKFold(5-fold) CV로 튜닝했다. solver·penalty 조합과 C(logspace −3~3), class_weight를 탐색공간으로 두고 deterministic하게 30회 후보를 평가(F1 기준)했다. 최적 파라미터는 다음과 같다."));
children.push(table(
  [4680, 4680],
  [
    ["하이퍼파라미터", "값"],
    ["C (역정규화 강도)", "0.0562"],
    ["penalty", "l2"],
    ["solver", "saga"],
    ["class_weight", "balanced"],
  ],
));
children.push(h2("4.3 결정 임계값(Threshold) 튜닝"));
children.push(p("validation 확률에 대해 0.05~0.95 구간을 0.01 step으로 sweep하며 F1을 최대화하는 임계값을 선택했다(동점 시 recall→precision→0.5와의 거리). 최종 임계값은 0.49이다. 기본 0.50 대비 recall이 향상되어 불량 누락(false negative)을 줄였다."));
children.push(table(
  [3360, 1500, 1500, 1500, 1500],
  [
    ["구분", "F1", "ROC-AUC", "Recall", "Precision"],
    ["Baseline validation @0.50", "0.523", "0.743", "0.819", "0.384"],
    ["Tuned validation @0.49", "0.534", "0.748", "0.843", "0.391"],
    ["Champion test @0.49", "0.488", "0.758", "0.759", "0.360"],
  ],
  { rightCols: [1, 2, 3, 4] },
));
children.push(image(path.join(PLOTS, "logistic_threshold_tuning.png"), 460));
children.push(caption("그림 4-2. 임계값 sweep에 따른 지표 변화 (선택값 0.49)"));
children.push(p([
  { text: "MLflow champion run ID: ", bold: true },
  { text: "c902f8c099e941e6ba2d3dc62a4cf3b1. 최종 모델은 train split만으로 적합되어 valid/test 누출 없이 평가되었다." },
]));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, children: [
  imageRun(path.join(PLOTS, "roc_curve.png"), 300),
  imageRun(path.join(PLOTS, "confusion_matrix.png"), 300),
] }));
children.push(caption("그림 4-3. Champion 모델 ROC curve(좌)와 confusion matrix(우, test set)"));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ===== 5. XAI =====
children.push(h1("5. XAI 및 오류 분석"));
children.push(h2("5.1 전역 해석 (Global Explanation)"));
children.push(p("SHAP(LinearExplainer)와 permutation importance를 함께 사용해 모델의 전역 판단 근거를 분석했다. 두 방법 모두 Factory_Humidity, Spray_1_Time, Spray_Time을 최상위 기여 변수로 지목해 해석의 일관성을 확인했다."));
children.push(table(
  [3120, 2080, 4160],
  [
    ["Feature", "평균 |SHAP|", "효과 방향"],
    ["Factory_Humidity", "0.779", "값이 높을수록 → normal"],
    ["Spray_1_Time", "0.489", "값이 높을수록 → defect"],
    ["Factory_Temp", "0.393", "값이 높을수록 → normal"],
    ["Spray_Time", "0.358", "값이 높을수록 → normal"],
    ["Velocity_2", "0.244", "값이 높을수록 → normal"],
    ["Air_Pressure", "0.188", "값이 높을수록 → defect"],
    ["Coolant_Temp", "0.187", "값이 높을수록 → normal"],
    ["Cylinder_Pressure", "0.133", "값이 높을수록 → normal"],
  ],
  { rightCols: [1] },
));
children.push(p("permutation importance(test set, F1 기준) 상위는 Factory_Humidity(0.079), Spray_1_Time(0.035), Spray_Time(0.020), Coolant_Temp(0.010)로, EDA의 SMD 분석 및 SHAP 결과와 부합한다. 즉 공장 습도와 분사 시간이 불량 판정의 핵심 동인이다."));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, children: [
  imageRun(path.join(PLOTS, "shap_summary_bar.png"), 300),
  imageRun(path.join(PLOTS, "shap_beeswarm.png"), 300),
] }));
children.push(caption("그림 5-1. SHAP 요약 — 평균 |SHAP| 막대(좌)와 beeswarm(우)"));
children.push(h2("5.2 지역 해석 (Local Explanation)"));
children.push(p("개별 예측의 근거를 확인하기 위해, 모델이 가장 높은 확신으로 불량으로 판정한 test 샘플(index 57, 실제=불량, 예측=불량, defect 확률 0.801)을 SHAP waterfall로 분해했다."));
children.push(table(
  [3120, 2080, 2080, 2080],
  [
    ["Feature", "입력값", "SHAP", "방향"],
    ["Factory_Humidity", "46.4", "+1.391", "불량 강하게 기여"],
    ["Factory_Temp", "36.1", "-0.556", "정상 쪽"],
    ["Velocity_2", "0.165", "+0.512", "불량 쪽"],
    ["Spray_Time", "8.0", "+0.268", "불량 쪽"],
    ["Spray_1_Time", "1.0", "+0.228", "불량 쪽"],
  ],
  { rightCols: [1, 2] },
));
children.push(p("이 샘플은 낮은 공장 습도(46.4)가 불량 판정에 가장 크게 기여했으며, 이는 전역 해석(습도가 낮을수록 불량)과 일치한다. 엔지니어는 이러한 변수별 기여도를 통해 '왜 불량으로 판정되었는지'를 확인하고 공정 조정의 단서를 얻을 수 있다."));
children.push(image(path.join(PLOTS, "shap_waterfall_defect_sample.png"), 480));
children.push(caption("그림 5-2. 불량 샘플(index 57)의 SHAP waterfall 지역 해석"));
children.push(h2("5.3 모델 한계와 개선 방향"));
children.push(bullet([{ text: "낮은 precision(test 0.36): ", bold: true }, { text: "recall을 높이는 임계값(0.49)을 택해 false positive가 많다. 불량 누락 비용이 큰 제조 맥락에서는 합리적이나, 과검 비용이 크면 임계값 재조정이 필요하다." }]));
children.push(bullet([{ text: "class imbalance(22% 불량): ", bold: true }, { text: "class_weight로 보정했으나 근본적 한계가 있어 SMOTE 등 리샘플링·비용민감 학습을 추가 검토할 수 있다." }]));
children.push(bullet([{ text: "다중공선성: ", bold: true }, { text: "압력 변수 간 상관이 매우 높아 계수 해석이 왜곡될 수 있다. feature 선택/차원 축소로 개선 가능하다." }]));
children.push(bullet([{ text: "선형 모델 한계: ", bold: true }, { text: "비선형 상호작용을 충분히 포착하지 못한다. 데이터가 늘면 부스팅 계열 재평가가 필요하다." }]));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ===== 6. Serving =====
children.push(h1("6. Serving 및 Demo"));
children.push(h2("6.1 FastAPI Endpoint"));
children.push(p("champion 모델을 FastAPI REST API로 서빙한다. 세 개의 endpoint를 제공한다."));
children.push(table(
  [2400, 1600, 5360],
  [
    ["Endpoint", "Method", "기능"],
    ["/health", "GET", "모델 로드 상태 확인 (status=ok)"],
    ["/model-info", "GET", "모델·데이터 버전, threshold, feature 목록 반환"],
    ["/predict", "POST", "feature 입력 → 예측·확률·상위 기여 feature 반환"],
  ],
));
children.push(p("/predict는 입력 feature를 DataFrame으로 변환해 모델 확률을 계산하고, metadata의 threshold(0.49)를 적용해 normal/defect를 결정한다. 응답에는 예측 라벨, 클래스별 확률, 상위 5개 기여 feature(입력값 포함), 모델·데이터 버전이 포함되어 추적성과 설명가능성을 함께 제공한다."));
children.push(image(path.join(EVID, "fastapi_swagger.png"), 520));
children.push(caption("그림 6-1. FastAPI Swagger UI — /health, /model-info, /predict"));
children.push(h2("6.2 Streamlit Web UI"));
children.push(p("비개발자도 사용할 수 있도록 Streamlit 대시보드를 6개 탭으로 구성했다."));
children.push(bullet("Predict: 고정 normal/defect 샘플로 /predict 호출, 예측·확률·상위 feature 표시"));
children.push(bullet("Metrics: champion 버전·threshold·데이터 프로파일과 validation/test 지표, classification report"));
children.push(bullet("Data EDA: 클래스 분포·상관 히트맵·분포/박스플롯·이상치 후보·상수 feature"));
children.push(bullet("Feature Importance: 상위 N개 permutation importance"));
children.push(bullet("Baseline/XAI: 5종 leaderboard 표·플롯과 SHAP 요약·지역 해석"));
children.push(bullet("Artifacts: 산출물 파일 존재 여부·크기 점검"));
children.push(image(path.join(EVID, "streamlit_champion_metrics.png"), 520));
children.push(caption("그림 6-2. Streamlit 대시보드 — champion 지표 화면"));
children.push(h2("6.3 Docker 배포 및 검증"));
children.push(p("API 코드·Python runtime·고정 의존성·champion 아티팩트를 하나의 image로 고정해, 로컬 .venv와 독립적으로 동일한 추론 환경을 재현한다. 이미지는 추론 전용이며 학습/DVC/MLflow/Streamlit은 포함하지 않는다."));
children.push(table(
  [3120, 6240],
  [
    ["항목", "값"],
    ["Image", "diecasting-api:logistic-champion-v1"],
    ["Base runtime", "python:3.10-slim"],
    ["Image size", "약 154 MB"],
    ["Health status", "healthy (volume 없이 image 자체 아티팩트로 실행)"],
    ["/health", "status = ok"],
    ["/model-info", "logistic_champion_v1 / binary_product1_v2_dedup / threshold 0.49"],
    ["normal /predict", "normal, normal 확률 0.8937"],
    ["defect /predict", "defect, defect 확률 0.6982"],
  ],
));
children.push(p("DVC pull로 champion 아티팩트를 복원한 뒤 docker build → docker run으로 컨테이너를 기동하고, 네 개 요청(/health, /model-info, 정상/불량 /predict)이 모두 HTTP 200으로 정상 동작함을 확인했다."));
children.push(image(path.join(EVID, "docker_fastapi_swagger.png"), 520));
children.push(caption("그림 6-3. Docker 컨테이너에서 제공되는 FastAPI Swagger UI"));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ===== 7. 결론 =====
children.push(h1("7. 결론 및 Future Work"));
children.push(h2("7.1 결론"));
children.push(p("본 프로젝트는 다이캐스팅 불량 예측이라는 단일 문제를, 데이터 버전관리(DVC) → 실험 추적(MLflow) → 모델 튜닝 → XAI → API/UI 서빙 → 컨테이너 배포로 이어지는 end-to-end MLOps 파이프라인으로 구현하고 각 단계를 실제로 검증했다. Champion 모델(Logistic Regression)은 test 기준 F1 0.488·recall 0.759·ROC-AUC 0.758로, 불량 탐지(recall)에 초점을 둔 운영 가능한 baseline을 제공한다. 무엇보다 dvc repro/pull과 Docker로 동일 결과를 재현할 수 있는 구조를 갖춘 것이 핵심 성과다."));
children.push(p("9개 평가 요구 항목(문제정의·데이터 이해·소스 버전관리·데이터 버전관리·전처리/FE·AutoML/실험관리·성능비교·XAI·API/UI 배포)을 모두 충족했다."));
children.push(h2("7.2 Future Work"));
children.push(bullet("데이터/예측 분포 드리프트 모니터링과 알림 체계 도입"));
children.push(bullet("신규 데이터 누적 시 자동 재학습·재평가 트리거 및 champion 자동 갱신"));
children.push(bullet("AutoGluon/H2O 등 본격 AutoML 도입으로 후보 모델 폭 확장"));
children.push(bullet("클라우드 registry push 및 원격 서버/오케스트레이션 배포, 모델 성능 모니터링"));
children.push(bullet("멀티클래스 결함 분류 및 비용민감/리샘플링 기반 불균형 대응 고도화"));

// ---- assemble ------------------------------------------------------------
const doc = new Document({
  creator: "AJOU MLOps Team 8",
  title: "다이캐스팅 정상/불량 예측을 위한 MLOps 기반 AI 서비스",
  styles: {
    default: { document: { run: { font: "맑은 고딕", size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: "1F4E79", font: "맑은 고딕" },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "1F4E79", space: 4 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, color: "2E75B6", font: "맑은 고딕" },
        paragraph: { spacing: { before: 220, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, color: "404040", font: "맑은 고딕" },
        paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 540, hanging: 280 } } } },
      ] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({
      alignment: AlignmentType.RIGHT, border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC", space: 2 } },
      children: [new TextRun({ text: "다이캐스팅 MLOps 최종 보고서 · 8조", size: 16, color: "888888" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "", size: 16, color: "888888" }), new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "888888" })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then(buffer => {
  const out = path.join(ROOT, "docs", "diecasting_mlops_final_report.docx");
  fs.writeFileSync(out, buffer);
  console.log("WROTE", out, buffer.length, "bytes");
});
