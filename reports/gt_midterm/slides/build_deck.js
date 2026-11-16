// HEAT 중간발표 슬라이드 (1차 발표 디자인 재현: 1920x1080, 배경 #F7F7F7, Kaisei Tokumin, 남색 #004490, 빨강 #AE3A1E)
// 실행: node build_deck.js  ->  HEAT_midterm.pptx
const pptxgen = require("pptxgenjs");
const path = require("path");
const N = require("./deck_numbers.json");     // 실험 결과 숫자 (make_numbers.py)
const FIG = (f) => path.join(__dirname, "fig", f);

const pres = new pptxgen();
pres.defineLayout({ name: "HD", width: 20, height: 11.25 });
pres.layout = "HD";
pres.title = "HEAT Midterm";
pres.theme = { headFontFace: "Kaisei Tokumin", bodyFontFace: "Kaisei Tokumin" };

const F = "Kaisei Tokumin";
const C = { bg: "F7F7F7", navy: "004490", red: "AE3A1E", ink: "222222", body: "333333", grey: "8A8F98", line: "004490", yel: "FEEFC5", peach: "FECCBE", title: "2E3D86" };
const px = (v) => v / 96;                       // 1920px = 20in
const PT = (p) => p * 0.75;                     // 디자인 px -> pt

function base(title) {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  s.addText(title, { x: 0, y: px(28), w: 20, h: px(103), fontFace: F, fontSize: PT(70), bold: true, color: "000000", align: "center", valign: "middle", margin: 0, isTextBox: true,
    shadow: { type: "outer", color: "888888", opacity: 0.45, blur: 2, offset: 2, angle: 45 } });
  s.addShape(pres.shapes.LINE, { x: 0, y: px(149), w: 20, h: 0, line: { color: C.line, width: 1.25 } });
  return s;
}
function bottom(s, text, y = 950) {
  s.addShape(pres.shapes.LINE, { x: 0, y: px(y - 22), w: 20, h: 0, line: { color: C.line, width: 1.25 } });
  s.addText([{ text, options: { highlight: C.peach } }], { x: px(60), y: px(y), w: px(1800), h: px(110), fontFace: F, fontSize: PT(40), bold: true, color: C.ink, align: "center", valign: "middle", margin: 0, isTextBox: true });
}
function head(s, text, x, y, w, color = C.navy, size = 50) {
  s.addText(text, { x: px(x), y: px(y), w: px(w), h: px(size * 1.45), fontFace: F, fontSize: PT(size), bold: true, color, margin: 0, valign: "middle", isTextBox: true });
}
// 본문: [{label, text}] 또는 문자열. label은 노란 형광 [대괄호]
function body(s, items, x, y, w, h, size = 32, gap = 10) {
  const runs = [];
  items.forEach((it, i) => {
    const last = i === items.length - 1;
    if (typeof it === "string") { runs.push({ text: it, options: { bullet: { indent: 22 }, breakLine: !last, paraSpaceAfter: gap } }); return; }
    if (it.label) runs.push({ text: `[${it.label}] `, options: { bold: true, highlight: C.yel, color: C.ink } });
    (Array.isArray(it.text) ? it.text : [{ t: it.text }]).forEach((r, j, arr) => {
      runs.push({ text: r.t, options: { bold: !!r.b, underline: r.u ? { style: "sng" } : undefined, color: r.c || C.body, breakLine: !last && j === arr.length - 1, paraSpaceAfter: gap } });
    });
  });
  s.addText(runs, { x: px(x), y: px(y), w: px(w), h: px(h), fontFace: F, fontSize: PT(size), color: C.body, valign: "top", margin: 0, isTextBox: true, lineSpacingMultiple: 1.05 });
}
function vdash(s, x, y0 = 165, y1 = 905) { s.addShape(pres.shapes.LINE, { x: px(x), y: px(y0), w: 0, h: px(y1 - y0), line: { color: "9AA6B8", width: 1.25, dashType: "dash" } }); }
const img = (s, f, x, y, w, h) => s.addImage({ path: FIG(f), x: px(x), y: px(y), w: px(w), h: px(h) });
function fitImg(s, f, x, y, w, h) {     // 원본 비율 유지, 상자 안 가운데 (비율은 deck_numbers.json)
  const ratio = N.ratio[f];
  let W = w, H = w / ratio; if (H > h) { H = h; W = h * ratio; }
  img(s, f, x + (w - W) / 2, y + (h - H) / 2, W, H);
}
const notes = (s, t) => s.addNotes(t);

// 1. 표지 ----------------------------------------------------------------
{
  const s = pres.addSlide(); s.background = { color: C.bg };
  s.addImage({ path: FIG("seoul_map.png"), x: px(554), y: px(60), w: px(1338), h: px(900), sizing: { type: "contain", w: px(1338), h: px(900) } });
  s.addShape(pres.shapes.LINE, { x: 0, y: px(115), w: 20, h: 0, line: { color: C.line, width: 1 } });
  s.addShape(pres.shapes.LINE, { x: 0, y: px(1011), w: 20, h: 0, line: { color: C.line, width: 1 } });
  s.addShape(pres.shapes.LINE, { x: px(1746), y: 0, w: 0, h: 11.25, line: { color: C.line, width: 1 } });
  s.addShape(pres.shapes.LINE, { x: px(122 - 90), y: 0, w: 0, h: 11.25, line: { color: C.line, width: 1 } });
  s.addText([{ text: "HEAT", options: { fontSize: PT(153), color: C.title, bold: true, breakLine: true,
      shadow: { type: "outer", color: "777777", opacity: 0.5, blur: 2, offset: 3, angle: 45 } } },
    { text: ": Hotspot Emergence Analysis in Seoul via Time series", options: { fontSize: PT(58), color: "000000", bold: true } }],
    { x: px(122), y: px(330), w: px(1600), h: px(330), fontFace: F, margin: 0, valign: "middle", isTextBox: true });
  s.addText([{ text: "Midterm Presentation · Team #14 - HEAT", options: { breakLine: true } },
    { text: "Jaemin Lim, Kyungho Kim, Jaehyun Kim, Junhyeok Choi" }],
    { x: px(122), y: px(735), w: px(1100), h: px(100), fontFace: F, fontSize: PT(31), bold: true, color: "000000", margin: 0, isTextBox: true });
  notes(s, "Hanyang University logo: paste from the first-round deck at the lower-left (x≈122, y≈843).");
}

// 2. 팀 역할 ---------------------------------------------------------------
{
  const s = base("TEAM ROLES");
  const names = ["Jaemin Lim", "Kyungho Kim", "Jaehyun Kim", "Junhyeok Choi"];
  const roles = [
    ["Statistical modeling", "ARIMA · SARIMA(X) first draft; search and lagged-mobility inputs; 13-week forecast setup"],
    ["Search data & dong labels", "427-dong labels (areas, stations, districts); Naver search collection; Seoul-share index and exclusion rules"],
    ["Ground truth & experiments", "Mobility target and cleaning; seasonality, model and input experiments; evaluation; HEAT system"],
    ["F&B & external data", "License data cleaning; openings, closures, net opening rate; subway exits; time alignment of inputs"]];
  names.forEach((n, i) => {
    const x = 60 + i * 455;
    s.addShape(pres.shapes.RECTANGLE, { x: px(x), y: px(200), w: px(425), h: px(660), fill: { color: "FFFFFF" }, line: { color: "C9CED6", width: 1 } });
    s.addText(n, { x: px(x + 25), y: px(225), w: px(375), h: px(70), fontFace: F, fontSize: PT(42), bold: true, color: C.navy, margin: 0, isTextBox: true });
    body(s, [{ label: "Role", text: [{ t: roles[i][0], b: true }] }, { label: "Contribution", text: roles[i][1] }], x + 25, 320, 375, 520, 36, 34);
  });
  bottom(s, "Four members, one pipeline: data → ground truth → forecasting → HEAT system");
}

// 3. 피드백 반영 ---------------------------------------------------------------
{
  const s = base("REFLECTION ON FEEDBACK");
  head(s, "What we heard", 40, 165, 760);
  head(s, "What we changed", 900, 165, 960, C.red);
  vdash(s, 870, 175, 905);
  const rowsFb = [
    ["Finalize the Hotness Score: weights, thresholds, persistence", 245, 300, [
      { label: "Ground truth", text: [{ t: "Hand-weighted score dropped. " }, { t: "Observed weekly arrivals per dong", b: true, u: true },
        { t: ", age 20–39 on shopping·leisure·other trips: people who visit, not residents or commuters" }] },
      { label: "Rules", text: "Hot = top 10% of the 13-week mean. Emerging = top 10% growth in Seoul share vs. the same 13 weeks last year; tiny dongs (< 2,050/day, fixed at 2023) excluded" }]],
    ["Test whether reasonable changes alter the results", 570, 100, [
      { label: "Robustness", text: [{ t: "Size floor recomputed at each date: " }, { t: `${N.sens_floor}% same emerging list`, b: true }] }]],
    ["Define a successful forecast", 690, 100, [
      { label: "Success", text: [{ t: "Beat " }, { t: "same week last year", b: true }, { t: " and " }, { t: "last week", b: true }, { t: " (13-week WAPE, DM test); beat persistence and random on top-10% hits" }] }]],
    ["Definition before data; one neighborhood example", 810, 90, [
      { label: "Example", text: "Definition first, then one dong end to end: Hongje 3-dong" }]]];
  rowsFb.forEach(([h, y, hh, items], i) => {
    body(s, [h], 40, y, 800, hh, 31);
    body(s, items, 900, y, 980, hh, 29, 12);
    if (i < rowsFb.length - 1) s.addShape(pres.shapes.LINE, { x: px(40), y: px(y + hh + 8), w: px(1840), h: 0, line: { color: "C9CED6", width: 0.75, dashType: "dash" } });
  });
  bottom(s, "From a hand-made score to an observable target with fixed, tested rules");
  notes(s, `Why mobility: it is measured, not weighted by us; weekly data gives ${N.n_weeks} weeks instead of 46 months; it is not one of our inputs. `
    + `If asked about age: with all ages only ${N.sens_age}% of the emerging list is the same, so the age band matters; we chose 20–39 leisure trips by purpose. Other age bands are future work.`);
}

// 4. 데이터 클리닝 ------------------------------------------------------------------
{
  const s = base("DATA CLEANING");
  head(s, "Anomaly: detected and rescaled", 40, 165, 1150);
  fitImg(s, "S1_anomaly.png", 20, 240, 1170, 660);
  body(s, [
    { label: "Anomaly", text: [{ t: "Aug–Oct 2024: mobility ×1.52 while card sales stayed flat → likely a data issue. " }, { t: "Divided by one factor from past data only", b: true }, { t: "; errors still scored on raw values" }] },
    { label: "Region mapping", text: "427 dongs on 2025 boundaries; mobility codes matched; Sinseol·Yongdu (split later) merged → 416 dongs with full history" },
    { label: "Time alignment", text: [{ t: `Daily → weekly mean (Mon–Sun, ≥ 4 valid days): ${N.n_weeks} weeks. ` }, { t: "Monthly search & F&B → value from 4 months earlier", b: true }, { t: "; last year = 364 days back (same weekday)" }] }],
    1215, 175, 670, 740, 29, 26);
  bottom(s, `Clean weekly panel: 416 dongs × ${N.n_weeks} weeks, no future information`);
}

// 5. 실험 -----------------------------------------------------------------------
{
  const s = base("WHAT WE TRIED");
  const H = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: C.navy } } });
  const L = (t) => ({ text: t, options: { bold: true, color: C.navy, fill: { color: C.yel } } });
  const T = (t, o = {}) => ({ text: t, options: { color: C.ink, ...o } });
  const R = (t) => T(t, { bold: true, color: C.red });
  const rows = [
    [H("Axis"), H("What we varied"), H("Finding (13-week WAPE, 4 dates)")],
    [L("Baselines"), T("last week · same week last year"), T(`last year ${N.sn4}% — the bar to beat`)],
    [L("Data"), T("raw · cleaned (Aug–Oct 2024 rescaled)"), R(`cleaning alone: ${N.sn4}% → ${N.snc4}%`)],
    [L("Seasonality"), T("none · 13-week AR · 52-week differencing · Fourier · last-year values"), T("52 weeks fits; no 13-week cycle in the data")],
    [L("Models"), T("ARIMA (fixed / AIC orders) · SARIMA · STL+ETS · Theta · 1:1 combination"), R(`combination best: ${N.best4}%`)],
    [L("Features"), T("search (index / Seoul share) · subway exits · F&B net openings · mobility 13 weeks ago · placebo"), T("no input lowered error")]];
  s.addTable(rows, { x: px(40), y: px(185), w: px(1840), colW: [px(300), px(860), px(680)], fontFace: F, fontSize: PT(29),
    border: { type: "solid", color: "C9CED6", pt: 0.75 }, rowH: px(112), valign: "middle", fill: { color: "FFFFFF" } });
  bottom(s, "Every setting: 4 forecast dates × 416 dongs, compared with both baselines");
  notes(s, "Metrics: WAPE = sum of absolute errors / sum of actuals over 13 weeks and all dongs; DM test for model vs. baseline; top-10% hit rate for hot and emerging lists. "
    + "4 dates: 2024-10-27, 2025-01-26, 2025-04-27, 2025-07-27. '2 clean dates' = the two not touched by the 2024 anomaly.");
}

// 6. 하나씩 고치기 -------------------------------------------------------------------
{
  const s = base("ONE FIX AT A TIME");
  head(s, "Start from our first draft, change one thing at a time (13-week WAPE, %)", 40, 165, 1840);
  fitImg(s, "S6_ladder.png", 30, 240, 1860, 670);
  bottom(s, "Seasonality, cleaning and combination cut error; search helped only the weak first draft");
  notes(s, `First draft = 13-week seasonal AR, fixed orders, raw search index, raw data. At 2025-04-27 search lowers error ${N.lad["검색X 04-27"][0]}% → ${N.lad["검색O 04-27"][0]}%, `
    + `but over 4 dates it rises ${N.lad["검색X 4시점"][0]}% → ${N.lad["검색O 4시점"][0]}%. After each fix the model without search is better.`);
}

// 7. 최고 모델 ---------------------------------------------------------------------
{
  const s = base("BEST MODEL SO FAR");
  fitImg(s, "S3_results.png", 20, 165, 1260, 760);
  head(s, "Same week last year + ARIMA·Fourier, averaged 1:1", 1300, 175, 590, C.red, 34);
  body(s, [
    { label: "Error", text: [{ t: `${N.best4}%`, b: true, u: true }, { t: ` vs. ${N.sn4}% for last year (4 dates); ${N.best2}% vs. ${N.sn2}% (2 clean dates)` }] },
    { label: "Per dong", text: `better than last year in ${N.best_imp}% of dong-dates (p < 0.001)` },
    { label: "+ Search", text: `${N.srch4}% / ${N.srch2}%; better in ${N.srch_imp}% of dong-dates → no average gain` },
    { label: "Hotspots", text: `emerging top-10% hit ${N.topk_model}% vs. persistence ${N.topk_pers}%, random 10%` }], 1300, 300, 590, 620, 31, 26);
  bottom(s, "Error almost halved vs. last year's values; search adds no average gain");
  notes(s, "4 dates = all forecast dates; 2 clean dates = 2025-01-26 and 2025-04-27, not touched by the 2024 anomaly.");
}

// 8. 사례 4곳 ----------------------------------------------------------------------
{
  const cs = N.case;
  const s = base("FOUR NEIGHBORHOODS (2025-04-27)");
  fitImg(s, "S8_cases.png", 15, 160, 1270, 775);
  body(s, [
    { label: "Best", text: [{ t: `Hongje 3-dong: forecast growth +${cs.rep_best.rep_growth_pct}%, actual +${cs.rep_best.actual_growth_pct}% → ` }, { t: "neither hot nor emerging, correct call", b: true }] },
    { label: "Worst", text: `Sinsu-dong: −60% overnight in June 2025, not seen last year → no model could see it; flagged for a data check` },
    { label: "Search helped", text: [{ t: `Chang 2-dong: error ${cs.search_best.wape}% → ${cs.search_best.wape_s}%; emerging rank ${cs.search_best.rep_emerging_rank} → ${cs.search_best.rep_s_emerging_rank} (actual ${cs.search_best.actual_emerging_rank}): ` }, { t: "caught a rising dong", b: true }] },
    { label: "Search hurt", text: [{ t: `Gayang 2-dong: error ${cs.search_worst.wape}% → ${cs.search_worst.wape_s}%; rank ${cs.search_worst.rep_emerging_rank} → ${cs.search_worst.rep_s_emerging_rank} (actual ${cs.search_worst.actual_emerging_rank}): ` }, { t: "false alarm", b: true }] }],
    1300, 170, 590, 760, 25, 14);
  bottom(s, "Search can flag a rising dong but also raises false alarms → a candidate to re-test");
  notes(s, `Chosen by rule among ${cs.rep_best.n} dongs above the size floor: lowest and highest error of our model; largest error drop and rise when search is added. `
    + `Emerging = top ${N.em_top} of ${N.n_em} dongs. Hongje 3-dong error ${cs.rep_best.wape}% vs. ${cs.rep_best.ly}% for last year's values.`);
}

// 9. 원인 분석 ---------------------------------------------------------------------
{
  const s = base("WHY? (OUR BEST GUESS)");
  head(s, "Why the best model works", 40, 165, 880);
  head(s, "Why extra data did not help", 980, 165, 900, C.red);
  vdash(s, 950, 175, 905);
  body(s, [
    { label: "Cleaning", text: "removes the 2024 jump that both last-year values and ARIMA would copy" },
    { label: "Season", text: "last year's same week carries the yearly pattern: holidays, semesters, weather" },
    { label: "Recent level", text: "ARIMA + Fourier follows this year's level with a smooth yearly curve" },
    { label: "Averaging", text: "the two make different errors; a fixed 1:1 average cancels part of them, nothing to overfit" }],
    40, 250, 880, 660, 35, 34);
  body(s, [
    { label: "Timing", text: "inputs move with mobility, not ahead; we may only use values 13 weeks or 4 months old" },
    { label: "Where vs. rising", text: "inputs match where it is busy, but barely where it is rising" },
    { label: "Short history", text: "~100 weeks per dong → noisy input effects" },
    { label: "Placebo", text: "over 4 dates, own search and another dong's search add the same error" }],
    980, 250, 900, 660, 35, 34);
  bottom(s, "So far, gains came from cleaning and seasonality, not from extra data");
  notes(s, `Placebo (first-draft setup): none ${N.pl.none["4시점"]}%, own search ${N.pl.search_index["4시점"]}%, another dong's search ${N.pl.placebo["4시점"]}% over 4 dates.`);
}

// 10. 다음 단계 -------------------------------------------------------------------
{
  const s = base("NEXT STEPS");
  const cols = [["Plan", 40, C.navy], ["Rules fixed in advance", 670, C.navy], ["Expected outcome", 1300, C.red]];
  cols.forEach(([t, x, c]) => head(s, t, x, 165, 600, c));
  vdash(s, 645); vdash(s, 1275);
  body(s, [
    { label: "Pooled model", text: [{ t: "LightGBM + Ridge on all dongs together", b: true }, { t: " so input effects can be estimated" }] },
    { label: "Inputs", text: "as changes (search growth, opening rate); fresher values for near weeks" },
    { label: "Target", text: "also rank emerging dongs directly" },
    { label: "Events", text: "concert & festival calendar for spikes" }], 40, 245, 580, 660, 32, 26);
  body(s, [
    { label: "Keep an input only if", text: "it lowers error on 4 dates and 2 clean dates, passes the DM test, and beats a placebo input" },
    { label: "Compare", text: "with today's best and with Ridge on the same features" },
    { label: "Then", text: [{ t: "freeze & commit, test once on 2026", b: true }] }], 670, 245, 580, 660, 32, 26);
  body(s, [
    { label: "Error", text: "about the same; most signal is already in mobility itself" },
    { label: "Ranking", text: "possible gain for emerging dongs, where search showed a signal (Chang 2-dong)" },
    { label: "Final", text: [{ t: "refit through Dec 2026 → 2027 H1 hotspots on the HEAT map", b: true, u: true }] }], 1300, 245, 580, 660, 32, 26);
  bottom(s, "Re-test extra data in a pooled model under fixed rules, then forecast 2027 H1");
}

// 10. 감사 -----------------------------------------------------------------------
{
  const s = pres.addSlide(); s.background = { color: C.bg };
  s.addImage({ path: FIG("seoul_map.png"), x: px(290), y: px(60), w: px(1338), h: px(900), sizing: { type: "contain", w: px(1338), h: px(900) } });
  s.addShape(pres.shapes.LINE, { x: 0, y: px(115), w: 20, h: 0, line: { color: C.line, width: 1 } });
  s.addShape(pres.shapes.LINE, { x: 0, y: px(1011), w: 20, h: 0, line: { color: C.line, width: 1 } });
  s.addShape(pres.shapes.LINE, { x: px(82), y: 0, w: 0, h: 11.25, line: { color: C.line, width: 1 } });
  s.addShape(pres.shapes.LINE, { x: px(1746), y: 0, w: 0, h: 11.25, line: { color: C.line, width: 1 } });
  s.addText([{ text: "THANK YOU", options: { fontSize: PT(140), color: C.title, bold: true, breakLine: true, shadow: { type: "outer", color: "777777", opacity: 0.5, blur: 2, offset: 3, angle: 45 } } },
    { text: "Team #14", options: { fontSize: PT(31), bold: true, color: C.title, breakLine: true } },
    { text: "Jaemin Lim, Kyungho Kim, Jaehyun Kim, Junhyeok Choi", options: { fontSize: PT(31), bold: true, color: C.title } }],
    { x: 0, y: px(400), w: 20, h: px(400), fontFace: F, align: "center", valign: "middle", margin: 0, isTextBox: true });
}

// 부록 ---------------------------------------------------------------------------
{
  const s = base("APPENDIX: RESULTS BY FORECAST DATE");
  const rows = [["Model (13-week WAPE, %)", ...N.origins, "4 dates", "2 clean"]].concat(N.table);
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: String(c), options: { bold: i === 0 || (i === rows.length - 1), color: i === rows.length - 1 ? C.red : C.ink,
      fill: { color: i === 0 ? "E3E9F3" : "FFFFFF" }, align: j === 0 ? "left" : "center" } }))),
    { x: px(60), y: px(190), w: px(1800), colW: [px(620), px(200), px(200), px(200), px(200), px(190), px(190)], fontFace: F, fontSize: PT(30), border: { type: "solid", color: "C9CED6", pt: 0.75 }, rowH: px(62) });
  bottom(s, "'cleaned' = trained on rescaled Aug–Oct 2024; all errors vs. raw actuals");
}
if (N.ratio["S4_inputs.png"]) {
  const s = base("APPENDIX: INPUT COMBINATIONS (RAW DATA)");
  fitImg(s, "S4_inputs.png", 60, 170, 1800, 740);
  bottom(s, "Seven input sets added to the same ARIMA: none lowered error");
}
if (N.pl && N.ratio["S7_placebo.png"]) {
  const s = base("APPENDIX: PLACEBO INPUT TEST");
  head(s, "First-draft setup: does any input lower error? (13-week WAPE)", 40, 165, 1840);
  fitImg(s, "S7_placebo.png", 30, 240, 1860, 670);
  bottom(s, N.pl_msg || "");
}
if (N.ratio["heat_system.png"]) {
  const s = base("APPENDIX: HEAT SYSTEM");
  fitImg(s, "heat_system.png", 60, 170, 1800, 740);
  bottom(s, "Hotness index, emerging dongs and 13-week forecasts for all 427 dongs");
}

pres.writeFile({ fileName: path.join(__dirname, "HEAT_midterm.pptx") }).then((f) => console.log("saved", f));
