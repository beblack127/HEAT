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
  names.forEach((n, i) => {
    const x = 60 + i * 455;
    s.addShape(pres.shapes.RECTANGLE, { x: px(x), y: px(200), w: px(425), h: px(660), fill: { color: "FFFFFF" }, line: { color: "C9CED6", width: 1 } });
    s.addText(n, { x: px(x + 25), y: px(225), w: px(375), h: px(70), fontFace: F, fontSize: PT(42), bold: true, color: C.navy, margin: 0, isTextBox: true });
    body(s, [{ label: "Role", text: "(to be filled)" }, { label: "Contribution", text: "(to be filled)" }], x + 25, 320, 375, 520, 30, 18);
  });
  bottom(s, "Four members, one pipeline: data → ground truth → forecasting → HEAT system");
}

// 3. 피드백 반영 (1/2) ---------------------------------------------------------
{
  const s = base("REFLECTION ON FEEDBACK (1/2)");
  head(s, "What we heard", 40, 165, 760);
  head(s, "What we changed", 960, 165, 900, C.red);
  vdash(s, 930, 175, 905);
  const fb = [
    ["Finalize the Hotness Score: weights, thresholds, persistence", 248],
    ["Test whether reasonable changes alter the results", 470],
    ["Define a successful forecast", 590],
    ["Definition before data; one neighborhood example", 745]];
  fb.forEach(([t, y]) => body(s, [t], 40, y, 860, 140, 31));
  body(s, [
    { label: "No composite score", text: [{ t: "We dropped hand-weighted scores. " }, { t: "Ground truth = observed weekly arrivals", b: true, u: true }, { t: " (age 20–39, shopping·leisure·other) per dong" }] },
    { label: "Thresholds", text: [{ t: "Hotspot = top 10% of 13-week mean. Emerging = top 10% growth of Seoul share vs. same 13 weeks last year, size floor fixed at 2023 (≥ 2,050/day). 13-week window = persistence." }] },
    { label: "Robustness", text: [{ t: `Floor rule re-computed each time: ${N.sens_floor}% same list · all ages: ${N.sens_age}% · no floor: ${N.sens_nofloor}%` }] },
    { label: "Success", text: [{ t: "Beat ", }, { t: "same-week-last-year", b: true }, { t: " and " }, { t: "last-week", b: true }, { t: " baselines (WAPE, DM test) and beat random & persistence on top-10% hit rate" }] },
    { label: "Structure", text: [{ t: "This deck: definition → validity → Garak 1-dong example" }] }],
    960, 248, 920, 660, 29, 12);
  bottom(s, "From a hand-made score to an observable target with fixed, tested rules");
  notes(s, "Feedback 1 and 3. We replaced the Hotness Score with an observable target, fixed every threshold in advance, and tested alternative rules.");
}

// 4. 피드백 반영 (2/2): 타당성 -----------------------------------------------------
{
  const s = base("REFLECTION ON FEEDBACK (2/2)");
  head(s, "Is mobility a valid measure of 'hot'?", 40, 165, 1100);
  fitImg(s, "S2_validity.png", 20, 235, 1170, 670);
  body(s, [
    { label: "Beyond inputs", text: [{ t: `Card sales (not a model input) rank-correlate ` }, { t: `${N.sales_level}`, b: true, u: true }, { t: ` with mobility; ${N.sales_top}% of top-10% dongs match` }] },
    { label: "Overlap", text: [{ t: "Living population (same telecom source) removed from inputs; every input lagged ≥ 13 weeks" }] },
    { label: "Short history", text: [{ t: `Monthly → weekly target: ${N.n_weeks} weeks instead of 46 months` }] },
    { label: "Added value", text: [{ t: "Maps & search show where it is busy " }, { t: "now", b: true }, { t: "; HEAT forecasts " }, { t: "13 weeks ahead", b: true }, { t: " and flags rising dongs, validated by card sales" }] }],
    1215, 230, 670, 690, 27, 14);
  bottom(s, "Mobility agrees strongly on where it is hot, weakly on where it is rising");
  notes(s, "Feedback 2. Level: strong agreement with card sales, F&B stores and subway exits. Change: weak but positive. This is also why emerging-place forecasting is hard.");
}

// 5. 동네 예시 -------------------------------------------------------------------
{
  const s = base("ONE NEIGHBORHOOD: GARAK 1-DONG");
  fitImg(s, "S5_example.png", 20, 165, 1250, 750);
  head(s, "Definition", 1290, 175, 600);
  body(s, [`Emerging on 2025-04-27: Seoul share +${N.ex_growth}% vs. last year (rank ${N.ex_rank} of ${N.ex_n})`, `Card F&B sales share +${N.ex_sales}% → commercial, not residential`], 1290, 245, 600, 230, 26, 8);
  head(s, "3-month forecast", 1290, 470, 600);
  body(s, [`Our forecast error ${N.ex_wape}% vs. ${N.ex_sn}% for last year's values`, "Stayed emerging in the next 13 weeks"], 1290, 540, 600, 200, 26, 8);
  head(s, "Decision", 1290, 720, 600, C.red);
  body(s, ["Shortlist for store opening or district support 3 months early"], 1290, 790, 600, 120, 26, 8);
  bottom(s, "Definition → 13-week forecast → decision, in one neighborhood");
  notes(s, "Chosen by rule: emerging at the decision date and in the next 13 weeks, card F&B sales share up >5%, smallest forecast error.");
}

// 6. 진행 상황 -------------------------------------------------------------------
{
  const s = base("PROGRESS SO FAR");
  const cols = [["Data cleaning", 40], ["Feature engineering", 670], ["Initial modeling", 1300]];
  cols.forEach(([t, x]) => head(s, t, x, 165, 600));
  vdash(s, 645); vdash(s, 1275);
  body(s, [
    { label: "Unit", text: "427 dongs (name-tag map), split dongs merged" },
    { label: "Target", text: "daily arrivals → week mean (≥ 4 valid days)" },
    { label: "Level shift", text: [{ t: "Aug–Oct 2024 ×1.52 in mobility only → rescaled", b: true }] },
    { label: "Leakage", text: "every input checked to end before the forecast date" }], 40, 245, 580, 640, 31, 22);
  body(s, [
    { label: "Search", text: "Seoul share, month −4" },
    { label: "Subway", text: "exits, 4-week mean, week −13" },
    { label: "F&B", text: "net opening rate (3 months), month −4" },
    { label: "Calendar", text: "holidays; same week last year" },
    { label: "Scaling", text: "standardized on training data only" }], 670, 245, 580, 640, 31, 22);
  body(s, [
    { label: "Baselines", text: "last week; same week last year" },
    { label: "Classical", text: "ARIMA / SARIMA(X), per-dong AIC orders" },
    { label: "Seasonal", text: "ARIMA + Fourier, STL + ETS, Theta" },
    { label: "Combination", text: "last year + model, fixed 1:1" },
    { label: "Design", text: "4 forecast dates × 13 weeks × 416 dongs" }], 1300, 245, 580, 640, 31, 22);
  bottom(s, "Weekly, leakage-checked pipeline from raw files to 13-week forecasts");
}

// 7. 결과 1 ----------------------------------------------------------------------
{
  const s = base("RESULTS: BASELINES AND METRICS");
  fitImg(s, "S3_results.png", 30, 160, 1240, 760);
  head(s, "Key numbers", 1300, 175, 580);
  body(s, [
    { label: "Best", text: [{ t: `Combination ${N.best4}%`, b: true, u: true }, { t: ` vs. last year ${N.sn4}% (4 dates)` }] },
    { label: "Clean dates", text: `${N.best2}% vs. ${N.sn2}% (2 dates without the anomaly)` },
    { label: "Per dong", text: `better than last year in ${N.best_imp}% of dong-dates (p < 0.001)` },
    { label: "Cleaning", text: `same-week-last-year ${N.sn4}% → ${N.snc4}%` }], 1300, 250, 590, 650, 27, 16);
  bottom(s, "First model to beat the same-week-last-year baseline");
  notes(s, "WAPE = sum of absolute errors / sum of actuals over 13 weeks, all dongs. Evaluated against the raw (uncleaned) actuals.");
}

// 8. 결과 2: 부정적 결과 --------------------------------------------------------------
{
  const s = base("RESULTS: NEGATIVE FINDINGS");
  head(s, "External inputs did not reduce error", 40, 165, 1250);
  fitImg(s, "S4_inputs.png", 30, 240, 1240, 640);
  head(s, "Why we think so", 1300, 175, 580, C.red);
  body(s, [
    { label: "Timing", text: "inputs move with mobility, not ahead; a 13-week-old value adds little to mobility's own past" },
    { label: "Data per dong", text: "~100 training weeks → noisy coefficients" },
    { label: "Error source", text: "seasons and data shifts, not what inputs measure" },
    { label: "Ranking", text: `emerging top-10%: ${N.topk_model}% hit vs. random 10%, persistence ${N.topk_pers}%` }], 1300, 250, 590, 650, 26, 14);
  bottom(s, "Inputs may still help ranking, not error: next we test them in a pooled model");
  notes(s, "All input sets were added to the same per-dong ARIMA orders. Hot top-10% hit ~95% but persistence is also ~95%.");
}

// 8-2. 결과 3: 검색이 왜 좋아 보였나 (전체 조합 실험) ------------------------------------
{
  const s = base("RESULTS: WHY SEARCH SEEMED TO HELP");
  head(s, "Teammate's setup first, then one fix at a time (13-week WAPE, %)", 40, 165, 1840);
  fitImg(s, "S6_ladder.png", 30, 240, 1860, 670);
  bottom(s, "Search lowered error only on top of a weak model; with better models it adds error");
  notes(s, `Teammate setup = 13-week seasonal AR, fixed orders, raw search index, raw data. At 2025-04-27 search lowers error ${N.lad["검색X 04-27"][0]}% → ${N.lad["검색O 04-27"][0]}%, `
    + `but averaged over 4 dates it rises ${N.lad["검색X 4시점"][0]}% → ${N.lad["검색O 4시점"][0]}%. After every fix the model without search is better. `
    + (N.pl ? `Placebo: another dong's search index gives ${N.pl.placebo["2025-04-27"]}% vs. own search ${N.pl.search_index["2025-04-27"]}% and none ${N.pl.none["2025-04-27"]}% at 2025-04-27.` : ""));
}

// 9. 다음 단계 -------------------------------------------------------------------
{
  const s = base("NEXT STEPS");
  const cols = [["Evaluation plan", 40, C.navy], ["Remaining steps", 670, C.navy], ["Expected outcome", 1300, C.red]];
  cols.forEach(([t, x, c]) => head(s, t, x, 165, 600, c));
  vdash(s, 645); vdash(s, 1275);
  body(s, [
    { label: "More dates", text: "monthly forecast dates, 2024-07 ~ 2025-07" },
    { label: "Longer range", text: "26–34 weeks ahead, to reach 2027 H1" },
    { label: "One-shot test", text: [{ t: "freeze & commit, then test once on 2026", b: true }] }], 40, 245, 580, 640, 32, 26);
  body(s, [
    { label: "Pooled model", text: "LightGBM + Ridge across dongs, inputs as changes" },
    { label: "Events", text: "concert & festival calendars to separate spikes" },
    { label: "Data check", text: "spring-2025 level shift; source inquiry" },
    { label: "Inputs", text: "change how inputs enter; keep only what helps" }], 670, 245, 580, 640, 32, 26);
  body(s, [
    { label: "Forecast", text: [{ t: "2027 H1 hotspots and emerging dongs", b: true, u: true }] },
    { label: "System", text: "HEAT map with weekly forecasts" },
    { label: "Success", text: "beat last year & persistence on the 2026 test" }], 1300, 245, 580, 640, 32, 26);
  bottom(s, "Refit through Dec 2026 and forecast Seoul's 2027 H1 hotspots");
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
  const s = base("APPENDIX: DATA ANOMALY");
  fitImg(s, "S1_anomaly.png", 60, 170, 1800, 720);
  bottom(s, "Aug–Oct 2024: mobility ×1.52 while card sales stayed flat → rescaled");
}
{
  const s = base("APPENDIX: RESULTS BY FORECAST DATE");
  const rows = [["Model (13-week WAPE, %)", ...N.origins, "4 dates", "2 clean"]].concat(N.table);
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: String(c), options: { bold: i === 0 || (i === rows.length - 1), color: i === rows.length - 1 ? C.red : C.ink,
      fill: { color: i === 0 ? "E3E9F3" : "FFFFFF" }, align: j === 0 ? "left" : "center" } }))),
    { x: px(60), y: px(190), w: px(1800), colW: [px(620), px(200), px(200), px(200), px(200), px(190), px(190)], fontFace: F, fontSize: PT(30), border: { type: "solid", color: "C9CED6", pt: 0.75 }, rowH: px(62) });
  bottom(s, "'cleaned' = trained on rescaled Aug–Oct 2024; all errors vs. raw actuals");
}
if (N.pl && N.ratio["S7_placebo.png"]) {
  const s = base("APPENDIX: PLACEBO INPUT TEST");
  head(s, "Teammate's setup: does any input lower error? (13-week WAPE)", 40, 165, 1840);
  fitImg(s, "S7_placebo.png", 30, 240, 1860, 670);
  bottom(s, N.pl_msg || "");
}
if (N.ratio["heat_system.png"]) {
  const s = base("APPENDIX: HEAT SYSTEM");
  fitImg(s, "heat_system.png", 60, 170, 1800, 740);
  bottom(s, "Hotness index, emerging dongs and 13-week forecasts for all 427 dongs");
}

pres.writeFile({ fileName: path.join(__dirname, "HEAT_midterm.pptx") }).then((f) => console.log("saved", f));
