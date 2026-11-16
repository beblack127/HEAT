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


// 데이터 매핑: 행정동 이름표 만드는 과정 (따로 붙여 넣을 한 장)
{
  const s = base("DATA MAPPING");
  const steps = [
    ["Boundaries", "427 dongs (2025); land = dong area minus the Han River"],
    ["Overlay names", "legal dongs · 1,000 m station circles (same river side) · commercial areas"],
    ["Link if large enough", "share of the dong's land: ≥ 10% · ≥ 5% · ≥ 10%"],
    ["Weight → keywords", "area share, sums to 1 per slot; each name + 6 visit words → Naver"]];
  steps.forEach(([h, t], i) => {
    const x = 40 + i * 470;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: px(x), y: px(172), w: px(420), h: px(165), fill: { color: "FFFFFF" }, line: { color: C.navy, width: 1.25 }, rectRadius: 0.08 });
    s.addText([{ text: `${i + 1}  ${h}`, options: { bold: true, color: C.navy, fontSize: PT(31), breakLine: true } }, { text: t, options: { color: C.body, fontSize: PT(24) } }],
      { x: px(x + 18), y: px(180), w: px(390), h: px(150), fontFace: F, valign: "top", margin: 0, isTextBox: true });
    if (i < 3) s.addText("→", { x: px(x + 422), y: px(220), w: px(46), h: px(60), fontFace: F, fontSize: PT(40), bold: true, color: C.navy, align: "center", margin: 0, isTextBox: true });
  });
  s.addText("Example: Sageun-dong (Seongdong-gu)", { x: px(40), y: px(352), w: px(900), h: px(50), fontFace: F, fontSize: PT(30), bold: true, color: C.red, margin: 0, isTextBox: true });
  s.addImage({ path: FIG("S10_mapping.png"), x: px(40), y: px(405), w: px(1840), h: px(1840 / 4.510) });
  [["Neighborhood: legal dongs ≥ 10% of land", 40], ["Station: ≥ 5% of land within 1,000 m", 653], ["Commercial area: ≥ 10% of it in the dong", 1266]].forEach(([t, x]) =>
    s.addText(t, { x: px(x), y: px(835), w: px(614), h: px(45), fontFace: F, fontSize: PT(26), bold: true, color: C.navy, align: "center", margin: 0, isTextBox: true }));
  bottom(s, "Boundaries and area only: simple, fixed over time, no overlap with F&B data");
  notes(s, "Left map shows area shares (Sageun 56%, Haengdang 39%; Seongsu, Yongdap, Majang about 2% each dropped); after dropping slivers the weights are 59% / 41%. Station labels show final weights (9 stations linked, top 4 labeled). No commercial area falls in Sageun-dong, so that slot stays empty. "
    + "Totals: 347 neighborhood, 315 station, 57 commercial-area names; 566 keyword groups. Visit words: restaurants, cafe, bar, date, places to go, hotplace. "
    + "Station radius 500 m and 1,500 m kept as sensitivity checks.");
}
pres.writeFile({ fileName: path.join(__dirname, "HEAT_mapping_slide.pptx") }).then(f => console.log("saved", f));
