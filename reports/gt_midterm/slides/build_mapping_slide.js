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


// 데이터 매핑: 행정동 이름표 기준 (따로 붙여 넣을 한 장)
{
  const s = base("DATA MAPPING");
  head(s, "Each dong gets search names in three slots, linked only by official boundaries and area", 40, 165, 1840, C.navy, 38);
  const H = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: C.navy } } });
  const L = (t) => ({ text: t, options: { bold: true, color: C.navy, fill: { color: C.yel } } });
  const T = (t, o = {}) => ({ text: t, options: { color: C.ink, ...o } });
  const B = (t) => T(t, { bold: true });
  const rows = [
    [H("Slot"), H("Names from"), H("Linked to a dong when"), H("Why")],
    [L("Neighborhood"), T("Legal-dong names (+ admin-dong name if different)"), B("≥ 10% of the dong's land area"), T("names people actually search; tiny slivers dropped")],
    [L("Station"), T("Seoul station list"), B("≥ 5% of the dong's land within 1,000 m, same side of the Han River"), T("about 15 minutes on foot; the river blocks access")],
    [L("Commercial area"), T("Tourist zones, nicknames (Hongdae), '~gil' streets"), B("≥ 10% of the area lies in the dong"), T("hotspot names that are not dong names (Garosu-gil)")]];
  s.addTable(rows, { x: px(40), y: px(250), w: px(1840), colW: [px(300), px(500), px(560), px(480)], fontFace: F, fontSize: PT(30),
    border: { type: "solid", color: "C9CED6", pt: 0.75 }, rowH: [px(80), px(150), px(150), px(150)], valign: "middle", fill: { color: "FFFFFF" } });
  body(s, [{ label: "Weight", text: "share of the dong's land area each name covers; land = dong area minus the Han River; weights sum to 1 in each slot" }],
    40, 835, 1840, 80, 30, 0);
  bottom(s, "Boundaries and area only: simple, fixed over time, no overlap with F&B data");
  notes(s, "427 dongs (2025-12-31 boundaries). Names: 347 neighborhood, 315 station, 57 commercial-area names; 566 keyword groups. "
    + "Each name + six visit words (restaurants, cafe, bar, date, places to go, hotplace) so the search reflects interest in going there. "
    + "Station radius 500 m and 1,500 m kept as sensitivity checks. Example: Seogyo-dong = Seogyo 74% / Donggyo 26%; Hongdae station area; Hongdae commercial area.");
}
pres.writeFile({ fileName: path.join(__dirname, "HEAT_mapping_slide.pptx") }).then(f => console.log("saved", f));
