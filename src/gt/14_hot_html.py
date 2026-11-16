"""핫플지수 전용 HTML (2026-11-16): 검색량·개폐업·이름표 탭과 자료를 빼고, 핫플지수·떠오름만 남긴다.
- 핫플지수 색: 초록 계열 (떠오름 색은 그대로)
- 예측 그래프: Best, Best + Search (실시간). 4개 시점 확인 그래프에만 Baseline(작년 같은 주, 점선)
- 13주 예측 탭, 설명 문장 제거. 이름표는 남기고 근거·검색어 제거
원본 템플릿(src/nametags/app_v4.html)은 그대로 두고 문자열 치환으로 만든다.
사용: python3 src/gt/14_hot_html.py 출력.html
"""
import json, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
NT = ROOT / "src/nametags"
t = (NT / "app_v4.html").read_text(encoding="utf-8")


def rep(old, new, count=1):
    global t
    assert old in t, old[:80]
    t = t.replace(old, new, count)


# 1. 색: 핫플지수 초록 계열
rep("--h1:#F3EFE9; --h2:#F8D9BE; --h3:#F4B78C; --h4:#EC8E5C; --h5:#DA6136; --h6:#B33D22; --h7:#7A2513;",
    "--h1:#EEF4EF; --h2:#CFE6D3; --h3:#A6D3AF; --h4:#74BA86; --h5:#46A064; --h6:#23804A; --h7:#0F5A33;")
rep("--h1:#2A2622; --h2:#4A3226; --h3:#71422B; --h4:#9A5330; --h5:#C46536; --h6:#E2834F; --h7:#F6B183;",
    "--h1:#1F2A22; --h2:#24412C; --h3:#2C5E3B; --h4:#36794B; --h5:#49965E; --h6:#6DB57D; --h7:#A2D6AE;")
# 2. 다른 탭 숨김
rep("</style>\n</head>", ".seg[role=tablist],#demo,.sctl,.fctl{display:none!important}\n</style>\n</head>")
rep('<div class="brand"><b>HEAT</b><span>서울 동네 지도</span></div>', '<div class="brand"><b>HEAT</b><span>서울 핫플 지도</span></div>')
# 3. 13주 예측 탭 제거
rep('<button data-v="grow">떠오름</button><button data-v="fc">13주 예측</button>', '<button data-v="grow">떠오름</button>')
# 4. 안내 문장 제거
rep("document.getElementById('h_note').textContent = fc ? `마지막 주(${wk(HT.weeks[HT.weeks.length - 1])}) 이후 13주를 ${HT.fc.model}로 예측했어요.` : '최근 13주 평균 기준이에요. 슬라이더로 기준 주를 바꿀 수 있어요.';",
    "document.getElementById('h_note').textContent = '';")
a = t.index('<details class="help">'); b = t.index("</details>", a) + len("</details>")
t = t[:a] + '''<details class="help"><summary>보는 법</summary>
        <p><b>핫플지수</b>: 20~39세가 쇼핑·여가 목적으로 그 동에 온 하루 평균 인원(최근 13주)을 서울 동 순위로 바꾼 값(0~100). 90 이상이 현재 핫스팟이에요.</p>
        <p><b>떠오름</b>: 서울 안 비중이 1년 전 같은 13주보다 늘어난 정도예요. 규모가 작은 동을 빼고 상위 10%가 떠오르는 곳(검은 테두리)이에요.</p>
      </details>''' + t[b:]

# 5. 동 상세: 예측 칸 이름
rep("<small>다음 13주 예측 <em>${wk(HT.fc.weeks[0])}~</em></small><b>${fmtn(fs.lvl[u])}</b><span class=\"sub\">하루 평균 · 생활이동만(${esc(HT.fc.model)})${HT.fc.y2 && HT.fc.y2[u] ? `<br>검색량 넣으면 ${fmtn(HT.fc.y2[u].reduce((a, b) => a + b, 0) / 13)}` : ''}</span>",
    "<small>다음 13주 예측 <em>${wk(HT.fc.weeks[0])}~</em></small><b>${fmtn(fs.lvl[u])}</b><span class=\"sub\">하루 평균</span>")
rep("<div class=\"kpi\"><small>예측 떠오름</small>", "<div class=\"kpi\"><small>예측 떠오름 <em>1년 전 같은 13주 대비</em></small>")
# 6. 주 그래프: 실제 + 예측 세 개
old_ser = t[t.index("  const ser = [{name: '주별 실제'"):t.index("  const a0 = Math.max(0")]
t = t.replace(old_ser, """  const fwd = a => Array(n - 1).fill(null).concat([d.y[n - 1]], a);
  const ser = [{name: '실제', color: css('--sta'), width: 1.8, data: pad(d.y), fmt: fmtn},
    {name: 'Best', color: css('--c5'), width: 2.4, data: fwd(HT.fc.y[u] || Array(13).fill(null)), fmt: fmtn}];
  if (HT.fc.y2 && HT.fc.y2[u]) ser.push({name: 'Best + Search', color: '#7B4FB8', width: 2.2, data: fwd(HT.fc.y2[u]), fmt: fmtn});
""")
rep("<b>주별 생활이동과 13주 예측</b><span><span class=\"key\"><i style=\"background:${css('--sta')}\"></i>실제</span><span class=\"key\"><i style=\"background:${css('--c5')}\"></i>생활이동만</span><span class=\"key\"><i style=\"background:#7B4FB8\"></i>+검색량</span></span>",
    "<b>주별 생활이동과 13주 예측</b><span><span class=\"key\"><i style=\"background:${css('--sta')}\"></i>실제</span><span class=\"key\"><i style=\"background:${css('--c5')}\"></i>Best</span><span class=\"key\"><i style=\"background:#7B4FB8\"></i>Best + Search</span></span>")
a = t.index('    <p class="note">2024년 8~10월은'); b = t.index("</p>`;", a) + len("</p>")
t = t[:a] + t[b:]
# 7. 실험 비교: 표 + 4개 시점 그래프 (Best, Best + Search, Baseline 점선)
a = t.index("  const E = HT.exp;"); b = t.index("  return h;\n}", a)
t = t[:a] + """  const E = HT.exp;
  if (E && E.y[u]) {
    const wt = E.wape, P2 = E.y2 && E.y2[u];
    const mn = o => Math.min(wt.main[o], wt.cmp[o], wt.snaive[o]);
    const row = (lab, v) => `<tr><td>${lab}</td>${E.origins.map(o => `<td class="${v[o] === mn(o) ? 'b' : ''}">${(+v[o]).toFixed(1)}%</td>`).join('')}</tr>`;
    h += `<details open><summary>지난 예측 확인 (4개 시점)</summary>
      ${wt ? `<table class="wt"><tr><th>서울 전체 13주 오차</th>${E.origins.map(o => `<th>${o.slice(2, 7).replace('-', '.')}</th>`).join('')}</tr>
      ${row('Best', wt.main)}${row('Best + Search', wt.cmp)}${row('Baseline', wt.snaive)}</table>` : ''}` + E.origins.map(o => {
      const oi = HT.weeks.indexOf(o), p = E.y[u][o]; if (oi < 0 || !p) return '';
      const a = Math.max(0, oi - 20), b = Math.min(n - 1, oi + 13), ms = HT.weeks.slice(a, b + 1), act = d.y.slice(a, b + 1);
      const pr = ms.map((_, j) => j + a > oi && j + a <= oi + 13 ? p[j + a - oi - 1] : (j + a === oi ? d.y[oi] : null));
      const sn = ms.map((_, j) => j + a > oi && j + a - 52 >= 0 ? d.y[j + a - 52] : null);
      const q2 = P2 && P2[o], pr2 = q2 ? ms.map((_, j) => j + a > oi && j + a <= oi + 13 ? q2[j + a - oi - 1] : (j + a === oi ? d.y[oi] : null)) : null;
      let ea = 0, eb = 0, ec = 0, sy = 0; for (let k = 1; k <= 13 && oi + k < n; k++) { const y = d.y[oi + k]; if (y == null) continue; ea += Math.abs(y - p[k - 1]); eb += Math.abs(y - d.y[oi + k - 52]); if (q2) ec += Math.abs(y - q2[k - 1]); sy += y; }
      const pc = v => sy ? (v / sy * 100).toFixed(1) + '%' : '–', best = Math.min(ea, q2 ? ec : Infinity, eb);
      const sr = [{name: '실제', color: css('--sta'), width: 1.6, data: act, fmt: fmtn}, {name: 'Baseline', color: '#8A8F98', width: 1.4, dash: '4 3', data: sn, fmt: fmtn},
        {name: 'Best', color: css('--c5'), width: 2.2, data: pr, fmt: fmtn}];
      if (pr2) sr.push({name: 'Best + Search', color: '#7B4FB8', width: 2, data: pr2, fmt: fmtn});
      return `<div class="ch"><div class="chh"><b>${wk(o)} 시점</b><span>Best <b class="${ea === best ? 'win' : ''}">${pc(ea)}</b> · Best + Search <b class="${q2 && ec === best ? 'win' : ''}">${q2 ? pc(ec) : '–'}</b> · Baseline <b class="${eb === best ? 'win' : ''}">${pc(eb)}</b></span></div>${lineChart(sr, {w: W, h: 130, small: true, months: ms, i: oi - a, tf: fmtK, tick: (m, j, mm) => j % 8 === 0 ? `${+m.slice(5, 7)}월` : null, fx: wkFx, shade: [oi - a, ms.length - 1]})}</div>`; }).join('')
      + `<div class="chh"><span><span class="key"><i style="background:${css('--sta')}"></i>실제</span><span class="key"><i style="background:${css('--c5')}"></i>Best</span><span class="key"><i style="background:#7B4FB8"></i>Best + Search</span><span class="key"><i style="background:#8A8F98"></i>Baseline (점선)</span></span></div></details>`;
  }
""" + t[b:]
# 선 그래프 점선 지원
rep('stroke-width="${s.width}" stroke-opacity="${s.op || 1}"', 'stroke-width="${s.width}" ${s.dash ? `stroke-dasharray="${s.dash}"` : \'\'} stroke-opacity="${s.op || 1}"')
# 8. 이름표는 남기고 '근거와 검색어'만 제거
rep("    + `<details><summary>근거와 검색어</summary><div class=\"why\">${why}</div></details>`;", ";")

t = t.replace("__LEAFLET_CSS__", (NT / "leaflet_css.html").read_text(encoding="utf-8"))
t = t.replace("__LEAFLET_JS__", (NT / "leaflet_js.html").read_text(encoding="utf-8"))
m = json.load(open(ROOT / "data/processed/nametags/_mapdata_small.json", encoding="utf-8"))
m["landmarks"] = json.load(open(ROOT / "data/external/geo/landmarks.json", encoding="utf-8"))
m["hot"] = json.load(open(ROOT / "data/processed/gt/hot_map.json", encoding="utf-8"))
data = json.dumps(m, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
t = t.replace("__DATA__", data)
out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "maps/HEAT_hot.html")
out.write_text(t, encoding="utf-8")
print("written", out, round(len(t.encode()) / 1e6, 2), "MB")
