#!/usr/bin/env python3
"""Create district-level annual and monthly restaurant event comparisons for 2023-2025."""

import csv
from collections import Counter, defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "EDA_서울전체흐름" / "지역구별_월별_집계.csv"
OUT = ROOT / "EDA_지역별분석"
GUS = "종로구 중구 용산구 성동구 광진구 동대문구 중랑구 성북구 강북구 도봉구 노원구 은평구 서대문구 마포구 양천구 강서구 구로구 금천구 영등포구 동작구 관악구 서초구 강남구 송파구 강동구".split()
SOURCES = ("일반음식점", "휴게음식점")
EVENTS = ("개업", "폐업")
MONTHS = [f"{year:04d}-{month:02d}" for year in range(2022, 2026) for month in range(1, 13)]
FOCUS_MONTHS = [month for month in MONTHS if month.startswith(("2023", "2024", "2025"))]


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pct_change(current, previous):
    return round((current / previous - 1) * 100, 3) if previous else ""


def color_for_change(value, limit=100):
    if value is None:
        return "#e5e7eb"
    clipped = max(-limit, min(limit, value)) / limit
    mag = abs(clipped)
    if clipped >= 0:
        start, end = (239, 246, 255), (29, 78, 216)
    else:
        start, end = (255, 241, 242), (190, 18, 60)
    rgb = tuple(round(a + (b - a) * mag) for a, b in zip(start, end))
    return "#%02x%02x%02x" % rgb


def heatmap_svg(path, title, subtitle, values, columns, label_kind):
    left, top, cell_h = 112, 105, 22
    cell_w = 78 if label_kind == "annual" else 19
    width = left + len(columns) * cell_w + 38
    height = top + len(GUS) * cell_h + 64
    bits = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            '<style>text{font-family:Arial,"Noto Sans KR",sans-serif;fill:#172033}.title{font-size:18px;font-weight:bold}.sub{font-size:11px;fill:#475569}.lab{font-size:11px}.celltext{font-size:10px}</style>',
            f'<text class="title" x="{left}" y="28">{escape(title)}</text>',
            f'<text class="sub" x="{left}" y="49">{escape(subtitle)}</text>']
    if label_kind == "annual":
        year_cols = ["2023", "2024", "2025"]
        for ci, col in enumerate(year_cols):
            x = left + ci * cell_w + cell_w / 2
            bits.append(f'<text class="lab" text-anchor="middle" x="{x}" y="86">{col}</text>')
    else:
        for year_i, year in enumerate((2023, 2024, 2025)):
            x = left + year_i * 12 * cell_w
            bits.append(f'<text class="lab" text-anchor="middle" x="{x+6*cell_w}" y="75">{year}</text>')
            bits.append(f'<line x1="{x}" y1="80" x2="{x}" y2="{top+len(GUS)*cell_h}" stroke="#94a3b8"/>')
        for ci in range(len(columns)):
            x = left + ci * cell_w + cell_w / 2
            bits.append(f'<text class="lab" text-anchor="middle" x="{x}" y="94">{ci%12+1}</text>')
    for ri, gu in enumerate(GUS):
        y = top + ri * cell_h
        bits.append(f'<text class="lab" text-anchor="end" x="{left-8}" y="{y+15}">{escape(gu)}</text>')
        for ci, col in enumerate(columns):
            value = values.get((gu, col))
            fill = color_for_change(value)
            x = left + ci * cell_w
            bits.append(f'<rect x="{x}" y="{y}" width="{cell_w-1}" height="{cell_h-2}" rx="1" fill="{fill}"/>')
            if label_kind == "annual" and value is not None:
                bits.append(f'<text class="celltext" text-anchor="middle" x="{x+cell_w/2}" y="{y+14}">{value:+.0f}%</text>')
    legend_y = top + len(GUS) * cell_h + 25
    bits.append(f'<text class="lab" x="{left}" y="{legend_y+4}">감소</text>')
    for i in range(41):
        val = -100 + i * 5
        bits.append(f'<rect x="{left+38+i*10}" y="{legend_y-9}" width="10" height="11" fill="{color_for_change(val)}"/>')
    bits.append(f'<text class="lab" x="{left+462}" y="{legend_y+4}">증가</text>')
    bits.append(f'<text class="sub" x="{left+525}" y="{legend_y+4}">색상 범위 ±100% (범위 밖은 잘라 표시)</text>')
    bits.append('</svg>')
    path.write_text("\n".join(bits), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with INPUT.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            counts[(row["month"], row["district"], row["source_type"], row["event_type"])] = int(row["event_count"])

    annual_rows = []
    annual_change_maps = {(source, event): {} for source in SOURCES for event in EVENTS}
    annual_index = defaultdict(dict)
    for year in range(2022, 2026):
        for gu in GUS:
            for source in SOURCES:
                opening = sum(counts[(f"{year}-{month:02d}", gu, source, "개업")] for month in range(1, 13))
                closure = sum(counts[(f"{year}-{month:02d}", gu, source, "폐업")] for month in range(1, 13))
                annual_index[(gu, source, "개업")][year] = opening
                annual_index[(gu, source, "폐업")][year] = closure
    for year in range(2023, 2026):
        for gu in GUS:
            for source in SOURCES:
                prev_open = annual_index[(gu, source, "개업")][year - 1]
                prev_close = annual_index[(gu, source, "폐업")][year - 1]
                opening = annual_index[(gu, source, "개업")][year]
                closure = annual_index[(gu, source, "폐업")][year]
                total = opening + closure
                balance = (opening - closure) / total if total else None
                annual_rows.append({
                    "year": year, "previous_year": year - 1, "district": gu, "source_type": source,
                    "opening_count": opening, "opening_change_count_yoy": opening - prev_open,
                    "opening_change_pct_yoy": pct_change(opening, prev_open),
                    "closure_count": closure, "closure_change_count_yoy": closure - prev_close,
                    "closure_change_pct_yoy": pct_change(closure, prev_close),
                    "recorded_net_change": opening - closure,
                    "opening_closure_balance": round(balance, 6) if balance is not None else "",
                })
                for event, delta, pct in (("개업", opening - prev_open, pct_change(opening, prev_open)),
                                          ("폐업", closure - prev_close, pct_change(closure, prev_close))):
                    annual_change_maps[(source, event)][(gu, str(year))] = pct
    annual_fields = ["year", "previous_year", "district", "source_type", "opening_count", "opening_change_count_yoy", "opening_change_pct_yoy", "closure_count", "closure_change_count_yoy", "closure_change_pct_yoy", "recorded_net_change", "opening_closure_balance"]
    write_csv(OUT / "지역구별_연도별_개폐업_2023-2025.csv", annual_fields, annual_rows)

    monthly_rows = []
    monthly_yoy_maps = {(source, event): {} for source in SOURCES for event in EVENTS}
    monthly_profile = []
    for gu in GUS:
        for source in SOURCES:
            for month in FOCUS_MONTHS:
                prev_year_month = f"{int(month[:4])-1:04d}-{month[-2:]}"
                opening = counts[(month, gu, source, "개업")]
                closure = counts[(month, gu, source, "폐업")]
                prev_open = counts[(prev_year_month, gu, source, "개업")]
                prev_close = counts[(prev_year_month, gu, source, "폐업")]
                total = opening + closure
                bal = (opening - closure) / total if total else None
                row = {
                    "month": month, "previous_year_same_month": prev_year_month, "district": gu, "source_type": source,
                    "opening_count": opening, "opening_change_count_yoy": opening - prev_open,
                    "opening_change_pct_yoy": pct_change(opening, prev_open),
                    "closure_count": closure, "closure_change_count_yoy": closure - prev_close,
                    "closure_change_pct_yoy": pct_change(closure, prev_close),
                    "recorded_net_change": opening - closure,
                    "opening_closure_balance": round(bal, 6) if bal is not None else "",
                }
                monthly_rows.append(row)
                year = month[:4]
                for event, pct in (("개업", row["opening_change_pct_yoy"]), ("폐업", row["closure_change_pct_yoy"])):
                    monthly_yoy_maps[(source, event)][(gu, month)] = float(pct) if pct != "" else None

            for m in range(1, 13):
                same_months = [f"{year}-{m:02d}" for year in ("2023", "2024", "2025")]
                ops = [counts[(sm, gu, source, "개업")] for sm in same_months]
                cls = [counts[(sm, gu, source, "폐업")] for sm in same_months]
                monthly_profile.append({
                    "calendar_month": m, "district": gu, "source_type": source,
                    "opening_mean_monthly_2023_2025": round(sum(ops) / 3, 3),
                    "opening_2023": ops[0], "opening_2024": ops[1], "opening_2025": ops[2],
                    "closure_mean_monthly_2023_2025": round(sum(cls) / 3, 3),
                    "closure_2023": cls[0], "closure_2024": cls[1], "closure_2025": cls[2],
                    "net_mean_monthly_2023_2025": round(sum(o - c for o, c in zip(ops, cls)) / 3, 3),
                })
    monthly_fields = ["month", "previous_year_same_month", "district", "source_type", "opening_count", "opening_change_count_yoy", "opening_change_pct_yoy", "closure_count", "closure_change_count_yoy", "closure_change_pct_yoy", "recorded_net_change", "opening_closure_balance"]
    write_csv(OUT / "지역구별_월별_개폐업_2023-2025.csv", monthly_fields, monthly_rows)
    write_csv(OUT / "지역구별_월별평균_2023-2025.csv", list(monthly_profile[0].keys()), monthly_profile)

    annual_columns = ["2023", "2024", "2025"]
    monthly_columns = FOCUS_MONTHS
    for source in SOURCES:
        for event in EVENTS:
            source_slug = "일반음식점" if source == "일반음식점" else "휴게음식점"
            heatmap_svg(OUT / f"그래프_연도별_지역변화_{source_slug}_{event}.svg",
                        f"{source} 지역별 연간 {event} 변화", "전년 대비 변화율; 2023년은 2022년 대비, 2024·2025년은 직전 연도 대비", annual_change_maps[(source, event)], annual_columns, "annual")
            heatmap_svg(OUT / f"그래프_월별_지역변화_{source_slug}_{event}.svg",
                        f"{source} 지역별 월간 {event} 변화", "전년 동월 대비 변화율; 2023년은 2022년 동월 대비 · 색상 범위 ±100%", monthly_yoy_maps[(source, event)], monthly_columns, "monthly")

    # Compact summary of YoY regional changes and season-by-calendar-month profiles.
    report_path = OUT / "지역별_개폐업_분석.md"
    with report_path.open("w", encoding="utf-8") as report:
        report.write("# 지역별 개업·폐업 변화 (2023–2025)\n\n")
        report.write("## 집계 기준\n\n")
        report.write("이전 초안의 ‘2018–2019 월평균 대 2024–2025 월평균’은 24개월씩 합쳐 하나의 평균과 변화율로 만든 요약값이었습니다. 그 값만으로는 연도별 변화나 계절별 월 변화를 볼 수 없으므로 이번 주 분석에서는 제외하고, 2023–2025 자치구별 연간·월간 흐름으로 다시 정리했습니다.\n\n")
        report.write("- 공간 단위는 임시 자치구입니다. 현 스냅샷 주소에서 구명을 추출했으며 행정동·상권 단위가 아닙니다.\n")
        report.write("- 연간 변화율은 2023년은 2022년 대비, 2024년은 2023년 대비, 2025년은 2024년 대비 계산했습니다. 2023년 전년 대비 계산에 필요한 2022년 값만 비교 기준으로 사용했습니다.\n")
        report.write("- 월간 변화율은 각 월을 전년 동월과 비교했습니다. 따라서 2023년은 2022년 동월 대비, 2024년은 2023년 동월 대비, 2025년은 2024년 동월 대비입니다.\n")
        report.write("- 변화율과 함께 절대 건수 및 변화 건수도 CSV에 담았습니다. 분모가 0인 전년 값은 변화율을 비워 두었습니다. 낮은 기준 건수에서는 백분율 변동이 크게 보일 수 있습니다.\n")
        report.write("- 개업−폐업은 기록 사건의 차이이며 실제 점포 재고 증감으로 간주하지 않습니다. 폐업일 미상과 행정 갱신 지연, 과거 사건 당시 주소와 현 주소의 차이가 결과에 영향을 줄 수 있습니다.\n\n")
        report.write("## 연도별 지역 변화\n\n")
        for year in (2023, 2024, 2025):
            report.write(f"### {year}년: 전년 대비 개업·폐업 변화율 상위 자치구\n\n")
            report.write("| 업종 | 사건 | 자치구 | 전년 건수 | 해당 연도 건수 | 변화 건수 | 변화율 |\n|---|---|---|---:|---:|---:|---:|\n")
            for source in SOURCES:
                for event in EVENTS:
                    col = "opening_change_pct_yoy" if event == "개업" else "closure_change_pct_yoy"
                    count_col = "opening_count" if event == "개업" else "closure_count"
                    delta_col = "opening_change_count_yoy" if event == "개업" else "closure_change_count_yoy"
                    eligible = [r for r in annual_rows if r["year"] == year and r["source_type"] == source and r[col] != ""]
                    eligible.sort(key=lambda r: float(r[col]), reverse=True)
                    for r in eligible[:3]:
                        report.write(f"| {source} | {event} | {r['district']} | {int(r[count_col])-int(r[delta_col]):,} | {int(r[count_col]):,} | {int(r[delta_col]):+,} | {float(r[col]):+.1f}% |\n")
            report.write("\n")
        report.write("## 지역별 증가·감소 분포\n\n")
        report.write("각 값은 전년 대비 개업 또는 폐업 건수가 증가한 자치구 수입니다.\n\n")
        report.write("| 연도 | 업종 | 개업 증가 구 | 개업 감소 구 | 폐업 증가 구 | 폐업 감소 구 |\n|---:|---|---:|---:|---:|---:|\n")
        for year in (2023, 2024, 2025):
            for source in SOURCES:
                rows = [r for r in annual_rows if r["year"] == year and r["source_type"] == source]
                op_up = sum(float(r["opening_change_pct_yoy"]) > 0 for r in rows if r["opening_change_pct_yoy"] != "")
                op_down = sum(float(r["opening_change_pct_yoy"]) < 0 for r in rows if r["opening_change_pct_yoy"] != "")
                cl_up = sum(float(r["closure_change_pct_yoy"]) > 0 for r in rows if r["closure_change_pct_yoy"] != "")
                cl_down = sum(float(r["closure_change_pct_yoy"]) < 0 for r in rows if r["closure_change_pct_yoy"] != "")
                report.write(f"| {year} | {source} | {op_up}/25 | {op_down}/25 | {cl_up}/25 | {cl_down}/25 |\n")
        report.write("\n2025년 일반음식점 개업은 전년보다 늘어난 구가 25개 중 1곳이고, 24곳은 감소했습니다. 휴게음식점은 14곳에서 개업이 늘고 10곳에서 줄었습니다. 자치구마다 변동 방향이 다르므로 서울 합계만으로는 이 차이를 볼 수 없습니다.\n\n")
        report.write("## 월별 변화 해석\n\n")
        report.write("월별 집계 파일에는 자치구·업종별 36개월의 실제 개업·폐업 건수와 전년 동월 대비 증감 건수·증감률을 모두 보존했습니다. 월평균 요약 파일은 같은 달(예: 1월)을 2023·2024·2025년에 걸쳐 평균해 계절 위치를 비교합니다. 이 평균은 월별 원자료를 대체하지 않습니다.\n\n")
        report.write("예를 들어 2025년 8월 노원구 일반음식점 개업은 전년 동월보다 43건 늘었고(+102.4%), 9월 성북구 폐업은 217건 늘었습니다(+461.7%). 월간 변화율은 기준 건수가 작으면 크게 흔들릴 수 있고 행정상 일괄 등록 가능성도 있으므로, 변화 건수·연간 흐름과 함께 확인해야 합니다.\n\n")
        report.write("월간 변화 그래프의 각 칸은 전년 동월 대비 변화율이며 색은 ±100%에서 제한됩니다. 정확한 변화율과 건수는 CSV를 확인해야 합니다.\n\n")
        report.write("## 해석 범위\n\n")
        report.write("이번 결과는 지역별 음식점 개폐업 동학을 찾는 EDA입니다. 개업 증가나 개폐업 균형만으로 핫플레이스 여부를 판정하지 않습니다. 최종 Emerging Hotspot은 생활인구·외부 유입 등 다른 오프라인 활동 지표와 지속적 성장 조건을 함께 정한 뒤 구성해야 합니다. 또한 자치구 단위는 넓으므로 다른 자료와 공통으로 연결 가능한 행정동 또는 상권 단위를 후속으로 검토해야 합니다.\n\n")
        report.write("## 저장 결과\n\n")
        report.write("- `지역구별_연도별_개폐업_2023-2025.csv`: 자치구×업종×연도 사건 수, 전년 대비 건수·변화율, 기록 순증감·균형.\n")
        report.write("- `지역구별_월별_개폐업_2023-2025.csv`: 자치구×업종×월(36개월) 사건 수, 전년 동월 대비 건수·변화율.\n")
        report.write("- `지역구별_월별평균_2023-2025.csv`: 자치구·업종별 달력 월(1–12월) 평균과 각 연도 건수.\n")
        report.write("- `그래프_연도별_지역변화_*.svg`: 업종·사건별 자치구 연간 전년 대비 변화율 히트맵 4개.\n")
        report.write("- `그래프_월별_지역변화_*.svg`: 업종·사건별 자치구 월간 전년 동월 대비 변화율 히트맵 4개.\n")
    print(f"완료: {report_path}")
    print(f"연도별 행: {len(annual_rows):,}; 월별 행: {len(monthly_rows):,}; 월평균 행: {len(monthly_profile):,}")
    print(f"SVG 그래프: {len(list(OUT.glob('그래프_*.svg')))}개")


if __name__ == "__main__":
    main()
