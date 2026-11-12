#!/usr/bin/env python3
"""Build stages 2-6 EDA tables and standalone SVG charts using the standard library."""

import csv
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "전처리_결과"
OUT = ROOT / "EDA_서울전체흐름"
EVENTS = DATA_DIR / "음식점_개폐업사건.csv"
CLEAN = DATA_DIR / "음식점_정리본.csv"
CORE_START = date(2018, 1, 1)
CORE_END = date(2025, 12, 31)
SOURCE_TYPES = ("일반음식점", "휴게음식점")
EVENT_TYPES = ("개업", "폐업")
SERIES = [(s, e) for s in SOURCE_TYPES for e in EVENT_TYPES]
COLORS = {
    ("일반음식점", "개업"): "#2563eb",
    ("일반음식점", "폐업"): "#dc2626",
    ("휴게음식점", "개업"): "#059669",
    ("휴게음식점", "폐업"): "#d97706",
}
SEOUL_GUS = (
    "종로구 중구 용산구 성동구 광진구 동대문구 중랑구 성북구 강북구 도봉구 노원구"
    " 은평구 서대문구 마포구 양천구 강서구 구로구 금천구 영등포구 동작구 관악구"
    " 서초구 강남구 송파구 강동구"
).split()
GU_RE = re.compile(r"(" + "|".join(map(re.escape, SEOUL_GUS)) + r")")


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def month_range(start, end):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m == 13:
            y += 1
            m = 1


def median_mad(values):
    med = statistics.median(values)
    mad = statistics.median(abs(x - med) for x in values)
    return med, mad


def robust_spikes(values, labels, dimension, source, event):
    med, mad = median_mad(values)
    threshold = med + 3 * 1.4826 * mad if mad else med + 3 * math.sqrt(max(med, 1))
    result = []
    for label, value in zip(labels, values):
        if value > threshold:
            score = (value - med) / (1.4826 * mad) if mad else ""
            result.append({
                "dimension": dimension,
                "period": label,
                "source_type": source,
                "event_type": event,
                "event_count": value,
                "median": round(med, 3),
                "robust_threshold": round(threshold, 3),
                "robust_score": round(score, 3) if score != "" else "",
                "flag_rule": "median + 3*1.4826*MAD (MAD=0이면 median + 3*sqrt(median)",
            })
    return result


def svg_chart(path, title, subtitle, panels, labels, width=1100, panel_height=260):
    """Render line-chart panels as dependency-free SVG."""
    margin_left, margin_right = 78, 28
    top, bottom = 78, 48
    height = top + len(panels) * panel_height + bottom
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,"Noto Sans KR",sans-serif;fill:#172033}.grid{stroke:#e5e7eb;stroke-width:1}.axis{stroke:#64748b;stroke-width:1}.legend{font-size:12px}.small{font-size:11px}.title{font-size:21px;font-weight:700}.sub{font-size:12px;fill:#475569}</style>',
        f'<text class="title" x="{margin_left}" y="30">{escape(title)}</text>',
        f'<text class="sub" x="{margin_left}" y="52">{escape(subtitle)}</text>',
    ]
    plot_w = width - margin_left - margin_right
    for panel_i, panel in enumerate(panels):
        y0 = top + panel_i * panel_height
        plot_top, plot_bottom = y0 + 18, y0 + panel_height - 38
        plot_h = plot_bottom - plot_top
        all_values = [v for _, vals, _ in panel["series"] for v in vals if v is not None]
        lo = panel.get("ymin", min([0] + all_values))
        hi = panel.get("ymax", max([1] + all_values))
        if panel.get("symmetric"):
            extent = max(abs(lo), abs(hi), 1)
            lo, hi = -extent, extent
        elif panel.get("ymin") is None:
            lo = min(0, lo)
        if hi <= lo:
            hi = lo + 1
        pieces.append(f'<text x="{margin_left}" y="{y0 + 10}" font-size="13" font-weight="700">{escape(panel["name"])}</text>')
        for tick in range(5):
            v = lo + (hi - lo) * tick / 4
            y = plot_bottom - (v - lo) / (hi - lo) * plot_h
            pieces.append(f'<line class="grid" x1="{margin_left}" y1="{y:.1f}" x2="{width-margin_right}" y2="{y:.1f}"/>')
            pieces.append(f'<text class="small" text-anchor="end" x="{margin_left-9}" y="{y+4:.1f}">{v:,.0f}</text>')
        pieces.append(f'<line class="axis" x1="{margin_left}" y1="{plot_bottom}" x2="{width-margin_right}" y2="{plot_bottom}"/>')
        if panel.get("zero_line") and lo < 0 < hi:
            y = plot_bottom - (0 - lo) / (hi - lo) * plot_h
            pieces.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width-margin_right}" y2="{y:.1f}" stroke="#475569" stroke-width="1.3" stroke-dasharray="5 4"/>')
        n = max((len(vals) for _, vals, _ in panel["series"]), default=0)
        for name, vals, color in panel["series"]:
            coords = []
            for i, value in enumerate(vals):
                if value is None:
                    continue
                x = margin_left + (i / max(n - 1, 1)) * plot_w
                y = plot_bottom - (value - lo) / (hi - lo) * plot_h
                coords.append(f"{x:.1f},{y:.1f}")
            if coords:
                pieces.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.8" points="{" ".join(coords)}"/>')
        tick_indices = panel.get("tick_indices", [0, n - 1])
        for idx in tick_indices:
            if 0 <= idx < len(labels):
                x = margin_left + (idx / max(n - 1, 1)) * plot_w
                pieces.append(f'<text class="small" text-anchor="middle" x="{x:.1f}" y="{plot_bottom+18}">{escape(labels[idx])}</text>')
        legend_y = plot_bottom + 35
        legend_x = margin_left
        for name, _, color in panel["series"]:
            pieces.append(f'<line x1="{legend_x}" y1="{legend_y-4}" x2="{legend_x+20}" y2="{legend_y-4}" stroke="{color}" stroke-width="2"/>')
            pieces.append(f'<text class="legend" x="{legend_x+25}" y="{legend_y}">{escape(name)}</text>')
            legend_x += 25 + len(name) * 7 + 38
    pieces.append("</svg>")
    path.write_text("\n".join(pieces), encoding="utf-8")


def rolling_mean(values, window):
    out, running = [], 0
    for i, value in enumerate(values):
        running += value
        if i >= window:
            running -= values[i - window]
        out.append(running / window if i >= window - 1 else None)
    return out


def decomposition(values):
    """Classical additive decomposition: centered 12-month moving average."""
    n = len(values)
    ma = [None] * (n - 11)
    for i in range(n - 11):
        ma[i] = sum(values[i:i + 12]) / 12
    trend = [None] * n
    for t in range(6, n - 6):
        trend[t] = (ma[t - 6] + ma[t - 5]) / 2
    month_residuals = defaultdict(list)
    for i, value in enumerate(values):
        if trend[i] is not None:
            month_residuals[i % 12].append(value - trend[i])
    raw_season = [statistics.mean(month_residuals[m]) if month_residuals[m] else 0 for m in range(12)]
    adjust = sum(raw_season) / 12
    season = [x - adjust for x in raw_season]
    seasonal = [season[i % 12] for i in range(n)]
    remainder = [values[i] - trend[i] - seasonal[i] if trend[i] is not None else None for i in range(n)]
    return trend, seasonal, remainder, season


def district_from_row(row):
    for field in ("lot_address", "road_address"):
        value = row.get(field, "") or ""
        match = GU_RE.search(value)
        if match:
            return match.group(1)
    return "매핑불가"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    event_counts = Counter()
    event_dates = []
    seen = set()
    with EVENTS.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            event_day = date.fromisoformat(row["event_date"])
            if not CORE_START <= event_day:
                continue
            key = (row["source_type"], row["management_id"], row["event_type"], row["event_date"])
            if key in seen:
                raise ValueError(f"중복 사건 키: {key}")
            seen.add(key)
            event_counts[(event_day, row["source_type"], row["event_type"])] += 1
            event_dates.append(event_day)
    if not event_dates:
        raise ValueError("2018년 이후 사건이 없습니다.")
    last_event = max(event_dates)
    days = []
    current = CORE_START
    while current <= last_event:
        days.append(current)
        current += timedelta(days=1)

    daily_rows = []
    daily_values = {}
    for source, event in SERIES:
        vals = [event_counts[(d, source, event)] for d in days]
        daily_values[(source, event)] = vals
        daily_rows.extend({"date": d.isoformat(), "year": d.year, "month": f"{d:%Y-%m}", "weekday": d.strftime("%A"),
                           "period_status": "완전연도" if d <= CORE_END else "부분연도_2026",
                           "source_type": source, "event_type": event, "event_count": value}
                          for d, value in zip(days, vals))
    write_csv(OUT / "일별_사건수.csv", ["date", "year", "month", "weekday", "period_status", "source_type", "event_type", "event_count"], daily_rows)

    weekday_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_ko = ["월", "화", "수", "목", "금", "토", "일"]
    weekday_sums = Counter()
    weekday_n = Counter()
    weekday_rows = []
    for day in days:
        if day > CORE_END:
            continue
        for source, event in SERIES:
            weekday_sums[(day.weekday(), source, event)] += event_counts[(day, source, event)]
            weekday_n[(day.weekday(), source, event)] += 1
    for weekday in range(7):
        for source, event in SERIES:
            total = weekday_sums[(weekday, source, event)]
            n = weekday_n[(weekday, source, event)]
            weekday_rows.append({"weekday_number_monday0": weekday, "weekday": weekday_ko[weekday], "source_type": source,
                                 "event_type": event, "days_observed": n, "event_total": total,
                                 "daily_mean": round(total / n, 4) if n else 0})
    write_csv(OUT / "요일별_요약.csv", ["weekday_number_monday0", "weekday", "source_type", "event_type", "days_observed", "event_total", "daily_mean"], weekday_rows)

    week_counts = Counter()
    week_days = defaultdict(set)
    for day in days:
        monday = day - timedelta(days=day.weekday())
        week_label = monday.isoformat()
        week_days[week_label].add(day)
        for source, event in SERIES:
            week_counts[(week_label, source, event)] += event_counts[(day, source, event)]
    weeks = sorted(week_days)
    weekly_rows = []
    weekly_values = {}
    for source, event in SERIES:
        vals = [week_counts[(week, source, event)] for week in weeks]
        weekly_values[(source, event)] = vals
        for week, value in zip(weeks, vals):
            weekly_rows.append({"week_start_monday": week, "source_type": source, "event_type": event, "event_count": value,
                                "days_in_calendar": len(week_days[week]),
                                "period_status": "완전연도구간" if date.fromisoformat(week) <= CORE_END else "부분연도_2026"})
    write_csv(OUT / "주별_사건수.csv", ["week_start_monday", "source_type", "event_type", "event_count", "days_in_calendar", "period_status"], weekly_rows)

    spike_rows = []
    core_day_labels = [d.isoformat() for d in days if d <= CORE_END]
    core_day_n = len(core_day_labels)
    for source, event in SERIES:
        spike_rows.extend(robust_spikes(daily_values[(source, event)][:core_day_n], core_day_labels, "일", source, event))
        core_week_vals = [week_counts[(w, source, event)] for w in weeks if date.fromisoformat(w) <= CORE_END]
        core_week_labels = [w for w in weeks if date.fromisoformat(w) <= CORE_END]
        spike_rows.extend(robust_spikes(core_week_vals, core_week_labels, "주", source, event))
    write_csv(OUT / "급증_후보_일주.csv", ["dimension", "period", "source_type", "event_type", "event_count", "median", "robust_threshold", "robust_score", "flag_rule"], spike_rows)

    months = list(month_range(CORE_START, last_event))
    month_counts = Counter()
    for (day, source, event), n in event_counts.items():
        month_counts[(f"{day:%Y-%m}", source, event)] += n
    monthly_rows = []
    monthly_series = {}
    for source, event in SERIES:
        vals = [month_counts[(m, source, event)] for m in months]
        monthly_series[(source, event)] = vals
        smooth = rolling_mean(vals, 12)
        for i, (month, value) in enumerate(zip(months, vals)):
            monthly_rows.append({"month": month, "year": int(month[:4]), "month_of_year": int(month[-2:]),
                                 "period_status": "완전연도" if month <= "2025-12" else ("부분월_2026-10" if month == "2026-10" else "부분연도_2026"),
                                 "source_type": source, "event_type": event, "event_count": value,
                                 "rolling_mean_12m_trailing": round(smooth[i], 4) if smooth[i] is not None else ""})
    # Monthly net change (recorded openings minus recorded closures), paired by source type.
    monthly_lookup = {(r["month"], r["source_type"], r["event_type"]): r["event_count"] for r in monthly_rows}
    for month in months:
        for source in SOURCE_TYPES:
            for row in monthly_rows:
                if row["month"] == month and row["source_type"] == source:
                    row["recorded_net_change"] = monthly_lookup[(month, source, "개업")] - monthly_lookup[(month, source, "폐업")]
    write_csv(OUT / "월별_집계_이동평균.csv", ["month", "year", "month_of_year", "period_status", "source_type", "event_type", "event_count", "rolling_mean_12m_trailing", "recorded_net_change"], monthly_rows)

    annual = Counter()
    for (day, source, event), n in event_counts.items():
        annual[(day.year, source, event)] += n
    annual_rows = []
    for year in range(2018, last_event.year + 1):
        for source in SOURCE_TYPES:
            op = annual[(year, source, "개업")]
            cl = annual[(year, source, "폐업")]
            annual_rows.append({"year": year, "period_status": "완전연도" if year <= 2025 else "부분연도", "source_type": source,
                                "opening_count": op, "closure_count": cl, "recorded_net_change": op - cl})
    write_csv(OUT / "연도별_요약.csv", ["year", "period_status", "source_type", "opening_count", "closure_count", "recorded_net_change"], annual_rows)

    seasonal_rows, decomp_rows = [], []
    core_months = [m for m in months if "2018-01" <= m <= "2025-12"]
    seasonal_lookup = {}
    for source, event in SERIES:
        values = [month_counts[(m, source, event)] for m in core_months]
        trend, seasonal, resid, season_factors = decomposition(values)
        for mon in range(12):
            seasonal_lookup[(source, event, mon + 1)] = season_factors[mon]
            seasonal_rows.append({"month_of_year": mon + 1, "source_type": source, "event_type": event,
                                  "monthly_mean_2018_2025": round(statistics.mean(month_counts[(m, source, event)] for m in core_months if int(m[-2:]) == mon + 1), 3),
                                  "additive_seasonal_factor": round(season_factors[mon], 3)})
        for i, month in enumerate(core_months):
            decomp_rows.append({"month": month, "source_type": source, "event_type": event, "observed": values[i],
                                "trend_centered_12m": round(trend[i], 4) if trend[i] is not None else "",
                                "seasonal_additive": round(seasonal[i], 4), "remainder": round(resid[i], 4) if resid[i] is not None else ""})
    write_csv(OUT / "월별_계절패턴.csv", ["month_of_year", "source_type", "event_type", "monthly_mean_2018_2025", "additive_seasonal_factor"], seasonal_rows)
    write_csv(OUT / "월별_추세_계절_잔차분해.csv", ["month", "source_type", "event_type", "observed", "trend_centered_12m", "seasonal_additive", "remainder"], decomp_rows)

    # Use the current snapshot address fields to map each record to a Seoul district.
    district_by_id = {}
    id_counts = Counter()
    mapped_ids = Counter()
    with CLEAN.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            source, management_id = row["source_type"], row["management_id"]
            if not management_id:
                continue
            id_key = (source, management_id)
            if id_key not in district_by_id or district_by_id[id_key] == "매핑불가":
                district_by_id[id_key] = district_from_row(row)
            id_counts[source] += 1
    # Count distinct IDs and mapped IDs from the address map.
    distinct_ids = Counter()
    for (source, _), gu in district_by_id.items():
        distinct_ids[source] += 1
        if gu != "매핑불가":
            mapped_ids[source] += 1
    mapping_rows = []
    for source in SOURCE_TYPES:
        mapping_rows.append({"source_type": source, "distinct_management_ids": distinct_ids[source], "district_mapped_ids": mapped_ids[source],
                             "unmapped_ids": distinct_ids[source] - mapped_ids[source],
                             "mapping_rate_pct": round(100 * mapped_ids[source] / distinct_ids[source], 2) if distinct_ids[source] else 0,
                             "mapping_basis": "정리본의 지번주소 우선, 도로명주소 보완; 구명 정규식"})
    write_csv(OUT / "지역구_매핑품질.csv", ["source_type", "distinct_management_ids", "district_mapped_ids", "unmapped_ids", "mapping_rate_pct", "mapping_basis"], mapping_rows)

    region_counts = Counter()
    unmapped_event_counts = Counter()
    with EVENTS.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            event_day = date.fromisoformat(row["event_date"])
            if not CORE_START <= event_day <= CORE_END:
                continue
            source, event = row["source_type"], row["event_type"]
            gu = district_by_id.get((source, row["management_id"]), "매핑불가")
            if gu == "매핑불가":
                unmapped_event_counts[(source, event)] += 1
            else:
                region_counts[(f"{event_day:%Y-%m}", gu, source, event)] += 1
    region_month_rows = []
    for month in core_months:
        for gu in SEOUL_GUS:
            for source, event in SERIES:
                region_month_rows.append({"month": month, "district": gu, "source_type": source, "event_type": event,
                                          "event_count": region_counts[(month, gu, source, event)]})
    write_csv(OUT / "지역구별_월별_집계.csv", ["month", "district", "source_type", "event_type", "event_count"], region_month_rows)
    region_totals = Counter()
    for (month, gu, source, event), count in region_counts.items():
        region_totals[(gu, source, event)] += count
    region_summary_rows = []
    for gu in SEOUL_GUS:
        for source, event in SERIES:
            region_summary_rows.append({"district": gu, "source_type": source, "event_type": event,
                                        "event_total_2018_2025": region_totals[(gu, source, event)]})
    write_csv(OUT / "지역구별_합계_2018_2025.csv", ["district", "source_type", "event_type", "event_total_2018_2025"], region_summary_rows)

    # Charts
    core_labels_daily = [d for d in days if d <= CORE_END]
    daily_indices = list(range(0, len(core_labels_daily), 365))
    if daily_indices[-1] != len(core_labels_daily) - 1:
        daily_indices.append(len(core_labels_daily) - 1)
    daily_tick_labels = [""] * len(core_labels_daily)
    for i in daily_indices:
        daily_tick_labels[i] = core_labels_daily[i].strftime("%Y-%m")
    daily_panels = [{"name": "기록된 일별 개업·폐업", "series": [(f"{s} {e}", daily_values[(s, e)][:core_day_n], COLORS[(s, e)]) for s, e in SERIES], "tick_indices": daily_indices}]
    svg_chart(OUT / "그래프_일별_사건수.svg", "일별 개업·폐업 사건 수", "2018–2025 · 일자 기준, 0은 해당 날짜에 기록된 사건이 없음을 뜻함", daily_panels, daily_tick_labels)

    core_week_indices = [i for i, w in enumerate(weeks) if date.fromisoformat(w) <= CORE_END]
    core_weeks = [weeks[i] for i in core_week_indices]
    week_tick_idx = list(range(0, len(core_weeks), 52))
    if week_tick_idx[-1] != len(core_weeks) - 1:
        week_tick_idx.append(len(core_weeks) - 1)
    week_labels = [""] * len(core_weeks)
    for i in week_tick_idx:
        week_labels[i] = date.fromisoformat(core_weeks[i]).strftime("%Y-%m")
    weekly_panels = [{"name": "기록된 주별 합계 (주 시작: 월요일)", "series": [(f"{s} {e}", weekly_values[(s, e)][:len(core_weeks)], COLORS[(s, e)]) for s, e in SERIES], "tick_indices": week_tick_idx}]
    svg_chart(OUT / "그래프_주별_사건수.svg", "주별 개업·폐업 사건 수", "2018–2025 · ISO 주 시작일 기준; 연도 경계 주는 양쪽 연도 날짜를 포함할 수 있음", weekly_panels, week_labels)

    weekday_panels = [{"name": "요일별 평균 일 사건 수 (2018–2025)",
                       "series": [(f"{s} {e}", [next(r["daily_mean"] for r in weekday_rows if r["weekday_number_monday0"] == i and r["source_type"] == s and r["event_type"] == e) for i in range(7)], COLORS[(s, e)]) for s, e in SERIES],
                       "tick_indices": list(range(7))}]
    svg_chart(OUT / "그래프_요일별_평균.svg", "요일별 사건 수 비교", "일별 평균; 법정 공휴일 및 행정 처리일 효과는 별도 통제하지 않음", weekday_panels, weekday_ko, width=900, panel_height=260)

    core_month_labels = [m for m in months if m <= "2025-12"]
    all_month_labels = months
    for source in SOURCE_TYPES:
        count_series = []
        for event in EVENT_TYPES:
            vals = monthly_series[(source, event)]
            smooth = rolling_mean(vals, 12)
            count_series.append((f"{event} 월계", vals, COLORS[(source, event)]))
            count_series.append((f"{event} 12개월 평균", smooth, "#93c5fd" if event == "개업" else "#fca5a5"))
        net_vals = [monthly_lookup[(m, source, "개업")] - monthly_lookup[(m, source, "폐업")] for m in all_month_labels]
        net_panel = {"name": "월별 기록 순증감 = 개업 − 폐업", "series": [("순증감", net_vals, "#7c3aed")], "symmetric": True, "zero_line": True}
        tick_idx = [i for i, m in enumerate(all_month_labels) if m.endswith("-01") or m == all_month_labels[-1]]
        labels = [m for i, m in enumerate(all_month_labels) if i in tick_idx]
        # svg_chart expects the shared labels indexed directly, so use a blank-label vector by month.
        chart_labels = all_month_labels
        panels = [{"name": "월별 개업·폐업 건수와 후행 12개월 평균", "series": count_series, "tick_indices": tick_idx}, net_panel]
        svg_chart(OUT / f"그래프_월별_추이_{source}.svg", f"{source} 월별 개업·폐업 추이", "2018–2025 완전연도 + 2026 부분연도; 12개월 평균은 후행 평균", panels, chart_labels, panel_height=270)

    season_panels = [{"name": "월별 평균 사건 수 (2018–2025)",
                      "series": [(f"{s} {e}", [next(r["monthly_mean_2018_2025"] for r in seasonal_rows if r["month_of_year"] == i and r["source_type"] == s and r["event_type"] == e) for i in range(1, 13)], COLORS[(s, e)]) for s, e in SERIES],
                      "tick_indices": list(range(12))}]
    svg_chart(OUT / "그래프_월별_계절패턴.svg", "월별 계절 패턴 탐색", "8개 연도의 월별 평균; 행정 기록의 시점 패턴이며 확정적 계절성 추론은 아님", season_panels, [f"{i}월" for i in range(1, 13)])

    annual_labels = [str(y) for y in range(2018, min(last_event.year, 2025) + 1)]
    annual_panels = [{"name": "연도별 개업·폐업 사건 수", "series": [(f"{s} {e}", [annual[(int(y), s, e)] for y in annual_labels], COLORS[(s, e)]) for s, e in SERIES], "tick_indices": list(range(len(annual_labels)))}]
    svg_chart(OUT / "그래프_연도별_개폐업.svg", "연도별 개업·폐업 사건 수", "2018–2025 완전연도만 비교; 2026 부분연도는 표에 별도 수록", annual_panels, annual_labels)

    for source, event in SERIES:
        vals = [month_counts[(m, source, event)] for m in core_months]
        trend, seasonal, resid, _ = decomposition(vals)
        components = [("관측값", vals, COLORS[(source, event)]), ("추세 (중심 12개월 이동평균)", trend, "#1d4ed8"), ("계절 성분 (가산)", seasonal, "#059669"), ("잔차", resid, "#7c3aed")]
        panels = [{"name": name, "series": [(name, data, color)], "symmetric": name in ("계절 성분 (가산)", "잔차"), "zero_line": name in ("계절 성분 (가산)", "잔차"), "tick_indices": [0, 12, 24, 36, 48, 60, 72, 84, 95]} for name, data, color in components]
        svg_chart(OUT / f"그래프_분해_{source}_{event}.svg", f"{source} {event} 월별 가산 분해", "2018–2025 · 중심 12개월 이동평균 추세; 양 끝 6개월은 중심 추세·잔차가 정의되지 않음", panels, core_months, width=1100, panel_height=205)

    district_totals_combined = Counter()
    for gu in SEOUL_GUS:
        district_totals_combined[gu] = sum(region_totals[(gu, source, event)] for source, event in SERIES)
    top_gus = [gu for gu, _ in district_totals_combined.most_common(5)]
    region_monthly_total = {}
    for gu in top_gus:
        region_monthly_total[gu] = [sum(region_counts[(m, gu, source, event)] for source, event in SERIES) for m in core_months]
    gu_palette = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]
    region_panels = [{"name": "상위 5개 자치구의 월별 사건 합계 (개업+폐업, 두 업종)",
                      "series": [(gu, region_monthly_total[gu], gu_palette[i]) for i, gu in enumerate(top_gus)],
                      "tick_indices": [0, 12, 24, 36, 48, 60, 72, 84, 95]}]
    svg_chart(OUT / "그래프_지역구_상위5_월별사건.svg", "지역구별 월간 사건 패턴", "주소에서 자치구를 추출한 기록 사건 수; 2018–2025; 상위 5개 구는 기간 합계 기준", region_panels, core_months)

    report_path = OUT / "EDA_2-6_결과보고서.md"
    total_days = (CORE_END - CORE_START).days + 1
    no_event_days = sum(all(event_counts[(d, s, e)] == 0 for s, e in SERIES) for d in (CORE_START + timedelta(days=i) for i in range(total_days)))
    peak_rows = sorted(spike_rows, key=lambda r: float(r["event_count"]), reverse=True)[:12]
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# 음식점 개업·폐업 시계열 EDA (2–6단계)\n\n")
        f.write("이 문서는 서울 전체 흐름을 요약한 배경 분석입니다. 핫플레이스 탐색에 필요한 지역별 월별 개폐업 비교는 `../EDA_지역별분석/지역별_개폐업_분석.md`와 그 폴더의 지표·그래프를 중심으로 확인합니다.\n\n")
        f.write("## 분석 범위와 해석 기준\n\n")
        f.write(f"- 주 비교 구간: {CORE_START}~{CORE_END} (완전한 달력 연도 8개). 2026년은 현재 스냅샷 기준 부분 연도로 표에 분리했습니다 (관측 사건일 최대: {last_event}).\n")
        f.write("- 사건 시계열은 일반음식점·휴게음식점의 인허가일(개업)과 폐업일(폐업)을 각각 집계했습니다. 폐업일이 없는 폐업 상태 행은 날짜가 없어 일자별 폐업 사건에 넣을 수 없습니다.\n")
        f.write("- 개업/폐업일은 행정 원장의 사건 날짜입니다. 실제 영업 개시·종료일, 일별 수요, 점포 재고를 그대로 나타내지는 않습니다. 과거 기록의 누락이나 갱신 지연 여부도 스냅샷만으로 검증할 수 없습니다.\n")
        f.write("- 날짜가 관측되지 않은 날의 0은 ‘이 파일에 해당 날짜 사건이 없다’는 뜻입니다. 원자료 수집 누락과 구분해야 합니다.\n\n")
        f.write("## 2단계: 일별 사건 시계열\n\n")
        f.write(f"- 일별 파일 범위: {CORE_START}~{last_event}; 2018–2025 핵심 분석 구간에는 {no_event_days:,}일(전체 {total_days:,}일)에 네 개 시계열 모두 기록 사건이 없습니다.\n")
        f.write("- 날짜 공백은 모든 날짜를 달력으로 생성해 0건으로 보완했습니다. 결측 날짜는 별도 상태로 취급하지 않았습니다.\n\n")
        f.write("## 3단계: 일·주별 변동과 급증 후보\n\n")
        f.write("- 요일별 평균은 2018–2025 날짜를 기준으로 계산했습니다. 법정 공휴일, 연휴, 행정기관의 일괄 입력·정정 효과는 통제하지 않았으므로 요일 차이를 운영 행태로 단정하지 않습니다.\n")
        weekday_highlights = []
        for source, event in SERIES:
            candidates = [r for r in weekday_rows if r["source_type"] == source and r["event_type"] == event]
            highest = max(candidates, key=lambda r: r["daily_mean"])
            lowest = min(candidates, key=lambda r: r["daily_mean"])
            weekday_highlights.append(f"{source} {event}: {highest['weekday']}요일 평균 {highest['daily_mean']:.2f}건, 최저 {lowest['weekday']}요일 {lowest['daily_mean']:.2f}건")
        f.write("- 요일별 최고·최저 평균: " + "; ".join(weekday_highlights) + ". 행정일자 편중일 수 있어 실제 개업·폐업 행동과 동일시하면 안 됩니다.\n")
        f.write("- 주간 합계는 월요일 시작으로 묶었고, 연도 경계 주는 두 연도 날짜를 포함할 수 있습니다. 급증 후보는 중앙값과 MAD 기반 규칙으로 표기한 탐색 대상이며 원인 판정이 아닙니다.\n\n")
        f.write("### 급증 후보 상위 12개 (일·주 합산 정렬)\n\n")
        f.write("| 단위 | 기간 | 업종 | 사건 | 건수 | 탐색 기준 |\n|---|---|---|---|---:|---|\n")
        for r in peak_rows:
            f.write(f"| {r['dimension']} | {r['period']} | {r['source_type']} | {r['event_type']} | {int(r['event_count']):,} | median+3×1.4826×MAD 계열 |\n")
        f.write("\n## 4단계: 월별 주 분석\n\n")
        f.write("- 월별 사건 수, 기록 순증감(개업−폐업), 후행 12개월 평균을 계산했습니다. 기록 순증감은 현재 영업 점포 수의 증감으로 해석할 수 없습니다.\n")
        f.write("- 월별 평균 및 가산 계절성 성분은 2018–2025년을 사용했고, 2026년 부분연도는 계절 패턴 계산에서 제외했습니다.\n")
        f.write("- 월별 평균은 두 업종 모두 4월 개업이 가장 많고 2월이 가장 적었습니다. 폐업은 두 업종 모두 12월이 가장 많고 2월이 가장 적었습니다. 이는 스냅샷에 기록된 날짜의 탐색 패턴입니다.\n\n")
        f.write("## 5단계: 연도별 흐름과 분해\n\n")
        f.write("- 연도별 표에는 개업, 폐업, 기록 순증감을 담았습니다. 2026년은 연간 합계 비교에서 제외하고 부분연도로 표시했습니다.\n")
        f.write(f"- 두 업종의 기록 순증감은 2024년부터 음수로 전환했고, 2025년에는 일반음식점 {annual[(2025, SOURCE_TYPES[0], '개업')]-annual[(2025, SOURCE_TYPES[0], '폐업')]:,}건, 휴게음식점 {annual[(2025, SOURCE_TYPES[1], '개업')]-annual[(2025, SOURCE_TYPES[1], '폐업')]:,}건입니다. 폐업일 미상 행과 스냅샷의 과거 누락 가능성 때문에 실제 점포 재고 증감으로 읽으면 안 됩니다.\n")
        f.write("- 분해는 12개월 중심이동평균 기반의 고전적 가산 분해입니다. 8년 자료로 계절 패턴을 탐색할 수 있지만 장기 순환(Cycle)의 확정적 검정에는 기간이 짧습니다. 시작과 끝 각 6개월은 추세·잔차가 정의되지 않습니다.\n\n")
        f.write("## 6단계: 업종·지역 비교\n\n")
        f.write("- 업종별 월별 개업·폐업 및 순증감은 월별 추이 그래프와 표에서 비교할 수 있습니다.\n")
        f.write("- 지역은 정리본의 현 주소에서 지번주소를 우선, 도로명주소를 보완해 서울 25개 자치구명을 정규식으로 추출했습니다. 행정동 단위 매핑이 아니며, 과거 사건 당시 주소와 현재 스냅샷 주소가 다를 가능성이 있어 지역 시계열은 탐색용입니다. 주소 품질·매핑률은 별도 표에 있습니다.\n")
        f.write("- 지역별 월 집계에는 2018–2025 사건만 포함했습니다. 매핑 불가 사건은 자치구별 합계에서 제외되며 매핑 품질 파일에서 확인할 수 있습니다.\n")
        top_regions = sorted(((gu, sum(region_totals[(gu, source, event)] for source, event in SERIES)) for gu in SEOUL_GUS), key=lambda item: -item[1])[:5]
        f.write("- 두 업종의 개업·폐업 합계 상위 자치구(2018–2025): " + ", ".join(f"{gu} {count:,}건" for gu, count in top_regions) + ". 이 값은 주소 매핑 가능한 사건 수의 단순 합이며 인구·점포 수로 표준화하지 않았습니다.\n\n")
        f.write("지역별 연간·월간 변화와 전년 대비 증감률은 `../EDA_지역별분석/지역별_개폐업_분석.md`에서 별도로 비교했습니다. 자치구 단위도 최종 핫플레이스 공간 단위로 확정된 것은 아닙니다.\n\n")
        f.write("## 저장된 결과\n\n")
        f.write("표 CSV: `일별_사건수.csv`, `주별_사건수.csv`, `요일별_요약.csv`, `급증_후보_일주.csv`, `월별_집계_이동평균.csv`, `연도별_요약.csv`, `월별_계절패턴.csv`, `월별_추세_계절_잔차분해.csv`, `지역구_매핑품질.csv`, `지역구별_월별_집계.csv`, `지역구별_합계_2018_2025.csv`.\n\n")
        f.write("그래프는 같은 폴더의 `그래프_*.svg` 파일입니다. SVG는 브라우저에서 열 수 있고 확대해도 선명합니다.\n\n")
        f.write("## 대표 연도별 값\n\n")
        f.write("| 연도 | 일반 개업 | 일반 폐업 | 일반 순증감 | 휴게 개업 | 휴게 폐업 | 휴게 순증감 |\n|---:|---:|---:|---:|---:|---:|---:|\n")
        for year in range(2018, 2026):
            vals = [annual[(year, s, e)] for s, e in ((SOURCE_TYPES[0], "개업"), (SOURCE_TYPES[0], "폐업"), (SOURCE_TYPES[1], "개업"), (SOURCE_TYPES[1], "폐업"))]
            f.write(f"| {year} | {vals[0]:,} | {vals[1]:,} | {vals[0]-vals[1]:,} | {vals[2]:,} | {vals[3]:,} | {vals[2]-vals[3]:,} |\n")
        f.write("\n## 지역 자치구 매핑률\n\n")
        f.write("| 업종 | 고유 관리번호 | 매핑 성공 | 매핑률 |\n|---|---:|---:|---:|\n")
        for r in mapping_rows:
            f.write(f"| {r['source_type']} | {r['distinct_management_ids']:,} | {r['district_mapped_ids']:,} | {r['mapping_rate_pct']:.2f}% |\n")

    print(f"완료: {OUT}")
    print(f"기간: {CORE_START}~{last_event}; 핵심 완전연도: 2018~2025")
    print(f"일별 행: {len(daily_rows):,}; 주별 행: {len(weekly_rows):,}; 월별 행: {len(monthly_rows):,}")
    print(f"지역 월별 행: {len(region_month_rows):,}; SVG 그래프: {len(list(OUT.glob('그래프_*.svg')))}개")
    print("자치구 매핑률: " + ", ".join(f"{r['source_type']} {r['mapping_rate_pct']:.2f}%" for r in mapping_rows))
    print(f"보고서: {report_path}")


if __name__ == "__main__":
    main()
