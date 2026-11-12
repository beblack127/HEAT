#!/usr/bin/env python3
"""Create subtype and industry-family EDA outputs and refresh the Seoul dashboard."""

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "전처리_결과"
OUT = ROOT / "EDA_지역별분석"
EVENTS = DATA_DIR / "음식점_개폐업사건.csv"
CLEAN = DATA_DIR / "음식점_정리본.csv"
DASHBOARD = OUT / "서울_구별_음식점_개폐업_대시보드.html"
GUS = "종로구 중구 용산구 성동구 광진구 동대문구 중랑구 성북구 강북구 도봉구 노원구 은평구 서대문구 마포구 양천구 강서구 구로구 금천구 영등포구 동작구 관악구 서초구 강남구 송파구 강동구".split()
SOURCES = ("일반음식점", "휴게음식점")
EVENT_TYPES = ("개업", "폐업")
MONTHS = [f"{year:04d}-{month:02d}" for year in range(2023, 2026) for month in range(1, 13)]
FAMILY_ORDER = ["한식·고기·해산물", "외식 전문점", "간편식·분식·치킨", "카페·차·디저트", "주류·야간", "소매·입지·서비스", "기타·미분류"]

SEOUL_GU_RE = re.compile(r"(" + "|".join(map(re.escape, GUS)) + r")")

# This is an EDA grouping layer. The raw business_type value is retained in every output.
FAMILY_BY_LABEL = {
    "한식": "한식·고기·해산물", "식육(숯불구이)": "한식·고기·해산물", "냉면집": "한식·고기·해산물",
    "복어취급": "한식·고기·해산물", "탕류(보신용)": "한식·고기·해산물", "횟집": "한식·고기·해산물",
    "경양식": "외식 전문점", "일식": "외식 전문점", "중국식": "외식 전문점",
    "외국음식전문점(인도,태국등)": "외식 전문점", "뷔페식": "외식 전문점",
    "패밀리레스트랑": "외식 전문점", "패밀리레스토랑": "외식 전문점",
    "분식": "간편식·분식·치킨", "김밥(도시락)": "간편식·분식·치킨", "통닭(치킨)": "간편식·분식·치킨",
    "패스트푸드": "간편식·분식·치킨", "일반조리판매": "간편식·분식·치킨", "푸드트럭": "간편식·분식·치킨",
    "까페": "카페·차·디저트", "커피숍": "카페·차·디저트", "다방": "카페·차·디저트",
    "전통찻집": "카페·차·디저트", "과자점": "카페·차·디저트", "아이스크림": "카페·차·디저트", "떡카페": "카페·차·디저트",
    "호프/통닭": "주류·야간", "정종/대포집/소주방": "주류·야간", "감성주점": "주류·야간", "라이브카페": "주류·야간",
    "편의점": "소매·입지·서비스", "백화점": "소매·입지·서비스", "철도역구내": "소매·입지·서비스",
    "고속도로": "소매·입지·서비스", "공항": "소매·입지·서비스", "관광호텔": "소매·입지·서비스",
    "극장": "소매·입지·서비스", "유원지": "소매·입지·서비스", "키즈카페": "소매·입지·서비스",
    "출장조리": "소매·입지·서비스", "이동조리": "소매·입지·서비스",
    "기타": "기타·미분류", "기타 휴게음식점": "기타·미분류", "업태 미상": "기타·미분류",
}


def family_for(label):
    return FAMILY_BY_LABEL.get(label or "업태 미상", "기타·미분류")


def parse_iso(value):
    value = (value or "").strip()
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def district_from_record(row):
    for field in ("lot_address", "road_address"):
        match = SEOUL_GU_RE.search(row.get(field, "") or "")
        if match:
            return match.group(1)
    return "매핑불가"


def add_year(day):
    try:
        return day.replace(year=day.year + 1)
    except ValueError:
        return day.replace(year=day.year + 1, day=28)


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pct(numerator, denominator):
    return round(numerator / denominator * 100, 3) if denominator else ""


def safe_num(value):
    return int(value) if value else 0


def read_event_rows():
    selected = []
    needed = set()
    with EVENTS.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            day = parse_iso(row.get("event_date"))
            if not day or f"{day:%Y-%m}" not in MONTHS:
                continue
            row["_date"] = day
            row["_key"] = (row["source_type"], row["source_row_number"])
            selected.append(row)
            needed.add(row["_key"])
    return selected, needed


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    event_rows, needed_keys = read_event_rows()
    registry_by_event_key = {}
    current_stock = Counter()
    seen_management_ids = set()
    category_catalog = defaultdict(set)
    mapping_catalog = {}
    data_cutoff = None

    # Keep only the event-linked registry rows in memory; count current inventory once per ID.
    with CLEAN.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            source = row["source_type"]
            label = (row.get("business_type") or "").strip() or "업태 미상"
            raw_label = (row.get("business_type_raw") or "").strip()
            key = (source, row.get("source_row_number", ""))
            if key in needed_keys:
                registry_by_event_key[key] = {
                    "district": district_from_record(row), "business_type": label,
                    "business_type_raw": raw_label, "permit_date": parse_iso(row.get("permit_date")),
                    "closure_date": parse_iso(row.get("closure_date")), "status": (row.get("status_normalized") or "").strip(),
                }
            category_catalog[source].add(label)
            mapping_catalog[(source, raw_label, label)] = family_for(label)
            updated = parse_iso(row.get("data_update_raw"))
            if updated and (data_cutoff is None or updated > data_cutoff):
                data_cutoff = updated
            management_id = row.get("management_id", "")
            id_key = (source, management_id)
            if management_id and id_key not in seen_management_ids:
                seen_management_ids.add(id_key)
                if row.get("status_normalized") == "영업":
                    district = district_from_record(row)
                    if district in GUS:
                        current_stock[(district, source, label)] += 1

    if data_cutoff is None:
        raise RuntimeError("정리본에서 스냅샷 기준일(data_update_raw)을 확인할 수 없습니다.")

    # key: month, district, source, raw type, event
    counts = Counter()
    unmapped = Counter()
    missing_registry = Counter()
    subtype_labels = {source: set() for source in SOURCES}
    survival = Counter()  # (district, source, type) -> survived, failed, unknown, eligible
    eligible_by_source = Counter()
    observed_by_source = Counter()
    for row in event_rows:
        source, event = row["source_type"], row["event_type"]
        observed_by_source[(source, event)] += 1
        rec = registry_by_event_key.get(row["_key"])
        if not rec:
            missing_registry[(source, event)] += 1
            continue
        district = rec["district"]
        if district not in GUS:
            unmapped[(source, event)] += 1
            continue
        label = rec["business_type"] or (row.get("business_type") or "").strip() or "업태 미상"
        subtype_labels[source].add(label)
        counts[(row["_date"].strftime("%Y-%m"), district, source, label, event)] += 1

        if event == "개업":
            eligible_by_source[source] += 1
            anniversary = add_year(row["_date"])
            if anniversary > data_cutoff:
                continue
            closed = rec["closure_date"]
            status = rec["status"]
            result = "unknown"
            if closed:
                result = "survived" if closed >= anniversary else "failed"
            elif status == "영업":
                result = "survived"
            survival[(district, source, label, result)] += 1
            survival[(district, source, label, "eligible")] += 1

    # Dense monthly records make zero-event months explicit and keep all months aligned.
    source_month_total = Counter()
    for (month, district, source, label, event), value in counts.items():
        source_month_total[(month, district, source, event)] += value

    monthly_subtype_rows = []
    monthly_family_rows = []
    annual_family_rows = []
    annual_family_counter = Counter()
    annual_subtype_counter = Counter()
    for district in GUS:
        for source in SOURCES:
            labels = sorted(subtype_labels[source], key=lambda x: (FAMILY_ORDER.index(family_for(x)), x))
            families = [family for family in FAMILY_ORDER if any(family_for(label) == family for label in labels)]
            for label in labels:
                family = family_for(label)
                for month in MONTHS:
                    opening = counts[(month, district, source, label, "개업")]
                    closure = counts[(month, district, source, label, "폐업")]
                    total_open = source_month_total[(month, district, source, "개업")]
                    total_close = source_month_total[(month, district, source, "폐업")]
                    monthly_subtype_rows.append({
                        "month": month, "district": district, "source_type": source, "industry_family": family,
                        "business_type": label, "opening_count": opening, "closure_count": closure,
                        "recorded_net_event_count": opening - closure, "turnover_count": opening + closure,
                        "opening_share_in_source_pct": pct(opening, total_open),
                        "closure_share_in_source_pct": pct(closure, total_close),
                    })
                    annual_subtype_counter[(int(month[:4]), district, source, label, "개업")] += opening
                    annual_subtype_counter[(int(month[:4]), district, source, label, "폐업")] += closure
                    annual_family_counter[(int(month[:4]), district, source, family, "개업")] += opening
                    annual_family_counter[(int(month[:4]), district, source, family, "폐업")] += closure
            for family in families:
                for month in MONTHS:
                    opening = sum(counts[(month, district, source, label, "개업")] for label in labels if family_for(label) == family)
                    closure = sum(counts[(month, district, source, label, "폐업")] for label in labels if family_for(label) == family)
                    total_open = source_month_total[(month, district, source, "개업")]
                    total_close = source_month_total[(month, district, source, "폐업")]
                    monthly_family_rows.append({
                        "month": month, "district": district, "source_type": source, "industry_family": family,
                        "opening_count": opening, "closure_count": closure, "recorded_net_event_count": opening - closure,
                        "turnover_count": opening + closure,
                        "opening_share_in_source_pct": pct(opening, total_open),
                        "closure_share_in_source_pct": pct(closure, total_close),
                    })

    for year in range(2023, 2026):
        for district in GUS:
            for source in SOURCES:
                families = [family for family in FAMILY_ORDER if any(family_for(label) == family for label in subtype_labels[source])]
                for family in families:
                    opening = annual_family_counter[(year, district, source, family, "개업")]
                    closure = annual_family_counter[(year, district, source, family, "폐업")]
                    total_open = sum(annual_family_counter[(year, district, source, fam, "개업")] for fam in families)
                    total_close = sum(annual_family_counter[(year, district, source, fam, "폐업")] for fam in families)
                    previous_open = annual_family_counter[(year - 1, district, source, family, "개업")] if year > 2023 else ""
                    previous_close = annual_family_counter[(year - 1, district, source, family, "폐업")] if year > 2023 else ""
                    annual_family_rows.append({
                        "year": year, "district": district, "source_type": source, "industry_family": family,
                        "opening_count": opening, "closure_count": closure,
                        "opening_change_count_yoy": opening - previous_open if previous_open != "" else "",
                        "opening_change_pct_yoy": pct(opening - previous_open, previous_open) if previous_open not in ("", 0) else "",
                        "closure_change_count_yoy": closure - previous_close if previous_close != "" else "",
                        "closure_change_pct_yoy": pct(closure - previous_close, previous_close) if previous_close not in ("", 0) else "",
                        "recorded_net_event_count": opening - closure, "turnover_count": opening + closure,
                        "opening_share_in_source_pct": pct(opening, total_open),
                        "closure_share_in_source_pct": pct(closure, total_close),
                    })

    write_csv(OUT / "지역구별_월별_세부업태_개폐업_2023-2025.csv", [
        "month", "district", "source_type", "industry_family", "business_type", "opening_count", "closure_count",
        "recorded_net_event_count", "turnover_count", "opening_share_in_source_pct", "closure_share_in_source_pct"], monthly_subtype_rows)
    write_csv(OUT / "지역구별_월별_업태군_개폐업_2023-2025.csv", [
        "month", "district", "source_type", "industry_family", "opening_count", "closure_count",
        "recorded_net_event_count", "turnover_count", "opening_share_in_source_pct", "closure_share_in_source_pct"], monthly_family_rows)
    write_csv(OUT / "지역구별_연도별_업태군_개폐업_2023-2025.csv", [
        "year", "district", "source_type", "industry_family", "opening_count", "closure_count",
        "opening_change_count_yoy", "opening_change_pct_yoy", "closure_change_count_yoy", "closure_change_pct_yoy",
        "recorded_net_event_count", "turnover_count", "opening_share_in_source_pct", "closure_share_in_source_pct"], annual_family_rows)

    # Keep raw labels and a separate, documented semantic family in the mapping output.
    mapping_rows = [{"source_type": src, "business_type_raw": raw or "", "business_type": label,
                     "industry_family": family, "mapping_note": "EDA 비교용 임시 묶음; 원자료 업태 의미를 대체하지 않음"}
                    for (src, raw, label), family in sorted(mapping_catalog.items())]
    write_csv(OUT / "업태분류_매핑표.csv", ["source_type", "business_type_raw", "business_type", "industry_family", "mapping_note"], mapping_rows)

    survival_rows = []
    for district in GUS:
        for source in SOURCES:
            for label in sorted(subtype_labels[source], key=lambda x: (FAMILY_ORDER.index(family_for(x)), x)):
                success = survival[(district, source, label, "survived")]
                failed = survival[(district, source, label, "failed")]
                unknown = survival[(district, source, label, "unknown")]
                eligible = survival[(district, source, label, "eligible")]
                valid = success + failed
                survival_rows.append({
                    "district": district, "source_type": source, "industry_family": family_for(label), "business_type": label,
                    "matured_12m_opening_count": eligible, "known_12m_outcome_count": valid,
                    "survived_at_least_12m_count": success, "closed_before_12m_count": failed,
                    "unknown_outcome_count": unknown, "survival_rate_12m_pct": pct(success, valid),
                    "followup_cutoff_date": data_cutoff.isoformat(),
                })
    write_csv(OUT / "지역구별_업태별_12개월_유지율.csv", [
        "district", "source_type", "industry_family", "business_type", "matured_12m_opening_count",
        "known_12m_outcome_count", "survived_at_least_12m_count", "closed_before_12m_count",
        "unknown_outcome_count", "survival_rate_12m_pct", "followup_cutoff_date"], survival_rows)

    stock_rows = []
    for (district, source, label), value in sorted(current_stock.items()):
        stock_rows.append({"district": district, "source_type": source, "industry_family": family_for(label),
                           "business_type": label, "active_unique_management_id_count": value,
                           "snapshot_cutoff_date": data_cutoff.isoformat()})
    write_csv(OUT / "자치구별_현재_영업재고_업태별.csv", [
        "district", "source_type", "industry_family", "business_type", "active_unique_management_id_count", "snapshot_cutoff_date"], stock_rows)

    # Build compact chart data: one 36-month series per district/source/category.
    dashboard_data = {"periods": MONTHS, "districts": {}, "survival": {}, "stock": {}, "families": {}, "types": {}}
    for source in SOURCES:
        labels = sorted(subtype_labels[source], key=lambda x: (FAMILY_ORDER.index(family_for(x)), x))
        families = [family for family in FAMILY_ORDER if any(family_for(label) == family for label in labels)]
        dashboard_data["families"][source] = families
        dashboard_data["types"][source] = labels
    for district in GUS:
        dashboard_data["districts"][district] = {}
        dashboard_data["survival"][district] = {}
        dashboard_data["stock"][district] = {}
        for source in SOURCES:
            labels = dashboard_data["types"][source]
            families = dashboard_data["families"][source]
            type_series = {}
            for label in labels:
                type_series[label] = [[counts[(month, district, source, label, "개업")],
                                       counts[(month, district, source, label, "폐업")]] for month in MONTHS]
            all_series = [[sum(type_series[label][i][0] for label in labels),
                           sum(type_series[label][i][1] for label in labels)] for i in range(len(MONTHS))]
            family_series = {family: [[sum(type_series[label][i][0] for label in labels if family_for(label) == family),
                                       sum(type_series[label][i][1] for label in labels if family_for(label) == family)]
                                      for i in range(len(MONTHS))] for family in families}
            dashboard_data["districts"][district][source] = {"all": all_series, "families": family_series, "types": type_series}

            type_survival = {}
            for label in labels:
                success = survival[(district, source, label, "survived")]
                failed = survival[(district, source, label, "failed")]
                unknown = survival[(district, source, label, "unknown")]
                type_survival[label] = [success, success + failed, survival[(district, source, label, "eligible")], unknown]
            all_survival = [sum(v[0] for v in type_survival.values()), sum(v[1] for v in type_survival.values()),
                            sum(v[2] for v in type_survival.values()), sum(v[3] for v in type_survival.values())]
            family_survival = {family: [sum(v[i] for label, v in type_survival.items() if family_for(label) == family)
                                        for i in range(4)] for family in families}
            dashboard_data["survival"][district][source] = {"all": all_survival, "families": family_survival, "types": type_survival}

            type_stock = {label: current_stock[(district, source, label)] for label in labels}
            dashboard_data["stock"][district][source] = {
                "all": sum(type_stock.values()),
                "families": {family: sum(n for label, n in type_stock.items() if family_for(label) == family) for family in families},
                "types": type_stock,
            }

    html_data = json.dumps(dashboard_data, ensure_ascii=False, separators=(",", ":"))
    html_source = DASHBOARD.read_text(encoding="utf-8")
    start_marker = "const categoryData=" if "const categoryData=" in html_source else "const chartData="
    end_marker = ";\nconst runData="
    start = html_source.index(start_marker)
    end = html_source.index(end_marker, start)
    html_source = html_source[:start] + f"const categoryData={html_data};" + html_source[end + 1:]

    html_source = html_source.replace(
        '<p class="detail-caption">월별 행정 신고 건수 · 일반음식점 / 휴게음식점</p>',
        '<p class="detail-caption">월별 개업·폐업 건수 · 자치구 × 업태 · 2023.01–2025.12</p>')
    metrics_marker = '<div class="metrics" id="metrics"></div>'
    metrics_note = ('<p class="method-note">업태군은 비교를 위한 임시 묶음이며 원자료 업태명을 함께 선택할 수 있습니다. '
                    '과거 사건의 업태·자치구는 현재 스냅샷 기준입니다. 점선은 해당 월까지 포함한 후행 12개월 이동평균으로, 계절 변동을 완화해 추세를 보는 값이며 예측값이 아닙니다. '
                    '12개월 유지율은 ' + data_cutoff.isoformat() +
                    '까지 12개월 관찰되고 결과가 확인된 기록만 n에 포함합니다. 현재 영업 수와 유지율은 참고 지표이며 핫플레이스 판정값이 아닙니다.</p>')
    metrics_at = html_source.index(metrics_marker)
    note_end = metrics_at + len(metrics_marker)
    while html_source.startswith('<p class="method-note">', note_end):
        note_end = html_source.index('</p>', note_end) + len('</p>')
    html_source = html_source[:metrics_at] + metrics_marker + metrics_note + html_source[note_end:]
    html_source = html_source.replace(
        '<div class="plot-head"><h3>일반음식점</h3><div class="plot-total" id="generalTotal"></div></div>',
        '<div class="plot-head"><div class="plot-title"><h3 id="generalHeading">일반음식점</h3><label>업태 <select id="generalTypeSelect" class="type-select" aria-label="일반음식점 업태 선택"></select></label></div><div class="plot-total" id="generalTotal"></div></div>')
    html_source = html_source.replace(
        '<div class="plot-head"><h3>휴게음식점</h3><div class="plot-total" id="snackTotal"></div></div>',
        '<div class="plot-head"><div class="plot-title"><h3 id="snackHeading">휴게음식점</h3><label>업태 <select id="snackTypeSelect" class="type-select" aria-label="휴게음식점 업태 선택"></select></label></div><div class="plot-total" id="snackTotal"></div></div>')
    html_source = html_source.replace(
        '.plot-head{display:flex;justify-content:space-between;align-items:baseline;gap:10px;margin-bottom:5px}',
        '.plot-head{display:flex;justify-content:space-between;align-items:baseline;gap:10px;margin-bottom:5px}.plot-title{display:flex;align-items:center;gap:12px;min-width:0}.plot-title label{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:10px;white-space:nowrap}.type-select{min-width:155px;max-width:205px;padding:7px 27px 7px 9px;font-size:11px}.method-note{margin:0 0 13px;color:var(--muted);font-size:11px;line-height:1.6}.plot-total .subline{color:#718087}.plot-total .net-positive{color:#247f78}.plot-total .net-negative{color:#cb6c75}')
    html_source = html_source.replace(
        '.plot-total{font-size:10px;color:var(--muted);text-align:right;line-height:1.4}',
        '.plot-total{font-size:10px;color:var(--muted);text-align:right;line-height:1.5;max-width:50%}')
    series_rule = '.chart .series{fill:none;stroke-width:2.4;stroke-linecap:round;stroke-linejoin:round}'
    trend_rule = '.chart .trend-series{fill:none;stroke-width:2;stroke-dasharray:6 5;stroke-linecap:round;opacity:.88}'
    while trend_rule + trend_rule in html_source:
        html_source = html_source.replace(trend_rule + trend_rule, trend_rule)
    if trend_rule not in html_source:
        html_source = html_source.replace(series_rule, series_rule + trend_rule)
    html_source = html_source.replace(
        '.legend{display:flex;gap:14px;font-size:10px;color:var(--muted);margin:8px 0 1px}',
        '.legend{display:flex;flex-wrap:wrap;gap:10px 14px;font-size:10px;color:var(--muted);margin:8px 0 1px}')
    html_source = html_source.replace(
        '@media(max-width:520px){header{display:block}',
        '@media(max-width:520px){.plot-head{align-items:flex-start;flex-wrap:wrap}.plot-title{flex-wrap:wrap}.plot-total{text-align:left;max-width:100%}.type-select{min-width:140px;max-width:185px}header{display:block}')

    js_start = html_source.index("function fmt(n){")
    js_end = html_source.index("document.getElementById('districtSelect').innerHTML=", js_start)
    replacement = r'''function fmt(n){return Number(n||0).toLocaleString('ko-KR');}
function yCeiling(values){
  const peak=Math.max(1,...values), raw=peak/4, magnitude=Math.pow(10,Math.floor(Math.log10(raw))), normalized=raw/magnitude;
  const step=(normalized<=1?1:normalized<=2?2:normalized<=2.5?2.5:normalized<=5?5:10)*magnitude;
  return Math.ceil(peak/step)*step;
}
function getSelectedSeries(source,filter){
  const entry=categoryData.districts[currentDistrict][source];
  if(filter==='all')return entry.all;
  if(filter.startsWith('family::'))return entry.families[filter.slice(8)]||entry.all.map(()=>[0,0]);
  return entry.types[filter.slice(6)]||entry.all.map(()=>[0,0]);
}
function getSelectedSurvival(source,filter){
  const entry=categoryData.survival[currentDistrict][source];
  if(filter==='all')return entry.all;
  if(filter.startsWith('family::'))return entry.families[filter.slice(8)]||[0,0,0,0];
  return entry.types[filter.slice(6)]||[0,0,0,0];
}
function getSelectedStock(source,filter){
  const entry=categoryData.stock[currentDistrict][source];
  if(filter==='all')return entry.all;
  if(filter.startsWith('family::'))return entry.families[filter.slice(8)]||0;
  return entry.types[filter.slice(6)]||0;
}
function filterLabel(source,filter){
  if(filter==='all')return '전체 업태';
  if(filter.startsWith('family::'))return filter.slice(8);
  return filter.slice(6);
}
function makeChart(targetId,tipId,series){
  const target=document.getElementById(targetId), tip=document.getElementById(tipId);
  const rows=categoryData.periods.map((m,i)=>({m,o:series[i][0],c:series[i][1]}));
  const vw=840,vh=250,left=52,right=13,top=12,bottom=42,pw=vw-left-right,ph=vh-top-bottom;
  const opens=rows.map(r=>r.o), closes=rows.map(r=>r.c), ymax=yCeiling(opens.concat(closes));
  const x=i=>left+pw*i/(rows.length-1), y=v=>top+ph*(1-v/ymax);
  const movingAverage=key=>rows.map((row,i)=>i<11?null:rows.slice(i-11,i+1).reduce((sum,item)=>sum+item[key],0)/12);
  const trendOpen=movingAverage('o'), trendClose=movingAverage('c');
  let svg=`<svg class="chart" viewBox="0 0 ${vw} ${vh}" role="img" aria-label="월별 개업 폐업 건수와 후행 12개월 이동평균 그래프">`;
  for(let j=0;j<=4;j++){
    const value=ymax*j/4, yy=y(value);
    svg+=`<line class="grid" x1="${left}" y1="${yy}" x2="${vw-right}" y2="${yy}"/><text x="${left-9}" y="${yy+3}" text-anchor="end">${fmt(value)}</text>`;
  }
  svg+=`<line class="axis" x1="${left}" y1="${top}" x2="${left}" y2="${top+ph}"/><line class="axis" x1="${left}" y1="${top+ph}" x2="${vw-right}" y2="${top+ph}"/>`;
  for(let i=0;i<rows.length;i+=3){const xx=x(i),label=rows[i].m.replace('-','.');svg+=`<line class="grid" x1="${xx}" y1="${top}" x2="${xx}" y2="${top+ph}" opacity=".55"/><text x="${xx}" y="${vh-12}" text-anchor="middle">${label}</text>`;}
  const pathFor=(key)=>rows.map((r,i)=>`${i?'L':'M'}${x(i).toFixed(2)},${y(r[key]).toFixed(2)}`).join(' ');
  const trendPath=values=>values.map((value,i)=>value===null?'':`${i===11?'M':'L'}${x(i).toFixed(2)},${y(value).toFixed(2)}`).filter(Boolean).join(' ');
  svg+=`<path class="series" d="${pathFor('o')}" stroke="#247f78"/><path class="series" d="${pathFor('c')}" stroke="#cb6c75"/>`;
  svg+=`<path class="trend-series" d="${trendPath(trendOpen)}" stroke="#247f78"/><path class="trend-series" d="${trendPath(trendClose)}" stroke="#cb6c75"/>`;
  for(let i=0;i<rows.length;i++)svg+=`<circle cx="${x(i)}" cy="${y(rows[i].o)}" r="2.4" fill="#247f78"/><circle cx="${x(i)}" cy="${y(rows[i].c)}" r="2.4" fill="#cb6c75"/>`;
  const band=pw/rows.length;
  for(let i=0;i<rows.length;i++){const xx=Math.max(left,x(i)-band/2),ww=Math.min(band,left+pw-xx);svg+=`<rect class="month-hit" data-index="${i}" x="${xx}" y="${top}" width="${ww}" height="${ph}"/>`;}
  svg+='</svg>';target.innerHTML=svg;
  const plot=target.closest('.plot-card');
  target.querySelector('svg').addEventListener('pointermove',event=>{
    const hit=event.target.closest('[data-index]');if(!hit)return;
    const row=rows[Number(hit.dataset.index)];
    const i=Number(hit.dataset.index), openTrend=trendOpen[i]===null?'—':fmt(trendOpen[i].toFixed(1)), closeTrend=trendClose[i]===null?'—':fmt(trendClose[i].toFixed(1));
    tip.innerHTML=`<strong>${row.m.replace('-','.')}</strong><span class="tip-open">개업 ${fmt(row.o)}</span> &nbsp;·&nbsp; <span class="tip-close">폐업 ${fmt(row.c)}</span><br><span>후행 12개월 평균: 개업 ${openTrend} · 폐업 ${closeTrend}</span>`;
    const r=plot.getBoundingClientRect();tip.style.display='block';
    tip.style.left=Math.max(9,Math.min(event.clientX-r.left+10,r.width-190))+'px';
    tip.style.top=Math.max(43,Math.min(event.clientY-r.top-51,r.height-58))+'px';
  });
  target.querySelector('svg').addEventListener('pointerleave',()=>{tip.style.display='none';});
}
let currentDistrict='중구';
function populateTypeSelect(selectId,source){
  const select=document.getElementById(selectId);let options='<option value="all">전체 업태</option>';
  for(const family of categoryData.families[source])options+=`<option value="family::${family}">업태군 · ${family}</option>`;
  options+='<optgroup label="원자료 세부 업태">';
  for(const label of categoryData.types[source])options+=`<option value="type::${label}">${label==='업태 미상'?'업태 미상':label}</option>`;
  select.innerHTML=options+'</optgroup>';select.addEventListener('change',()=>renderPlots());
}
function updatePanel(source,filter,headingId,totalId,chartId,tipId){
  const selected=filterLabel(source,filter), series=getSelectedSeries(source,filter);
  const totalSeries=categoryData.districts[currentDistrict][source].all;
  const openings=series.slice(24).reduce((a,v)=>a+v[0],0), closures=series.slice(24).reduce((a,v)=>a+v[1],0), net=openings-closures;
  const totalOpen=totalSeries.slice(24).reduce((a,v)=>a+v[0],0), totalClose=totalSeries.slice(24).reduce((a,v)=>a+v[1],0);
  const [survived,known,eligible,unknown]=getSelectedSurvival(source,filter);
  const retention=known?`${(survived/known*100).toFixed(1)}%`:'자료 부족';
  const share=`해당 구·유형 내 비중: 개업 ${totalOpen?(openings/totalOpen*100).toFixed(1):'0.0'}% · 폐업 ${totalClose?(closures/totalClose*100).toFixed(1):'0.0'}%`;
  document.getElementById(headingId).textContent=`${source} · ${selected}`;
  document.getElementById(totalId).innerHTML=`2025 개업 ${fmt(openings)} / 폐업 ${fmt(closures)} / 기록차 <span class="${net>=0?'net-positive':'net-negative'}">${net>0?'+':''}${fmt(net)}</span><br><span class="subline">현재 영업 ${fmt(getSelectedStock(source,filter))} · 12개월 유지율 ${retention} (n=${fmt(known)})</span><br><span class="subline">${share} (개업 / 폐업)</span>`;
  makeChart(chartId,tipId,series);
}
function renderPlots(){
  updatePanel('일반음식점',document.getElementById('generalTypeSelect').value,'generalHeading','generalTotal','generalChart','generalTip');
  updatePanel('휴게음식점',document.getElementById('snackTypeSelect').value,'snackHeading','snackTotal','snackChart','snackTip');
}
function renderDetails(name){
  currentDistrict=name;
  const series=categoryData.districts[name];
  const totals={go:0,gc:0,so:0,sc:0};
  for(let i=24;i<36;i++){totals.go+=series['일반음식점'].all[i][0];totals.gc+=series['일반음식점'].all[i][1];totals.so+=series['휴게음식점'].all[i][0];totals.sc+=series['휴게음식점'].all[i][1];}
  const values=[['일반 개업','go','#247f78'],['일반 폐업','gc','#cb6c75'],['휴게 개업','so','#247f78'],['휴게 폐업','sc','#cb6c75']];
  document.getElementById('metrics').innerHTML=values.map(([label,key,color])=>`<div class="metric"><div class="metric-label"><i class="dot" style="background:${color}"></i>${label} · 2025</div><div class="metric-value">${fmt(totals[key])}건</div></div>`).join('');
  renderPlots();
}
populateTypeSelect('generalTypeSelect','일반음식점');
populateTypeSelect('snackTypeSelect','휴게음식점');
'''
    html_source = html_source[:js_start] + replacement + html_source[js_end:]
    DASHBOARD.write_text(html_source, encoding="utf-8")

    # Concise findings report, including both event mix and the data limits.
    total_by_year = Counter()
    subtype_by_year = Counter()
    family_by_year = Counter()
    district_family_by_year = Counter()
    for (month, district, source, label, event), value in counts.items():
        year = int(month[:4])
        total_by_year[(year, source, event)] += value
        subtype_by_year[(year, source, label, event)] += value
        family_by_year[(year, source, family_for(label), event)] += value
        district_family_by_year[(year, district, source, family_for(label), event)] += value

    report = OUT / "지역별_업태별_개폐업_EDA.md"
    with report.open("w", encoding="utf-8") as stream:
        stream.write("# 자치구·업태별 개업·폐업 EDA (2023–2025)\n\n")
        stream.write("## 분석 기준\n\n")
        stream.write("- 원자료 `business_type`은 유지하고, 비교를 돕는 업태군을 별도 매핑했습니다. 업태군은 탐색용이며 법정 업종분류나 소비수요 분류와 동일하지 않습니다.\n")
        stream.write("- 개폐업 사건과 정리본은 `(source_type, source_row_number)`로 연결했습니다. 관리번호만으로 결합하지 않아 휴게음식점 중복 관리번호의 행 혼합을 피했습니다.\n")
        stream.write("- 자치구는 현재 스냅샷의 지번주소 우선, 도로명주소 보완으로 매핑했습니다. 과거 사건 당시의 주소·업태와 다를 수 있습니다.\n")
        stream.write(f"- 현재 스냅샷 최대 자료갱신일은 {data_cutoff.isoformat()}입니다. 12개월 유지율은 이 날짜까지 개업 후 12개월 이상 관찰 가능한 기록만 대상으로 했고, 결과를 알 수 없는 폐업일 결측은 분모에서 제외했습니다.\n")
        stream.write("- 현재 영업 재고는 스냅샷 시점의 영업 상태를 관리번호별 한 건으로 센 규모 지표이며, 과거 월별 점포 재고 추이가 아닙니다.\n")
        stream.write("- 개업−폐업은 기록 사건 차이입니다. 점포 재고 변화나 지역 활성도, 핫플레이스를 단독으로 판정하지 않습니다.\n\n")
        stream.write("### 자치구 매핑·결합 범위\n\n")
        stream.write("| 인허가 유형 | 사건 | 기간 사건 행 | 자치구 매핑 | 매핑 제외 | 매핑률 | 정리본 결합 실패 |\n|---|---|---:|---:|---:|---:|---:|\n")
        for source in SOURCES:
            for event in EVENT_TYPES:
                observed = observed_by_source[(source, event)]
                dropped_map = unmapped[(source, event)]
                failed_join = missing_registry[(source, event)]
                mapped = observed - dropped_map - failed_join
                stream.write(f"| {source} | {event} | {observed:,} | {mapped:,} | {dropped_map:,} | {pct(mapped,observed):.2f}% | {failed_join:,} |\n")

        stream.write("\n")
        stream.write("## 업태군 매핑\n\n")
        stream.write("| 업태군 | 묶은 의미 | 주의할 점 |\n|---|---|---|\n")
        notes = {
            "한식·고기·해산물": "한식, 숯불구이, 냉면, 탕류, 복어, 횟집",
            "외식 전문점": "경양식·일식·중식·외국음식·뷔페 등",
            "간편식·분식·치킨": "분식·김밥·치킨·패스트푸드·조리판매·푸드트럭",
            "카페·차·디저트": "커피숍·까페·다방·찻집·과자·아이스크림·떡카페",
            "주류·야간": "호프/통닭·소주방·감성주점·라이브카페; 호프/통닭은 혼합 라벨",
            "소매·입지·서비스": "편의점·백화점·역·공항·호텔·극장·유원지·키즈카페·출장/이동조리",
            "기타·미분류": "기타·기타 휴게음식점·미상; 폭넓은 잔여 범주",
        }
        for family in FAMILY_ORDER:
            stream.write(f"| {family} | {notes[family]} | 원자료 세부 업태로 다시 확인 필요 |\n")

        stream.write("\n## 연도별 개업·폐업 흐름\n\n")
        stream.write("| 연도 | 인허가 유형 | 개업 | 폐업 | 기록 순증감 |\n|---:|---|---:|---:|---:|\n")
        for year in (2023, 2024, 2025):
            for source in SOURCES:
                opening = total_by_year[(year, source, "개업")]
                closure = total_by_year[(year, source, "폐업")]
                stream.write(f"| {year} | {source} | {opening:,} | {closure:,} | {opening-closure:+,} |\n")

        stream.write("\n## 2023–2025 세부 업태 상위\n\n")
        for source in SOURCES:
            stream.write(f"### {source}\n\n")
            stream.write("| 사건 | 세부 업태 | 3년 건수 | 전체 사건 내 비중 |\n|---|---|---:|---:|\n")
            source_total = sum(total_by_year[(year, source, event)] for year in (2023, 2024, 2025) for event in EVENT_TYPES)
            combined = Counter()
            for (year, s, label, event), value in subtype_by_year.items():
                if s == source:
                    combined[(label, event)] += value
            for event in EVENT_TYPES:
                for (label, ev), value in sorted(((k, v) for k, v in combined.items() if k[1] == event), key=lambda x: -x[1])[:8]:
                    stream.write(f"| {event} | {label or '업태 미상'} | {value:,} | {pct(value, source_total):.1f}% |\n")
            stream.write("\n")

        stream.write("## 업태군별 연간 구성\n\n")
        stream.write("각 업태군의 개업·폐업 비중은 같은 해·같은 인허가 유형의 자치구 합계 대비입니다.\n\n")
        stream.write("| 연도 | 인허가 유형 | 업태군 | 개업 | 개업 비중 | 폐업 | 폐업 비중 | 기록 순증감 |\n|---:|---|---|---:|---:|---:|---:|---:|\n")
        for year in (2023, 2024, 2025):
            for source in SOURCES:
                for family in FAMILY_ORDER:
                    opening = family_by_year[(year, source, family, "개업")]
                    closure = family_by_year[(year, source, family, "폐업")]
                    if not opening and not closure:
                        continue
                    stream.write(f"| {year} | {source} | {family} | {opening:,} | {pct(opening,total_by_year[(year,source,'개업')]):.1f}% | {closure:,} | {pct(closure,total_by_year[(year,source,'폐업')]):.1f}% | {opening-closure:+,} |\n")

        stream.write("\n## 자치구별 업태군 변화 사례\n\n")
        stream.write("2025년과 2023년의 연간 사건 건수 차이를 정렬했습니다. 비교가 어려운 폭넓은 `기타·미분류`는 제외했고, 시작 건수가 작은 조합의 급격한 비율 변동을 피하도록 2023년 개업이 20건 이상인 조합만 포함했습니다. 절대 건수이며 점포 수나 면적 기준으로 표준화하지 않았습니다.\n\n")
        stream.write("| 인허가 유형 | 자치구 | 업태군 | 2023 개업 | 2025 개업 | 개업 변화 | 2023 폐업 | 2025 폐업 | 폐업 변화 |\n|---|---|---|---:|---:|---:|---:|---:|---:|\n")
        district_deltas = []
        for district in GUS:
            for source in SOURCES:
                for family in FAMILY_ORDER:
                    op23 = district_family_by_year[(2023,district,source,family,"개업")]
                    op25 = district_family_by_year[(2025,district,source,family,"개업")]
                    cl23 = district_family_by_year[(2023,district,source,family,"폐업")]
                    cl25 = district_family_by_year[(2025,district,source,family,"폐업")]
                    if family != "기타·미분류" and op23 >= 20:
                        district_deltas.append((op25-op23, district, source, family, op23, op25, cl23, cl25))
        for delta, district, source, family, op23, op25, cl23, cl25 in sorted(district_deltas, reverse=True)[:6]:
            stream.write(f"| {source} | {district} | {family} | {op23:,} | {op25:,} | {delta:+,} | {cl23:,} | {cl25:,} | {cl25-cl23:+,} |\n")
        for delta, district, source, family, op23, op25, cl23, cl25 in sorted(district_deltas)[:6]:
            stream.write(f"| {source} | {district} | {family} | {op23:,} | {op25:,} | {delta:+,} | {cl23:,} | {cl25:,} | {cl25-cl23:+,} |\n")

        stream.write("\n## 12개월 유지 신호\n\n")
        stream.write("유지율은 허가 후 최소 12개월 관찰된 개업 기록 중, 12개월 이전 폐업 기록이 없는 비율입니다. 폐업일이 비어 있는데 현재 상태가 폐업인 기록은 결과 미상으로 제외했습니다.\n\n")
        stream.write("| 인허가 유형 | 업태군 | 유지율 | 판정 가능한 개업 | 12개월 내 폐업 | 결과 미상 |\n|---|---|---:|---:|---:|---:|\n")
        family_survival = Counter()
        for (district, source, label, result), value in survival.items():
            family = family_for(label)
            family_survival[(source, family, result)] += value
        survival_rank = []
        for source in SOURCES:
            for family in FAMILY_ORDER:
                survived = family_survival[(source, family, "survived")]
                failed = family_survival[(source, family, "failed")]
                unknown = family_survival[(source, family, "unknown")]
                eligible = family_survival[(source, family, "eligible")]
                valid = survived + failed
                if eligible:
                    survival_rank.append((source, family, survived, failed, unknown, eligible, valid))
        for source, family, survived, failed, unknown, eligible, valid in survival_rank:
            stream.write(f"| {source} | {family} | {pct(survived,valid):.1f}% | {valid:,} | {failed:,} | {unknown:,} |\n")

        stream.write("\n## 지역 패턴을 읽는 기준\n\n")
        stream.write("- **개업 확대 후보:** 세부 업태/업태군의 월별 개업이 전년 동월 대비 반복 상승하고, 12개월 유지율과 현재 영업 재고도 뒷받침되는지 확인합니다.\n")
        stream.write("- **고회전 후보:** 개업과 폐업이 모두 많고 기록 순증감은 작거나 음수이며, 12개월 유지율도 낮은지 봅니다.\n")
        stream.write("- **업종 재편 후보:** 전체 음식점 수의 흐름과 별개로 카페·간편식 등 특정 업태군의 구성비가 커지고 다른 업태군은 줄어드는지 확인합니다.\n")
        stream.write("- **쇠퇴 후보:** 개업 감소·폐업 증가가 여러 달 지속되고, 현재 영업 재고와 외부 방문·매출 같은 수요 지표도 함께 약해지는지 확인합니다.\n")
        stream.write("\n현재 음식점 자료만으로 네 유형을 확정하지 않고, 업태별 변화 후보를 추리는 단계로 사용합니다. 성장 판정에는 인구·점포 수로 표준화한 밀도, 방문/체류·매출 지표와 공간·시간상 지속성이 필요합니다. 온라인 검색 관심도는 타깃 정의가 아니라 별도 선행 예측 신호로 평가합니다.\n\n")
        stream.write("## 자치구별 월간 파일\n\n")
        stream.write("`지역구별_월별_세부업태_개폐업_2023-2025.csv`는 자치구×인허가 유형×원자료 업태×월의 개업·폐업 건수, 기록 순증감, 사건 회전량, 해당 구·유형 내 업태 비중을 담습니다. `지역구별_월별_업태군_개폐업_2023-2025.csv`는 같은 정보를 업태군으로 합쳤습니다.\n\n")
        stream.write("대시보드는 자치구 선택 후 일반·휴게음식점 패널마다 전체 업태, 임시 업태군 또는 원자료 세부 업태를 선택할 수 있습니다. X축은 `YYYY.MM`입니다.\n")

    print(f"기간 사건 행: {len(event_rows):,}")
    print("자치구 매핑률: " + ", ".join(
        f"{source} {100*(1-sum(unmapped[(source,e)] + missing_registry[(source,e)] for e in EVENT_TYPES)/max(1,sum(observed_by_source[(source,e)] for e in EVENT_TYPES))):.2f}%"
        for source in SOURCES))
    print(f"12개월 유지 추적 기준일: {data_cutoff.isoformat()}")
    print(f"세부 업태 월별 행: {len(monthly_subtype_rows):,} / 업태군 월별 행: {len(monthly_family_rows):,}")
    print(f"대시보드 크기: {DASHBOARD.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
