#!/usr/bin/env python3
"""Create a provisional monthly opening/closure count series for EDA."""

import csv
from collections import Counter
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "전처리_결과"
EVENTS = DATA_DIR / "음식점_개폐업사건.csv"
MONTHLY = DATA_DIR / "음식점_월별집계_잠정.csv"
SUMMARY = DATA_DIR / "음식점_월별_EDA_요약.md"
START = "2018-01"
FULL_YEAR_END = "2025-12"
END = "2026-10"
SOURCE_TYPES = ("일반음식점", "휴게음식점")
EVENT_TYPES = ("개업", "폐업")


def month_range(start, end):
    year, month = map(int, start.split("-"))
    end_year, end_month = map(int, end.split("-"))
    while (year, month) <= (end_year, end_month):
        yield f"{year:04d}-{month:02d}"
        month += 1
        if month == 13:
            year += 1
            month = 1


def main():
    counts = Counter()
    seen = set()
    with EVENTS.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            event_date = row["event_date"]
            if date.fromisoformat(event_date).strftime("%Y-%m") != row["event_month"]:
                raise ValueError(f"event_month와 event_date가 다릅니다: {event_date}")
            month = row["event_month"]
            if not START <= month <= END:
                continue
            key = (row["source_type"], row["management_id"], row["event_type"], event_date)
            if key in seen:
                raise ValueError(f"중복 사건 키가 남아 있습니다: {key}")
            seen.add(key)
            counts[(month, row["source_type"], row["event_type"])] += 1

    months = list(month_range(START, END))
    with MONTHLY.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["month", "year", "month_of_year", "period_status", "source_type", "event_type", "event_count"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for month in months:
            if month <= FULL_YEAR_END:
                period_status = "잠정_완전연도구간"
            elif month == END:
                period_status = "부분월_2026-10"
            else:
                period_status = "부분연도_2026"
            year, month_of_year = month.split("-")
            for source_type in SOURCE_TYPES:
                for event_type in EVENT_TYPES:
                    writer.writerow({
                        "month": month,
                        "year": year,
                        "month_of_year": month_of_year,
                        "period_status": period_status,
                        "source_type": source_type,
                        "event_type": event_type,
                        "event_count": counts[(month, source_type, event_type)],
                    })

    annual = Counter()
    seasonal = Counter()
    zero_months = Counter()
    for month in month_range(START, FULL_YEAR_END):
        year, month_of_year = month.split("-")
        for source_type in SOURCE_TYPES:
            for event_type in EVENT_TYPES:
                count = counts[(month, source_type, event_type)]
                annual[(year, source_type, event_type)] += count
                seasonal[(month_of_year, source_type, event_type)] += count
                if count == 0:
                    zero_months[(source_type, event_type)] += 1
    partial_2026 = Counter()
    for month in month_range("2026-01", END):
        for source_type in SOURCE_TYPES:
            for event_type in EVENT_TYPES:
                partial_2026[(source_type, event_type)] += counts[(month, source_type, event_type)]

    with SUMMARY.open("w", encoding="utf-8") as report:
        report.write("# 음식점 월별 EDA 2단계: 개업·폐업 사건 수\n\n")
        report.write(f"- 잠정 분석 구간: {START}~{FULL_YEAR_END} (8개 완전 연도)\n")
        report.write(f"- 부분 구간: 2026-01~{END}; 2026년 10월은 부분 월로 별도 표시\n")
        report.write("- 월별 사건 수는 현재 인허가 스냅샷의 사건일을 집계한 값이며, 기록의 과거 완전성을 보증하지 않습니다.\n")
        report.write("- 순증감은 개업 사건 수−폐업 사건 수로 별도 계산할 수 있지만, 실제 영업 점포 수와 동일하지 않습니다.\n")
        report.write("- 월별 데이터는 0건인 월도 0으로 채웠습니다. 이는 해당 월의 기록된 사건이 없다는 뜻이며 원자료 완전성 검증과 구분해야 합니다.\n\n")
        report.write("## 연도별 사건 수\n\n")
        report.write("| 연도 | 일반 개업 | 일반 폐업 | 휴게 개업 | 휴게 폐업 |\n|---|---:|---:|---:|---:|\n")
        for year in range(2018, 2026):
            y = str(year)
            vals = [annual[(y, source, event)] for source, event in (
                ("일반음식점", "개업"), ("일반음식점", "폐업"),
                ("휴게음식점", "개업"), ("휴게음식점", "폐업"),
            )]
            report.write(f"| {y} | " + " | ".join(f"{value:,}" for value in vals) + " |\n")
        vals_2026 = [partial_2026[(source, event)] for source, event in (
            ("일반음식점", "개업"), ("일반음식점", "폐업"),
            ("휴게음식점", "개업"), ("휴게음식점", "폐업"),
        )]
        report.write("| 2026 (부분) | " + " | ".join(f"{value:,}" for value in vals_2026) + " |\n\n")
        report.write("## 계절성 탐색용 월평균 사건 수 (2018~2025)\n\n")
        report.write("| 월 | 일반 개업 | 일반 폐업 | 휴게 개업 | 휴게 폐업 |\n|---|---:|---:|---:|---:|\n")
        for month_of_year in range(1, 13):
            m = f"{month_of_year:02d}"
            means = [seasonal[(m, source, event)] / 8 for source, event in (
                ("일반음식점", "개업"), ("일반음식점", "폐업"),
                ("휴게음식점", "개업"), ("휴게음식점", "폐업"),
            )]
            report.write(f"| {m} | " + " | ".join(f"{value:.1f}" for value in means) + " |\n")
        report.write("\n## 0건 월 수 (2018~2025)\n\n")
        for source_type in SOURCE_TYPES:
            for event_type in EVENT_TYPES:
                report.write(f"- {source_type} {event_type}: {zero_months[(source_type, event_type)]} / 96개월\n")
        report.write("\n이 표는 초기 집계입니다. 다음 EDA 단계에서 월별 그래프, 추세·계절 분해, 연도별 변화와 이상 월을 검토합니다.\n")

    print(f"월별 집계 행: {len(months) * len(SOURCE_TYPES) * len(EVENT_TYPES):,}")
    print(f"월별 집계 파일: {MONTHLY}")
    print(f"요약 파일: {SUMMARY}")


if __name__ == "__main__":
    main()
