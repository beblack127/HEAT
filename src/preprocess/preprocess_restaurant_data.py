#!/usr/bin/env python3
"""Prepare Seoul restaurant permit registers for monthly EDA.

Raw files are read only. Outputs are UTF-8 CSVs in 전처리_결과/.
The script intentionally does not select an analysis window, classify cafes,
or map records to administrative districts.
"""

import csv
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "전처리_결과"
ENCODING = "cp949"
SOURCES = [
    ("일반음식점", ROOT / "서울시 일반음식점 인허가 정보.csv"),
    ("휴게음식점", ROOT / "서울시 휴게음식점 인허가 정보.csv"),
]

REGISTRY_FIELDS = [
    "source_type", "source_row_number", "management_id", "duplicate_management_id_flag",
    "permit_date_raw", "permit_date", "permit_month", "permit_date_parse_status",
    "closure_date_raw", "closure_date", "closure_month", "closure_date_parse_status",
    "status_code_raw", "status_raw", "status_normalized",
    "business_type_raw", "business_type", "business_name",
    "lot_address", "road_address", "coordinate_x", "coordinate_y", "coordinate_source",
    "local_government_code", "legal_dong_code", "last_modified_raw", "data_update_raw",
    "pre_1960_permit_date_flag", "closed_status_missing_closure_date_flag",
    "active_status_with_closure_date_flag", "address_missing_both_flag",
    "coordinate_pair_status", "encoding_replacement_flag", "encoding_replacement_fields",
]

EVENT_FIELDS = [
    "source_type", "management_id", "event_type", "event_date", "event_month",
    "business_type", "status_in_current_snapshot", "source_row_number",
    "duplicate_management_id_flag", "record_quality_flags",
]


def text(value):
    return "" if value is None else str(value).strip()


def parse_date(value):
    """Parse the first ten characters while preserving missing/invalid states."""
    raw = "" if value is None else str(value)
    token = raw.strip()[:10]
    if not token:
        return "", "missing"
    try:
        return date.fromisoformat(token).isoformat(), "valid"
    except ValueError:
        if len(token) == 8 and token.isdigit():
            try:
                return date(int(token[:4]), int(token[4:6]), int(token[6:8])).isoformat(), "valid"
            except ValueError:
                pass
    return "", "invalid"


def normalized_status(value):
    raw = text(value)
    if raw in {"영업", "영업/정상"}:
        return "영업"
    if raw == "폐업":
        return "폐업"
    return raw if raw else "미상"


def selected_coordinates(row):
    primary_x = text(row.get("좌표정보(X)"))
    primary_y = text(row.get("좌표정보(Y)"))
    fallback_x = text(row.get("X좌표"))
    fallback_y = text(row.get("Y좌표"))
    if primary_x and primary_y:
        return primary_x, primary_y, "좌표정보(X/Y)", "완전"
    if fallback_x and fallback_y:
        return fallback_x, fallback_y, "X좌표/Y좌표", "완전"
    if primary_x or primary_y:
        return primary_x, primary_y, "좌표정보(X/Y)", "부분"
    if fallback_x or fallback_y:
        return fallback_x, fallback_y, "X좌표/Y좌표", "부분"
    return "", "", "", "없음"


def blank_stats(path):
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "rows": 0,
        "fields": [],
        "missing": Counter(),
        "statuses": Counter(),
        "business_types": Counter(),
        "permit": {"missing": 0, "invalid": 0, "valid": 0, "min": "", "max": "", "pre_1960": 0},
        "closure": {"missing": 0, "invalid": 0, "valid": 0, "min": "", "max": ""},
        "closure_before_permit": 0,
        "closed_status_missing_closure": 0,
        "active_status_with_closure": 0,
        "address_missing_both": 0,
        "coordinate_status": Counter(),
        "replacement_rows": 0,
        "replacement_cells": 0,
        "replacement_fields": Counter(),
        "data_update_min": "",
        "data_update_max": "",
        "management_id_counts": Counter(),
    }


def update_date_range(target, parsed):
    if not parsed:
        return
    if not target["min"] or parsed < target["min"]:
        target["min"] = parsed
    if not target["max"] or parsed > target["max"]:
        target["max"] = parsed


def scan_source(source_type, path):
    stats = blank_stats(path)
    with path.open("r", encoding=ENCODING, errors="replace", newline="") as stream:
        reader = csv.DictReader(stream)
        stats["fields"] = reader.fieldnames or []
        required = {"관리번호", "인허가일자", "폐업일자", "영업상태명"}
        missing_headers = sorted(required.difference(stats["fields"]))
        if missing_headers:
            raise ValueError(f"{path.name}: 필수 열이 없습니다: {missing_headers}")
        for row in reader:
            stats["rows"] += 1
            management_id = text(row.get("관리번호"))
            stats["management_id_counts"][management_id] += 1
            for field in ("관리번호", "인허가일자", "폐업일자", "영업상태명", "업태구분명", "지번주소", "도로명주소", "법정동코드"):
                if field in row and not text(row.get(field)):
                    stats["missing"][field] += 1

            status = normalized_status(row.get("영업상태명"))
            stats["statuses"][text(row.get("영업상태명")) or "(공란)"] += 1
            stats["business_types"][text(row.get("업태구분명")) or "(공란)"] += 1
            permit, permit_state = parse_date(row.get("인허가일자"))
            closure, closure_state = parse_date(row.get("폐업일자"))
            stats["permit"][permit_state] += 1
            stats["closure"][closure_state] += 1
            update_date_range(stats["permit"], permit)
            update_date_range(stats["closure"], closure)
            if permit and date.fromisoformat(permit).year < 1960:
                stats["permit"]["pre_1960"] += 1
            if permit and closure and closure < permit:
                stats["closure_before_permit"] += 1
            if status == "폐업" and not closure:
                stats["closed_status_missing_closure"] += 1
            if status == "영업" and closure:
                stats["active_status_with_closure"] += 1
            if not (text(row.get("지번주소")) or text(row.get("도로명주소"))):
                stats["address_missing_both"] += 1
            _, _, _, coord_status = selected_coordinates(row)
            stats["coordinate_status"][coord_status] += 1

            replacement_fields = [key for key, value in row.items() if key and value and "\ufffd" in value]
            if replacement_fields:
                stats["replacement_rows"] += 1
                stats["replacement_cells"] += len(replacement_fields)
                stats["replacement_fields"].update(replacement_fields)

            updated, update_state = parse_date(row.get("데이터갱신일자"))
            if update_state == "valid":
                if not stats["data_update_min"] or updated < stats["data_update_min"]:
                    stats["data_update_min"] = updated
                if not stats["data_update_max"] or updated > stats["data_update_max"]:
                    stats["data_update_max"] = updated

    ids = stats["management_id_counts"]
    stats["blank_management_ids"] = ids.get("", 0)
    stats["unique_management_ids"] = sum(1 for key in ids if key)
    stats["duplicate_ids"] = {key for key, count in ids.items() if key and count > 1}
    stats["duplicate_id_groups"] = len(stats["duplicate_ids"])
    stats["duplicate_excess_rows"] = sum(count - 1 for key, count in ids.items() if key and count > 1)
    return stats


def clean_row(source_type, source_row_number, row, duplicate_ids):
    management_id = text(row.get("관리번호"))
    permit, permit_state = parse_date(row.get("인허가일자"))
    closure, closure_state = parse_date(row.get("폐업일자"))
    status = normalized_status(row.get("영업상태명"))
    coord_x, coord_y, coord_source, coord_status = selected_coordinates(row)
    replacement_fields = [key for key, value in row.items() if key and value and "\ufffd" in value]
    lot_address = text(row.get("지번주소"))
    road_address = text(row.get("도로명주소"))
    quality_flags = []
    if management_id in duplicate_ids:
        quality_flags.append("중복관리번호")
    if permit_state != "valid":
        quality_flags.append(f"인허가일_{permit_state}")
    if closure_state == "invalid":
        quality_flags.append("폐업일_invalid")
    if permit and date.fromisoformat(permit).year < 1960:
        quality_flags.append("인허가일_1960년이전")
    if status == "폐업" and not closure:
        quality_flags.append("폐업일미상_폐업상태")
    if status == "영업" and closure:
        quality_flags.append("영업상태_폐업일존재")
    if not lot_address and not road_address:
        quality_flags.append("주소없음")
    if coord_status != "완전":
        quality_flags.append(f"좌표{coord_status}")
    if replacement_fields:
        quality_flags.append("인코딩대체문자")

    return {
        "source_type": source_type,
        "source_row_number": source_row_number,
        "management_id": management_id,
        "duplicate_management_id_flag": int(management_id in duplicate_ids),
        "permit_date_raw": row.get("인허가일자") or "",
        "permit_date": permit,
        "permit_month": permit[:7] if permit else "",
        "permit_date_parse_status": permit_state,
        "closure_date_raw": row.get("폐업일자") or "",
        "closure_date": closure,
        "closure_month": closure[:7] if closure else "",
        "closure_date_parse_status": closure_state,
        "status_code_raw": text(row.get("영업상태코드")),
        "status_raw": text(row.get("영업상태명")),
        "status_normalized": status,
        "business_type_raw": row.get("업태구분명") or "",
        "business_type": text(row.get("업태구분명")),
        "business_name": text(row.get("사업장명")),
        "lot_address": lot_address,
        "road_address": road_address,
        "coordinate_x": coord_x,
        "coordinate_y": coord_y,
        "coordinate_source": coord_source,
        "local_government_code": text(row.get("개방자치단체코드")),
        "legal_dong_code": text(row.get("법정동코드")),
        "last_modified_raw": text(row.get("최종수정일자")),
        "data_update_raw": text(row.get("데이터갱신일자")),
        "pre_1960_permit_date_flag": int(bool(permit) and date.fromisoformat(permit).year < 1960),
        "closed_status_missing_closure_date_flag": int(status == "폐업" and not closure),
        "active_status_with_closure_date_flag": int(status == "영업" and bool(closure)),
        "address_missing_both_flag": int(not lot_address and not road_address),
        "coordinate_pair_status": coord_status,
        "encoding_replacement_flag": int(bool(replacement_fields)),
        "encoding_replacement_fields": ";".join(replacement_fields),
        "_quality_flags": ";".join(quality_flags),
        "_permit_date_state": permit_state,
        "_closure_date_state": closure_state,
    }


def duplicate_group_summary(path, duplicate_ids):
    baselines = {}
    differing_fields = defaultdict(set)
    with path.open("r", encoding=ENCODING, errors="replace", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            management_id = text(row.get("관리번호"))
            if management_id not in duplicate_ids:
                continue
            normalized = {key: text(value) for key, value in row.items() if key}
            if management_id not in baselines:
                baselines[management_id] = normalized
                continue
            first = baselines[management_id]
            for key in set(first) | set(normalized):
                if first.get(key, "") != normalized.get(key, ""):
                    differing_fields[management_id].add(key)
    field_groups = Counter(field for fields in differing_fields.values() for field in fields)
    return {
        "duplicate_groups_identical_after_trim": sum(1 for key in duplicate_ids if not differing_fields.get(key)),
        "duplicate_groups_with_differences": len(differing_fields),
        "duplicate_differing_fields": field_groups,
    }


def percent(numerator, denominator):
    return f"{(100 * numerator / denominator):.2f}%" if denominator else "n/a"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_stats = []
    for source_type, path in SOURCES:
        if not path.exists():
            raise FileNotFoundError(path)
        all_stats.append((source_type, path, scan_source(source_type, path)))

    event_seen = set()
    event_counts = Counter()
    duplicate_event_rows = Counter()
    report_stats = []
    registry_path = OUTPUT_DIR / "음식점_정리본.csv"
    events_path = OUTPUT_DIR / "음식점_개폐업사건.csv"

    with registry_path.open("w", encoding="utf-8-sig", newline="") as registry_stream, events_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as event_stream:
        registry_writer = csv.DictWriter(registry_stream, fieldnames=REGISTRY_FIELDS, extrasaction="ignore")
        event_writer = csv.DictWriter(event_stream, fieldnames=EVENT_FIELDS)
        registry_writer.writeheader()
        event_writer.writeheader()

        for source_type, path, stats in all_stats:
            dup_summary = duplicate_group_summary(path, stats["duplicate_ids"])
            with path.open("r", encoding=ENCODING, errors="replace", newline="") as stream:
                reader = csv.DictReader(stream)
                for source_row_number, row in enumerate(reader, start=2):
                    cleaned = clean_row(source_type, source_row_number, row, stats["duplicate_ids"])
                    registry_writer.writerow({key: cleaned.get(key, "") for key in REGISTRY_FIELDS})
                    management_id = cleaned["management_id"]
                    for event_type, event_date, parse_status in (
                        ("개업", cleaned["permit_date"], cleaned["_permit_date_state"]),
                        ("폐업", cleaned["closure_date"], cleaned["_closure_date_state"]),
                    ):
                        if parse_status != "valid" or not event_date:
                            continue
                        identity_for_event = management_id or f"__missing_id_row_{source_row_number}"
                        event_key = (source_type, identity_for_event, event_type, event_date)
                        if event_key in event_seen:
                            duplicate_event_rows[(source_type, event_type)] += 1
                            continue
                        event_seen.add(event_key)
                        event_counts[(source_type, event_type)] += 1
                        event_writer.writerow({
                            "source_type": source_type,
                            "management_id": management_id,
                            "event_type": event_type,
                            "event_date": event_date,
                            "event_month": event_date[:7],
                            "business_type": cleaned["business_type"],
                            "status_in_current_snapshot": cleaned["status_normalized"],
                            "source_row_number": source_row_number,
                            "duplicate_management_id_flag": cleaned["duplicate_management_id_flag"],
                            "record_quality_flags": cleaned["_quality_flags"],
                        })
            stats["duplicate_summary"] = dup_summary
            stats["event_counts"] = {
                "개업": event_counts[(source_type, "개업")],
                "폐업": event_counts[(source_type, "폐업")],
            }
            stats["duplicate_event_rows"] = {
                "개업": duplicate_event_rows[(source_type, "개업")],
                "폐업": duplicate_event_rows[(source_type, "폐업")],
            }
            report_stats.append((source_type, stats))

    general_fields = set(report_stats[0][1]["fields"])
    snack_fields = set(report_stats[1][1]["fields"])
    report_path = OUTPUT_DIR / "전처리_품질보고서.md"
    with report_path.open("w", encoding="utf-8") as report:
        report.write("# 음식점 인허가 데이터 전처리 품질 보고서\n\n")
        report.write("- 실행 스크립트: `preprocess_restaurant_data.py`\n")
        report.write("- 원본 파일은 수정하지 않았습니다. 입력 인코딩은 CP949이며, 깨진 텍스트가 있는 셀은 대체문자와 품질 플래그로 보존했습니다.\n")
        report.write("- 날짜·상태·주소를 정규화했으며, 분석 기간·카페 분류·행정동 매핑은 결정하지 않았습니다.\n")
        report.write("- 개폐업 사건 파일은 월별 집계 전의 사건 단위 자료입니다.\n\n")
        report.write("## 결과 파일\n\n")
        report.write(f"- `{registry_path.name}`: 원본 행 수를 모두 유지한 EDA 핵심 열 정리 테이블. 전체 열은 원본 CSV에 보존되어 있습니다.\n")
        report.write(f"- `{events_path.name}`: 고유 `(업소 유형, 관리번호, 사건 유형, 사건일)` 기준 개업·폐업 사건 테이블\n\n")
        report.write("## 파일별 점검 결과\n\n")
        report.write("| 항목 | 일반음식점 | 휴게음식점 |\n|---|---:|---:|\n")
        g = report_stats[0][1]
        s = report_stats[1][1]
        rows_g, rows_s = g["rows"], s["rows"]
        rows_data = [
            ("원본 행 수", f"{rows_g:,}", f"{rows_s:,}"),
            ("열 수", str(len(g["fields"])), str(len(s["fields"]))),
            ("고유 관리번호", f"{g['unique_management_ids']:,}", f"{s['unique_management_ids']:,}"),
            ("관리번호 결측", f"{g['blank_management_ids']:,} ({percent(g['blank_management_ids'], rows_g)})", f"{s['blank_management_ids']:,} ({percent(s['blank_management_ids'], rows_s)})"),
            ("중복 관리번호 그룹", f"{g['duplicate_id_groups']:,}", f"{s['duplicate_id_groups']:,}"),
            ("인허가일 결측 / 오류", f"{g['permit']['missing']:,} / {g['permit']['invalid']:,}", f"{s['permit']['missing']:,} / {s['permit']['invalid']:,}"),
            ("폐업일 결측 / 오류", f"{g['closure']['missing']:,} / {g['closure']['invalid']:,}", f"{s['closure']['missing']:,} / {s['closure']['invalid']:,}"),
            ("인허가일 범위", f"{g['permit']['min']} ~ {g['permit']['max']}", f"{s['permit']['min']} ~ {s['permit']['max']}"),
            ("폐업일 범위", f"{g['closure']['min']} ~ {g['closure']['max']}", f"{s['closure']['min']} ~ {s['closure']['max']}"),
            ("1960년 이전 인허가일", f"{g['permit']['pre_1960']:,}", f"{s['permit']['pre_1960']:,}"),
            ("폐업 상태 + 폐업일 결측", f"{g['closed_status_missing_closure']:,}", f"{s['closed_status_missing_closure']:,}"),
            ("영업 상태 + 폐업일 존재", f"{g['active_status_with_closure']:,}", f"{s['active_status_with_closure']:,}"),
            ("인허가일보다 빠른 폐업일", f"{g['closure_before_permit']:,}", f"{s['closure_before_permit']:,}"),
            ("업태 결측", f"{g['missing'].get('업태구분명', 0):,} ({percent(g['missing'].get('업태구분명', 0), rows_g)})", f"{s['missing'].get('업태구분명', 0):,} ({percent(s['missing'].get('업태구분명', 0), rows_s)})"),
            ("지번주소 결측", f"{g['missing'].get('지번주소', 0):,} ({percent(g['missing'].get('지번주소', 0), rows_g)})", f"{s['missing'].get('지번주소', 0):,} ({percent(s['missing'].get('지번주소', 0), rows_s)})"),
            ("도로명주소 결측", f"{g['missing'].get('도로명주소', 0):,} ({percent(g['missing'].get('도로명주소', 0), rows_g)})", f"{s['missing'].get('도로명주소', 0):,} ({percent(s['missing'].get('도로명주소', 0), rows_s)})"),
            ("법정동코드 결측", f"{g['missing'].get('법정동코드', 0):,} ({percent(g['missing'].get('법정동코드', 0), rows_g)})" if '법정동코드' in g['fields'] else "열 없음", f"{s['missing'].get('법정동코드', 0):,} ({percent(s['missing'].get('법정동코드', 0), rows_s)})" if '법정동코드' in s['fields'] else "열 없음"),
            ("주소 두 종류 모두 결측", f"{g['address_missing_both']:,}", f"{s['address_missing_both']:,}"),
            ("좌표쌍 완전", f"{g['coordinate_status']['완전']:,} ({percent(g['coordinate_status']['완전'], rows_g)})", f"{s['coordinate_status']['완전']:,} ({percent(s['coordinate_status']['완전'], rows_s)})"),
            ("인코딩 대체문자 포함 행", f"{g['replacement_rows']:,}", f"{s['replacement_rows']:,}"),
            ("데이터갱신일 범위", f"{g['data_update_min']} ~ {g['data_update_max']}", f"{s['data_update_min']} ~ {s['data_update_max']}"),
        ]
        for label, value_g, value_s in rows_data:
            report.write(f"| {label} | {value_g} | {value_s} |\n")
        report.write("\n## 상태·중복·사건 수\n\n")
        for source_type, stats in report_stats:
            report.write(f"### {source_type}\n\n")
            report.write(f"- 영업상태 원문 분포: {dict(stats['statuses'])}\n")
            report.write(f"- 업태 상위 10개: {stats['business_types'].most_common(10)}\n")
            report.write(f"- 개업 사건 행: {stats['event_counts']['개업']:,}; 폐업 사건 행: {stats['event_counts']['폐업']:,}\n")
            report.write(f"- 중복 사건 행으로 제외한 수: {stats['duplicate_event_rows']}\n")
            report.write(f"- 관리번호 중복 그룹의 공백 정리 후 동일 여부: 동일 {stats['duplicate_summary']['duplicate_groups_identical_after_trim']:,}개, 차이 있음 {stats['duplicate_summary']['duplicate_groups_with_differences']:,}개\n")
            report.write(f"- 중복 그룹에서 값이 달랐던 열: {dict(stats['duplicate_summary']['duplicate_differing_fields'])}\n")
            report.write(f"- 인코딩 대체문자가 확인된 열: {dict(stats['replacement_fields'])}\n\n")
        report.write("## 스키마 차이와 남은 결정\n\n")
        report.write(f"- 일반음식점에만 있는 열: {', '.join(sorted(general_fields - snack_fields))}\n")
        report.write(f"- 휴게음식점에만 있는 열: {', '.join(sorted(snack_fields - general_fields)) or '없음'}\n")
        report.write("- 휴게음식점의 323개 중복 관리번호는 사건 테이블에서 사건 키 기준으로 1건만 남겼습니다. 정리 테이블에는 원본 행을 모두 유지하고 중복 플래그를 붙였습니다.\n")
        report.write("- 1960년 이전 인허가일, 폐업일이 없는 폐업 상태 기록, 주소·좌표 결측, 인코딩 대체문자는 삭제하지 않고 플래그로 남겼습니다.\n")
        report.write("- `데이터갱신일자`와 `최종수정일자`는 사건 시점이 아니라 원본 행의 관리 정보로 보존했습니다.\n")
        report.write("- 2026년은 각 파일의 마지막 사건일이 달라 부분 연도입니다. 공통 분석 기간과 월별 완전성은 다음 EDA 단계에서 정해야 합니다.\n")
        report.write("- 좌표계 확인, 행정동 코드/경계 매핑, 카페 업태 분류는 아직 수행하지 않았습니다.\n")

    print(f"정리 테이블: {registry_path}")
    print(f"사건 테이블: {events_path}")
    print(f"품질 보고서: {report_path}")
    print(f"사건 수: {dict(event_counts)}")


if __name__ == "__main__":
    main()
