"""Audit 31 Naver place names against Seoul's official place polygons and dongs.

Outputs are research crosswalks. A polygon intersecting an administrative dong
does not make the whole dong a measurement of that place.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tmp/pythonlibs"))
import shapefile  # type: ignore
from pyproj import Transformer  # type: ignore
from shapely.geometry import shape  # type: ignore
from shapely.ops import transform, unary_union  # type: ignore
from collect_naver_areas import AREAS

OUT = ROOT / "data/place_dong_crosswalk"
SOURCE_URL = "https://data.seoul.go.kr/dataList/OA-21285/F/1/datasetView.do"
BOUNDARY_URL = "https://data.seoul.go.kr/opendata/seoulStay/grid_viewer.html"
PLACE_NAMES = [
    "홍대", "성수", "연남", "합정", "망원", "상수", "을지로", "익선동",
    "서촌", "북촌", "삼청동", "신사", "압구정", "청담", "한남", "이태원",
    "해방촌", "문래", "여의도", "강남", "잠실", "건대", "혜화", "샤로수길",
    "노원", "신촌", "왕십리", "가로수길", "경리단길", "방이", "불광",
]

# These are *candidate* definitions. Aliases and spatial scope require review.
MATCH = {
    "홍대": ("POI007", "broad_place_alias"),
    "성수": ("POI068", "narrower_cafe_street_proxy"),
    "연남": ("POI073", "name_alias"),
    "합정": ("POI053", "station_area_proxy"),
    "익선동": ("POI116", "exact_name"),
    "서촌": ("POI067", "exact_name"),
    "북촌": ("POI066", "hanok_village_proxy"),
    "압구정": ("POI071", "narrower_rodeo_proxy"),
    "청담": ("POI080", "narrower_luxury_street_proxy"),
    "이태원": ("POI004", "broad_tourism_zone_proxy"),
    "해방촌": ("POI082", "shared_with_gyeongridan"),
    "여의도": ("POI072", "exact_name"),
    "강남": ("POI014", "gangnam_station_proxy"),
    "잠실": ("POI005", "broad_tourism_zone_proxy"),
    "건대": ("POI015", "geondae_station_proxy"),
    "혜화": ("POI054", "hyehwa_station_proxy"),
    "샤로수길": ("POI031", "seoul_national_univ_station_proxy"),
    "신촌": ("POI040", "combined_sinchon_ewha_proxy"),
    "왕십리": ("POI045", "wangsimni_station_proxy"),
    "가로수길": ("POI059", "exact_name"),
    "경리단길": ("POI082", "shared_with_haebangchon"),
}


def write_csv(path: Path, records: list[dict]):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    html = (ROOT / "data/reference/seoul_grid_viewer_202506.html").read_text(encoding="utf-8")
    def parse_js_json(name):
        line = next(line for line in html.splitlines() if line.startswith(f"const {name}="))
        return json.loads(line.split("=", 1)[1].rstrip(";"))
    districts = parse_js_json("SIGUNGU")["features"]
    dongs = parse_js_json("DONG")["features"]
    project = Transformer.from_crs(4326, 5181, always_xy=True).transform
    districts = [(f["properties"]["name"], transform(project, shape(f["geometry"]))) for f in districts]
    def normalize_dong_name(value: str) -> str:
        return re.sub(r"[.·\s]", "", value)

    official = {}
    with zipfile.ZipFile(ROOT / "data/reference/seoul_admin_dong_codes_official.zip") as archive:
        with archive.open("SEOUL_ADMI/ADMI_202506.csv") as handle:
            rows = csv.DictReader(line.decode("utf-8-sig") for line in handle)
            for row in rows:
                if row["SIDO_NM"] == "서울특별시":
                    key = (row["SGG_NM"], normalize_dong_name(row["ADMI_NM"]))
                    if key in official and official[key] != row["ADMI_CD"]:
                        raise ValueError(f"Ambiguous official dong after normalization: {key}")
                    official[key] = row["ADMI_CD"]
    dong_shapes = []
    for f in dongs:
        geom = transform(project, shape(f["geometry"]))
        centroid = geom.representative_point()
        district = next((name for name, poly in districts if poly.covers(centroid)), "")
        name = f["properties"]["name"]
        code = official.get((district, normalize_dong_name(name)), "")
        dong_shapes.append((district, name, code, geom))

    reader = shapefile.Reader(str(next((ROOT / "data/reference/seoul_121_place_boundaries").rglob("*.shp"))), encoding="utf-8")
    places = {}
    for item in reader.iterShapeRecords():
        rec = item.record.as_dict()
        places[rec["AREA_CD"]] = (rec["AREA_NM"], transform(project, shape(item.shape.__geo_interface__)))
    assert len(places) == 121
    assert len(PLACE_NAMES) == 31
    assert PLACE_NAMES == [name for name, _term in AREAS]
    assert len(dong_shapes) >= 420
    assert all(code in places for code, _ in MATCH.values())

    area_records = []
    bridge_records = []
    for place in PLACE_NAMES:
        match = MATCH.get(place)
        if not match:
            area_records.append({
                "search_area": place, "official_place_code": "", "official_place_name": "",
                "name_match_type": "no_defensible_official_match", "intersected_dongs": 0,
                "selected_dongs": 0, "largest_place_share": "", "whole_dong_area_to_place_area": "",
                "training_status": "do_not_join", "review_note": "Define a custom boundary or find another official place polygon.",
            })
            continue
        code, kind = match
        official_name, geom = places[code]
        overlaps = []
        for district, dong_name, dong_code, dong_geom in dong_shapes:
            if not geom.intersects(dong_geom):
                continue
            intersection = geom.intersection(dong_geom).area
            if intersection <= 0:
                continue
            overlaps.append((district, dong_name, dong_code, dong_geom, intersection,
                             intersection / geom.area, intersection / dong_geom.area))
        overlaps.sort(key=lambda row: row[4], reverse=True)
        # Keep dongs contributing >=5% of the official place, plus the largest.
        selected = [row for i, row in enumerate(overlaps) if i == 0 or row[5] >= .05]
        whole_to_place = unary_union([row[3] for row in selected]).area / geom.area if selected else None
        for rank, (district, dong_name, dong_code, _geom, area_m2, place_share, dong_share) in enumerate(overlaps, 1):
            bridge_records.append({
                "search_area": place, "official_place_code": code,
                "official_place_name": official_name, "district": district,
                "admin_dong_name": dong_name, "admin_dong_code_202506": dong_code,
                "overlap_rank": rank, "place_area_share": round(place_share, 5),
                "dong_area_share": round(dong_share, 5),
                "selected_by_5pct_rule": int(rank == 1 or place_share >= .05),
                "intersection_m2": round(area_m2, 1),
            })
        area_records.append({
            "search_area": place, "official_place_code": code,
            "official_place_name": official_name, "name_match_type": kind,
            "intersected_dongs": len(overlaps), "selected_dongs": len(selected),
            "largest_place_share": round(overlaps[0][5], 4) if overlaps else "",
            "whole_dong_area_to_place_area": round(whole_to_place, 2) if whole_to_place else "",
            "training_status": "needs_name_and_scale_review",
            "review_note": "Official 2026 place polygon; historical boundary stability unverified. Whole-dong counts overcover the polygon.",
        })

    code_counts = Counter(record["official_place_code"] for record in area_records if record["official_place_code"])
    for record in area_records:
        if record["official_place_code"] and code_counts[record["official_place_code"]] > 1:
            record["training_status"] = "exclude_shared_polygon"
            record["review_note"] = "The same official polygon is mapped to another search area; cannot train as independent observations."
    write_csv(OUT / "place_31_mapping_audit.csv", area_records)
    write_csv(OUT / "place_to_admin_dong_overlap.csv", bridge_records)
    (OUT / "README.md").write_text(
        "# 31개 검색 장소 공간 매핑 초안\n\n"
        f"- 공식 장소 경계: {SOURCE_URL} (2026-04-02 게시된 '서울시 주요 121장소 영역.zip')\n"
        f"- 행정동 경계: {BOUNDARY_URL} (2025-06-30 기준; 웹페이지 내 GeoJSON)\n"
        "- 행정동 코드: 서울시 수도권 생활이동 페이지의 전국 행정동 코드정보, 2025-06 파일.\n"
        "- 면적비 계산: 두 경계를 EPSG:5181로 투영 후 교집합. `place_area_share`는 공식 장소 중 해당 동에 걸친 비율, "
        "`dong_area_share`는 그 동 중 공식 장소에 들어가는 비율이다.\n"
        "- `selected_by_5pct_rule`은 공식 장소 면적의 5% 이상과 최대 겹침 동을 표시하는 탐색 규칙이다. "
        "**해당 동의 전체 방문량을 장소 방문량으로 간주해선 안 된다.**\n"
        "- 서울시 주요 121장소 경계는 2026년 게시자료다. 2023~2025년 학습기간의 경계로 고정 사용하려면 "
        "당시에도 같은 장소 범위였는지 확인해야 한다. 실시간 API에는 과거값이 없다.\n"
        "- 이름이 비슷하다는 이유로 붙인 proxy는 사람이 의미와 범위를 검토해야 한다. "
        "`exclude_shared_polygon`은 서로 독립된 학습지역으로 쓰면 안 된다.\n",
        encoding="utf-8",
    )
    print("31 places", len(area_records), "official polygon candidates", sum(bool(r["official_place_code"]) for r in area_records))
    print("no polygon", sum(r["training_status"] == "do_not_join" for r in area_records))
    print("shared polygon", sum(r["training_status"] == "exclude_shared_polygon" for r in area_records))
    print("official-code gaps in overlap rows", sum(not r["admin_dong_code_202506"] for r in bridge_records))
    print(OUT)


if __name__ == "__main__":
    main()
