#!/bin/bash
# 검색량: (수집) -> 동별 지수 -> 지도. 저장소 루트에서 실행: bash src/search/run.sh
# 이미 받은 API 응답(data/raw/search_api)은 다시 요청하지 않는다. 새로 받을 요청이 있을 때만 .env 가 필요하다.
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd); S=$ROOT/src/search; P=$ROOT/data/processed; W=$ROOT/build/search
mkdir -p "$W"; ln -sfn "$ROOT/data/raw/search_api" "$W/raw"
python3 "$S/collect_datalab.py" --groups "$P/nametags/02_검색어_묶음.csv" --env "$ROOT/.env" --out "$W" --anchor-root 문래
python3 "$S/compute_dong_search.py" --series "$W/검색량_묶음별_월별.csv" --links "$P/nametags/04_행정동_이름표_연결표.csv" --out "$W" \
  --nonseoul "$S/서울밖_시군구_뿌리.txt" --stoplist "$S/일반어_타지명_뿌리.txt"
python3 "$ROOT/src/nametags/build_template.py" "$W/template_v4.html"
python3 "$S/make_search_html.py" --mapdata "$P/nametags/_mapdata_small.json" --template "$W/template_v4.html" \
  --dong "$W/행정동_검색지수_월별.csv" --groups "$W/검색량_묶음별_월별.csv" --quality "$W/검색어묶음_품질.csv" \
  --landmarks "$ROOT/data/external/geo/landmarks.json" --anchor 문래 --out "$W/행정동_이름표_검색량_지도.html"
echo "결과: $W  (확인 후 data/processed/search, maps 로 옮긴다)"
