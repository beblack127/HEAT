#!/bin/bash
# 이름표: 연결표 01~07 -> 지도 데이터 -> 이름표 확인 지도. 저장소 루트에서 실행: bash src/nametags/run.sh
# 필요: data/interim/음식점_정리본.csv (git 제외, 팀 폴더에서 복사), data/external/geo/*
# 한강 폴리곤(han_river_5179.parquet)은 river.py 로 만든 것을 그대로 쓴다 (원본: OSM, ODbL)
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd); SRC=$ROOT/src/nametags; G=$ROOT/data/external/geo; W=$ROOT/build/nametags
mkdir -p "$W/in" "$W/out"
for f in emd_20251231.parquet han_river_5179.parquet umd.shp umd.shx umd.dbf umd.prj sk.shp sk.shx sk.dbf sk.prj sk.cpg stations.csv; do
  ln -sf "$G/$f" "$W/in/$f"; done
ln -sf "$ROOT/data/interim/음식점_정리본.csv" "$W/in/stores.csv"
cp "$SRC/mapgeo.py" "$W/"
python3 "$SRC/build_template.py" "$W/template_v4.html"
cd "$W"
python3 "$SRC/build_nametags.py" in out
python3 "$SRC/make_html.py" template_v4.html "$G/landmarks.json"
echo "결과: $W/out  (확인 후 data/processed/nametags, maps 로 옮긴다)"
