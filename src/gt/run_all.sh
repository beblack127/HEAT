#!/bin/bash
# 중간발표 실험 전체 다시 돌리기 (약 45분, 02_models가 대부분)
# 필요: data/raw/mobility/mob_YYYYMM.parquet, data/raw/living_pop/lp_YYYYMM.parquet, data/raw/subway/subway_daily.parquet
# 원자료에서 다시 만들려면: fetch_mobility.sh, fetch_living_pop.sh, seoul_dl.py + reduce_subway.py
set -e
cd "$(dirname "$0")"
for s in 00_target 01_features 02_models 03_evaluate 04_hotspots 05_figures 06_report 07_map_data; do
  echo "== $s"; python3 "$s.py"
done
cd ../.. && bash src/search/run.sh
