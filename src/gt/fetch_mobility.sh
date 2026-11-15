#!/bin/bash
# 생활이동 월 파일을 하나씩 받아 줄이고 원본을 지운다. 사용: bash fetch_mobility.sh 작업폴더 출력폴더 [시작YYYYMM] [끝YYYYMM]
set -u
W=$1; OUT=$2; S=${3:-202301}; E=${4:-202610}; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$W" "$OUT"
m=$S
while [ "$m" -le "$E" ]; do
  if [ ! -f "$OUT/mob_$m.parquet" ]; then
    python3 "$HERE/seoul_dl.py" OA-22298 "$W" "in_${m}\.zip" >/dev/null 2>&1
    z="$W/seoul_purpose_admdong1_in_$m.zip"
    if [ -f "$z" ]; then python3 "$HERE/reduce_mobility.py" "$z" "$OUT/mob_$m.parquet" && rm -f "$z"; else echo "FAIL download $m"; fi
  fi
  y=${m:0:4}; mo=${m:4:2}; mo=$((10#$mo + 1)); if [ $mo -gt 12 ]; then mo=1; y=$((y + 1)); fi; m=$(printf "%d%02d" $y $mo)
done
echo "DONE mobility"
