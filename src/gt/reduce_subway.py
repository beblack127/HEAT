"""지하철 일별 승하차(OA-12914) 월 파일 -> 역 x 일 하차 표.
사용: python3 reduce_subway.py 원본폴더 출력.parquet
- 같은 역 이름의 여러 호선은 더한다
- 역 이름은 괄호 안 부역명과 끝의 '역'을 뗀다
"""
import sys, re, glob
import pandas as pd

def norm(s):
    s = re.sub(r"\(.*?\)", "", str(s)).strip()
    return s[:-1] if s.endswith("역") and len(s) > 2 else s

src, out = sys.argv[1], sys.argv[2]
parts = []
for fp in sorted(glob.glob(f"{src}/CARD_SUBWAY_MONTH_*.csv")):
    for enc in ("utf-8-sig", "cp949"):
        try:
            d = pd.read_csv(fp, encoding=enc, dtype=str, index_col=False); break
        except UnicodeDecodeError:
            continue
    d.columns = [c.strip().strip('"') for c in d.columns]
    d = d.rename(columns={"사용일자": "date", "노선명": "line", "역명": "station_raw", "승차총승객수": "board", "하차총승객수": "alight"})
    d["alight"] = pd.to_numeric(d["alight"], errors="coerce")
    d["board"] = pd.to_numeric(d["board"], errors="coerce")
    parts.append(d[["date", "line", "station_raw", "board", "alight"]])
d = pd.concat(parts, ignore_index=True)
d["date"] = pd.to_datetime(d["date"], format="%Y%m%d")
d["station"] = d["station_raw"].map(norm)
g = d.groupby(["station", "date"], as_index=False)[["board", "alight"]].sum()
lines = d.groupby("station")["line"].agg(lambda s: ",".join(sorted(set(s)))).rename("lines")
g = g.merge(lines, on="station", how="left")
g.to_parquet(out, index=False)
print(f"역 {g.station.nunique()}개, {g.date.min().date()} ~ {g.date.max().date()}, 행 {len(g)}")
