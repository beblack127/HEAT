"""행정동 생활인구 (OA-14991) 월 압축 파일 -> 동 x 일 주간(11~21시) 평균 표.
사용: python3 reduce_living_pop.py 압축파일.zip 출력.parquet
열: dong_code8, date, lp_all(총생활인구), lp_2039(20~39세 남녀)
"""
import sys, io, zipfile
import pandas as pd

zp, out = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zp) as z:
    name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
    raw = z.read(name)
for enc in ("utf-8-sig", "cp949"):
    try:
        d = pd.read_csv(io.BytesIO(raw), encoding=enc, dtype=str, index_col=False); break
    except UnicodeDecodeError:
        continue
d.columns = [c.strip().strip('"').lstrip("﻿") for c in d.columns]
date_c, hour_c, dong_c, tot_c = d.columns[:4]
a2039 = [c for c in d.columns if any(f"{a}세부터{a + 4}세" in c for a in (20, 25, 30, 35))]
assert len(a2039) == 8, a2039
num = [tot_c] + a2039
d[num] = d[num].apply(pd.to_numeric, errors="coerce")
d["hour"] = pd.to_numeric(d[hour_c], errors="coerce")
d = d[(d.hour >= 11) & (d.hour <= 21)]
d["lp_2039"] = d[a2039].sum(axis=1)
g = d.groupby([dong_c, date_c], as_index=False)[[tot_c, "lp_2039"]].mean()
g.columns = ["dong_code8", "date", "lp_all", "lp_2039"]
g["date"] = pd.to_datetime(g.date, format="%Y%m%d")
g[["lp_all", "lp_2039"]] = g[["lp_all", "lp_2039"]].astype("float32")
g.to_parquet(out, index=False)
print(f"{zp.split('/')[-1]}: 동 {g.dong_code8.nunique()}, 일 {g.date.nunique()}")
