"""생활이동 (OA-22298, 도착지 기준·성연령별·내국인) 월 압축 파일 -> 서울 도착 동 x 일 x 목적 표.
사용: python3 reduce_mobility.py 압축파일.zip 출력.parquet

- 서울 도착(행정동 코드 11로 시작)만 남긴다
- 시간대(time_cd: 시간 또는 20분 구간)는 서로 겹치지 않으므로 모두 더해 하루 값으로 만든다
- 열: dong_code8, date, purpose(1~7), y2030(20대+30대 남녀), yall(전 연령)
"""
import sys, io, zipfile
import pandas as pd

zp, out = sys.argv[1], sys.argv[2]
A2030 = ["male_20_cnt", "male_30_cnt", "feml_20_cnt", "feml_30_cnt"]
parts = []
with zipfile.ZipFile(zp) as z:
    for name in sorted(n for n in z.namelist() if n.endswith(".csv")):
        d = pd.read_csv(io.BytesIO(z.read(name)), dtype={"d_admdong_cd": str, "time_cd": str, "etl_ymd": str})
        d = d[d.d_admdong_cd.str.startswith("11")]
        d["y2030"] = d[A2030].sum(axis=1)
        g = d.groupby(["d_admdong_cd", "etl_ymd", "move_purpose"], as_index=False)[["y2030", "total_cnt"]].sum()
        parts.append(g)
g = pd.concat(parts, ignore_index=True).rename(columns={"d_admdong_cd": "dong_code8", "move_purpose": "purpose", "total_cnt": "yall"})
g["date"] = pd.to_datetime(g.pop("etl_ymd"), format="%Y%m%d")
g["purpose"] = g.purpose.astype("int8")
g[["y2030", "yall"]] = g[["y2030", "yall"]].astype("float32")
g.to_parquet(out, index=False)
print(f"{zp.split('/')[-1]}: 동 {g.dong_code8.nunique()}, 일 {g.date.nunique()}, 행 {len(g)}")
