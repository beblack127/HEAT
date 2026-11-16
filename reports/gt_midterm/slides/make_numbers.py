"""슬라이드에 들어갈 숫자를 결과 표에서 모아 deck_numbers.json으로 저장 (슬라이드 숫자 = 결과 표 숫자)"""
import json, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]; TB = ROOT / "reports/gt_midterm/tables"; RES = ROOT / "results/gt"
sys.path.insert(0, str(ROOT / "src/gt"))
import config as C

N = {}
S = pd.read_csv(TB / "sensitivity.csv"); sm = S.groupby("variant").overlap_with_base.mean()
N["sens_floor"] = round(sm["규모 하한을 시점마다 다시 계산"] * 100); N["sens_age"] = round(sm["전 연령"] * 100); N["sens_nofloor"] = round(sm["규모 제외 없음"] * 100)
v3 = pd.read_csv(TB / "v3_sales.csv")
r = v3[(v3["비교"].str.startswith("수준")) & (v3["자료"] == "음식업 매출 건수")].iloc[0]
N["sales_level"] = f"{r['스피어만 상관']:.2f}"; N["sales_top"] = round(r["상위 10% 겹침"] * 100)
T = pd.read_parquet(C.INTERIM / "target_weekly.parquet"); N["n_weeks"] = int(T.week_end.nunique())

t7 = pd.read_csv(TB / "t7_more_models.csv", index_col=0)
REP = "결합: 작년 같은 주 (보정) + ARIMA + 푸리에 (보정)"
c4, c2 = "4시점 평균", "2시점 평균(이상 구간 제외)"
N["best4"] = f"{t7.loc[REP, c4]:.1f}"; N["best2"] = f"{t7.loc[REP, c2]:.1f}"
N["sn4"] = f"{t7.loc['작년 같은 주', c4]:.1f}"; N["sn2"] = f"{t7.loc['작년 같은 주', c2]:.1f}"; N["snc4"] = f"{t7.loc['작년 같은 주 (보정)', c4]:.1f}"
tt = pd.read_csv(TB / "t7_more_models_tests.csv")
N["best_imp"] = round(tt[(tt.model == REP) & (tt.vs == "작년 같은 주")].improved_share.iloc[0] * 100)

K = pd.read_csv(TB / "topk.csv"); k = K[(K.K.astype(str) == "10%") & (K.target == "emerging")].groupby("method").hit_rate.mean()
N["topk_pers"] = round(k["persistence"] * 100); N["topk_model"] = round(k.get("comb_fourier|none", np.nan) * 100) if "comb_fourier|none" in k else round(k.drop(["persistence", "random"]).max() * 100)

# 동네 예시 (가락1동)
EX, O = "11710631", pd.Timestamp("2025-04-27")
H = pd.read_csv(TB / "hotspots.csv", dtype={"unit": str})
h = H[(H.origin == "2025-04-27") & (H.window == "now") & (H.list == "emerging")]
N["ex_rank"] = int(h[h.unit == EX]["rank"].iloc[0]); N["ex_n"] = int(len(h)); N["ex_growth"] = round(h[h.unit == EX].growth_vs_last_year.iloc[0] * 100)
P = pd.concat([pd.read_parquet(f) for f in glob.glob(str(RES / "pred_comb_*.parquet")) + glob.glob(str(RES / "pred_more_*.parquet"))])
p = P[(P.unit == EX) & (P.origin == O)].pivot_table(index="week_end", columns="model", values="y_hat")
y = T[T.unit == EX].set_index("week_end").y.reindex(p.index)
N["ex_wape"] = f"{(y - p.comb_fourier).abs().sum() / y.sum() * 100:.1f}"; N["ex_sn"] = f"{(y - p.snaive_c).abs().sum() / y.sum() * 100:.1f}"
fr = []
for f in glob.glob(str(ROOT / "data/raw/sales/*.csv")):
    fr.append(pd.read_csv(f, encoding="cp949", usecols=["기준_년분기_코드", "행정동_코드", "서비스_업종_코드", "당월_매출_건수"], dtype={"행정동_코드": str}))
Sx = pd.concat(fr); Sx = Sx[Sx["서비스_업종_코드"].str.startswith("CS1")]
q = Sx.groupby(["행정동_코드", "기준_년분기_코드"])["당월_매출_건수"].sum().unstack(); sh = q / q.sum()
N["ex_sales"] = round((sh.loc[EX, [20251, 20252]].mean() / sh.loc[EX, [20241, 20242]].mean() - 1) * 100, 1)

# 부록 표
rows = [("Last week (naive)", "Naive"), ("Same week last year", "작년 같은 주"), ("ARIMA", "ARIMA"), ("SARIMA (52 weeks)", "SARIMA"),
        ("Same week last year · cleaned", "작년 같은 주 (보정)"), ("ARIMA · cleaned", "ARIMA (보정)"), ("ARIMA + Fourier · cleaned", "ARIMA + 푸리에 (보정)"),
        ("STL + ETS · cleaned", "STL + ETS (보정)"), ("Theta · cleaned", "Theta (보정)"), ("Combination (last year + ARIMA·Fourier)", REP)]
cols = [c for c in t7.columns if c[:2] == "20"]
N["origins"] = [c[2:7].replace("-", ".") for c in cols]
N["table"] = [[lab] + [f"{t7.loc[k, c]:.1f}" for c in cols] + [f"{t7.loc[k, c4]:.1f}", f"{t7.loc[k, c2]:.1f}"] for lab, k in rows]
# 사다리 (전체 조합 실험)
LD = pd.read_csv(TB / "t8_ladder.csv")
N["lad"] = {c: [f"{v:.1f}" for v in LD[c]] for c in LD.columns if c != "단계"}
# 가짜 입력 (팀원 설정, 13주 주기)
if (TB / "t10_placebo.csv").exists():
    W10 = pd.read_csv(TB / "t10_placebo.csv", index_col=[0, 1])
    N["pl"] = {k: {c: f"{W10.loc[('s13', k), c]:.1f}" for c in ("2025-04-27", "4시점")} for k in ("none", "search_index", "placebo", "subway_exit_4w", "fnb_net3")}
N["ratio"] = {p.name: (lambda im: im.size[0] / im.size[1])(Image.open(p)) for p in (HERE / "fig").glob("*.png")}
json.dump(N, open(HERE / "deck_numbers.json", "w"), ensure_ascii=False, indent=1)
print({k: v for k, v in N.items() if k not in ("table", "ratio")})
