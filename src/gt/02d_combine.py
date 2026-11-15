"""결합 모델 예측 저장: 작년 같은 주(보정) + ARIMA + 푸리에(보정)의 1:1 평균 -> results/gt/pred_comb_<origin>.parquet
가중치는 1/2로 고정 (결과를 보고 조정하지 않음). 비교용으로 작년 같은 주(보정) + ARIMA(보정) 결합도 저장
"""
import glob
import pandas as pd
import config as C

P = pd.concat([pd.read_parquet(f) for f in glob.glob(str(C.RESULTS / "pred_more_*.parquet"))], ignore_index=True)
W = P.pivot_table(index=["origin", "unit", "week_end", "h"], columns="model", values="y_hat").reset_index()
out = []
for name, b in (("comb_fourier", "arima_fourier"), ("comb_arima", "arima_c")):
    d = W[["origin", "unit", "week_end", "h"]].copy(); d["model"] = name; d["xset"] = "none"; d["y_hat"] = (W.snaive_c + W[b]) / 2; d["fallback"] = 0
    out.append(d.dropna(subset=["y_hat"]))
O = pd.concat(out)
for o, g in O.groupby("origin"): g.to_parquet(C.RESULTS / f"pred_comb_{pd.Timestamp(o).date()}.parquet", index=False)
print(O.groupby("model").size())
