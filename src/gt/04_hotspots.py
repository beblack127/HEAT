"""핫스팟과 상위 K개 적중률 (명세 §7).
현재 핫스팟 = 13주 평균 상위 10%
떠오르는 곳 = 13주 평균 서울 대비 비중의 1년 전 같은 13주 대비 증가율 상위 10%, 규모 하위 25% 제외
"""
import glob
import numpy as np, pandas as pd
import config as C
from common import log_run, unit_names

T = pd.read_parquet(C.INTERIM / "target_weekly.parquet")
NM = unit_names()
P = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(C.RESULTS / "pred_*.parquet")))], ignore_index=True)
P["key"] = P.model + "|" + P.xset
yw = T.pivot(index="week_end", columns="unit", values="y")
sh = T.pivot(index="week_end", columns="unit", values="share")


def window(o, ahead):
    o = pd.Timestamp(o)
    w = yw.index
    return w[(w > o) & (w <= o + pd.Timedelta(weeks=C.H))] if ahead else w[(w <= o) & (w > o - pd.Timedelta(weeks=C.H))]


def lists(level, growth, keep):
    n = max(1, int(round(len(level.dropna()) * C.HOT_TOP)))
    hot = level.dropna().sort_values(ascending=False).index[:n]
    g = growth[growth.index.isin(keep)].dropna().sort_values(ascending=False)
    em = g.index[:max(1, int(round(len(g) * C.HOT_TOP)))]
    return hot, em, g


def state(o, kind, key=None, ycol=yw, scol=sh):
    """kind: now(직전 13주 실제) / actual(다음 13주 실제) / pred(다음 13주 예측)"""
    ahead = kind != "now"
    w = window(o, ahead)
    prev = w - pd.Timedelta(weeks=C.SEASON)
    if kind == "pred":
        g = P[(P.origin == pd.Timestamp(o)) & (P.key == key)].pivot(index="week_end", columns="unit", values="y_hat").reindex(w)
        lvl = g.mean()
        s_pred = g.div(g.sum(axis=1), axis=0).mean()
    else:
        lvl = ycol.reindex(w).mean(); s_pred = scol.reindex(w).mean()
    base = scol.reindex(prev).mean()
    return lvl, s_pred / base - 1


def min_level(ycol="y"):
    """규모 하한: 기준 연도 동별 평균의 하위 25% 경계. 시점과 관계없이 고정"""
    b = T[T.week_end.dt.year == C.EMERGING_MIN_BASE_YEAR].groupby("unit")[ycol].mean()
    return float(b.quantile(C.EMERGING_MIN_SIZE_PCT))
THR = min_level()
print(f"규모 하한 ({C.EMERGING_MIN_BASE_YEAR}년 동별 평균 하위 {int(C.EMERGING_MIN_SIZE_PCT * 100)}%): 하루 {THR:,.0f}")
rows_list, topk, sens = [], [], []
for o in C.ORIGINS:
    o = pd.Timestamp(o)
    lvl_now, gr_now = state(o, "now")
    keep = lvl_now[lvl_now >= THR].index
    hot_a, em_a, g_a = lists(*state(o, "actual"), keep)
    lvl_a, _ = state(o, "actual")
    hot_n, em_n, g_n = lists(lvl_now, gr_now, keep)
    for kind, hot, em, lv, gr in (("now", hot_n, em_n, lvl_now, gr_now), ("actual", hot_a, em_a, lvl_a, g_a)):
        for rk, u in enumerate(hot, 1): rows_list.append((o.date(), kind, "hot", rk, u, *NM.get(u, ("", u)), lv.get(u), gr.get(u)))
        for rk, u in enumerate(em, 1): rows_list.append((o.date(), kind, "emerging", rk, u, *NM.get(u, ("", u)), lv.get(u), gr.get(u)))
    N_em = len(g_a); N_hot = lvl_a.notna().sum()
    Ks = list(C.TOPK) + ["10%"]
    def hits(pred_rank, true_set_rank, K):
        k = K if isinstance(K, int) else int(round(len(true_set_rank) * C.HOT_TOP))
        return len(set(pred_rank[:k]) & set(true_set_rank[:k])) / k, k
    true_em = list(g_a.index); true_hot = list(lvl_a.dropna().sort_values(ascending=False).index)
    # 기준: 지속 (직전 13주 증가율 / 수준 그대로)
    pers_em = list(gr_now[gr_now.index.isin(keep)].dropna().sort_values(ascending=False).index)
    pers_hot = list(lvl_now.dropna().sort_values(ascending=False).index)
    for K in Ks:
        h, ke = hits(pers_em, true_em, K); topk.append((o.date(), "emerging", "persistence", K, ke, h))
        h, kh = hits(pers_hot, true_hot, K); topk.append((o.date(), "hot", "persistence", K, kh, h))
        topk.append((o.date(), "emerging", "random", K, ke, ke / N_em)); topk.append((o.date(), "hot", "random", K, kh, kh / N_hot))
    for key in sorted(P[P.origin == o].key.unique()):
        if key.startswith("snaive"):
            for K in Ks: topk.append((o.date(), "emerging", key, K, np.nan, np.nan))
        lp, gp = state(o, "pred", key)
        pe = list(gp[gp.index.isin(keep)].dropna().sort_values(ascending=False).index)
        ph = list(lp.dropna().sort_values(ascending=False).index)
        for K in Ks:
            if not key.startswith("snaive"):
                h, k = hits(pe, true_em, K); topk.append((o.date(), "emerging", key, K, k, h))
            h, k = hits(ph, true_hot, K); topk.append((o.date(), "hot", key, K, k, h))
    # 민감도 (실제 다음 13주 떠오르는 곳)
    base = set(em_a)
    def jac(a, b): return len(a & b) / len(a | b) if a | b else np.nan
    for top in (0.05, 0.10, 0.20):
        n = max(1, int(round(len(g_a) * top))); s = set(g_a.index[:n])
        sens.append((o.date(), f"상위 {int(top * 100)}%", len(s), len(s & base) / min(len(s), len(base))))
    _, g0 = state(o, "actual"); s0 = set(g0.dropna().sort_values(ascending=False).index[:len(base)])
    sens.append((o.date(), "규모 제외 없음", len(s0), jac(s0, base)))
    kq = lvl_now[lvl_now >= lvl_now.quantile(C.EMERGING_MIN_SIZE_PCT)].index   # 이전 방식: 시점마다 하위 25%
    _, eq, _ = lists(*state(o, "actual"), kq); sens.append((o.date(), "규모 하한을 시점마다 다시 계산", len(eq), jac(set(eq), base)))
    for lab, yc, sc in (("목적 4+5", "y_p45", "share_p45"), ("전 연령", "y_allage", "share_allage")):
        Y2 = T.pivot(index="week_end", columns="unit", values=yc); S2 = T.pivot(index="week_end", columns="unit", values=sc)
        l2, g2 = state(o, "actual", ycol=Y2, scol=S2)
        ln2, _ = state(o, "now", ycol=Y2, scol=S2); k2 = ln2[ln2 >= min_level(yc)].index
        _, e2, _ = lists(l2, g2, k2); sens.append((o.date(), lab, len(e2), jac(set(e2), base)))

L = pd.DataFrame(rows_list, columns=["origin", "window", "list", "rank", "unit", "gu", "dong", "level_13w", "growth_vs_last_year"])
L.round(4).to_csv(C.TABLES / "hotspots.csv", index=False, encoding="utf-8-sig")
K = pd.DataFrame(topk, columns=["origin", "target", "method", "K", "k", "hit_rate"])
K.round(4).to_csv(C.TABLES / "topk.csv", index=False, encoding="utf-8-sig")
S = pd.DataFrame(sens, columns=["origin", "variant", "n", "overlap_with_base"])
S.round(3).to_csv(C.TABLES / "sensitivity.csv", index=False, encoding="utf-8-sig")
pd.Series({"min_level": THR, "base_year": C.EMERGING_MIN_BASE_YEAR, "pct": C.EMERGING_MIN_SIZE_PCT}).to_csv(C.TABLES / "emerging_min_level.csv", header=["value"], encoding="utf-8-sig")
log_run("04_hotspots", f"origins={len(C.ORIGINS)}")
print(K[(K.K == 20)].groupby(["target", "method"]).hit_rate.mean().round(3).sort_values(ascending=False).head(20))
print(S.groupby("variant").overlap_with_base.mean().round(3))
