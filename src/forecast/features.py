"""전역 모델 피처 (동 × origin). origin t월 말까지 알 수 있던 값만 쓴다.

그룹
  OWN_*  대상 자신의 시차·이동합 (검색: 비중, 개수: 개업·폐업)
  FNB    feature_spec G1~G3: 영업 점포, 진입·퇴출·순증·회전율(_ma3, EB), 12개월 순증률, 업력, 생존율, 업종 구성
  SRCH   검색 비중, chg_1m, chg_3m, chg_3m_sa(fold 계절 계수), chg_12m, 수준 백분위
  CAL    origin 달, 코로나 더미(t와 t+1..t+h), 구, 동 수준(fold 학습 구간 평균)
"""
import json

import numpy as np
import pandas as pd

from common import C, covid, safe_div
import data as D

DAY = 30.4375


def _roll(x, k, how="sum"):
    r = x.rolling(k, min_periods=k)
    return r.sum() if how == "sum" else r.mean()


def eb_rate(x, n, gu):
    """Empirical Bayes 축소 (feature_spec §0): 같은 달·같은 구 동들로 m, s² 추정. x, n은 넓은 표."""
    raw = pd.DataFrame(safe_div(x.values, n.values), index=x.index, columns=x.columns)
    out = raw.copy()
    for g in gu.unique():
        cols = [c for c in x.columns if gu.get(c) == g]
        X, N, R = x[cols].values, n[cols].values.astype(float), raw[cols].values
        ok = np.isfinite(R) & (N > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            Xs = np.where(ok, X, 0).sum(1)
            Ns = np.where(ok, N, 0).sum(1)
            m = Xs / Ns
            nbar = np.where(ok, N, 0).sum(1) / ok.sum(1)
            var = np.nansum(np.where(ok, N * (R - m[:, None]) ** 2, 0), 1) / Ns
            s2 = np.clip(var - m / nbar, 0, None)
            w = s2[:, None] / (s2[:, None] + m[:, None] / N)
            eb = w * R + (1 - w) * m[:, None]
        out[cols] = np.where(ok, eb, np.nan)
    return out


def stack(name, x, idx):
    x = x.reindex(idx)
    x.columns.name = "series_id"
    return x.stack().rename(name)


class Builder:
    def __init__(self, fold, final=False):
        self.fold, self.final = fold, final
        self.gu = D.gu_of(final)
        self.dongs = list(self.gu.index)
        _, f = D.monthly(final)
        self.f = f[f.level == "dong"]
        self.months = sorted(self.f.month.unique())
        self.cats = json.load(open(C.INTERIM / "fnb_categories.json"))["categories"]
        self.train_end = C.FINAL_FOLD["FINAL"]["train_end"] if fold == "FINAL" else C.FOLDS[fold]["train_end"]

    def w(self, col):
        return self.f.pivot(index="month", columns="series_id", values=col).reindex(self.months)[self.dongs]

    # ---------------- F&B ----------------
    def fnb(self, panel):
        O, Cc, S = self.w("O"), self.w(f"C_{panel}"), self.w("S")
        out = {}
        S3, S12 = S.shift(3), S.shift(12)
        O3, C3 = _roll(O, 3), _roll(Cc, 3)
        out["fnb_logS"] = np.log1p(S)
        out["fnb_S_g12"] = pd.DataFrame(safe_div(S, S12), index=S.index, columns=S.columns) - 1
        out["fnb_entry_ma3"] = pd.DataFrame(safe_div(O3, S3), index=S.index, columns=S.columns)
        out["fnb_exit_ma3"] = pd.DataFrame(safe_div(C3, S3), index=S.index, columns=S.columns)
        out["fnb_entry_eb"] = eb_rate(O3, S3, self.gu)
        out["fnb_exit_eb"] = eb_rate(C3, S3, self.gu)
        out["fnb_net_eb"] = out["fnb_entry_eb"] - out["fnb_exit_eb"]
        out["fnb_churn_eb"] = out["fnb_entry_eb"] + out["fnb_exit_eb"]
        out["fnb_net_12m"] = pd.DataFrame(safe_div(_roll(O, 12) - _roll(Cc, 12), S12), index=S.index, columns=S.columns)
        out["fnb_churn_12m"] = pd.DataFrame(safe_div(_roll(O, 12) + _roll(Cc, 12), S12), index=S.index, columns=S.columns)
        # 업력
        tend = pd.Series([pd.Period(m, "M").end_time.toordinal() for m in self.months], index=self.months)
        mean_ord = pd.DataFrame(safe_div(self.w("age_ord_sum"), S), index=S.index, columns=S.columns)
        out["fnb_age_mean"] = (mean_ord.rsub(tend, axis=0)) / DAY
        Cd12 = _roll(self.w("C_v_drop"), 12)
        ten = pd.DataFrame(safe_div(_roll(self.w("ten_sum"), 12), Cd12), index=S.index, columns=S.columns)
        out["fnb_tenure_closed"] = ten.where(Cd12 >= 5)
        # 6개월 생존: t-12 ~ t-7월 개업 코호트 (t-1월까지 관측 끝)
        cn, cs = _roll(self.w("coh_n"), 6).shift(7), _roll(self.w("coh_surv"), 6).shift(7)
        out["fnb_surv6_eb"] = eb_rate(cs, cn, self.gu)
        # 업종 구성
        out["fnb_trend_share"] = pd.DataFrame(safe_div(self.w("S_trend"), S), index=S.index, columns=S.columns)
        out["fnb_trend_d12"] = out["fnb_trend_share"] - out["fnb_trend_share"].shift(12)
        out["fnb_cafe_share"] = pd.DataFrame(safe_div(self.w("S_cafe"), S), index=S.index, columns=S.columns)
        out["fnb_cafe_entry_ma3"] = pd.DataFrame(safe_div(_roll(self.w("O_cafe"), 3), self.w("S_cafe").shift(3)),
                                                 index=S.index, columns=S.columns)
        P = np.stack([self.w(f"Scat_{k:02d}").values for k in range(len(self.cats))], -1)
        tot = P.sum(-1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(tot > 0, P / tot, np.nan)
            ent = -np.nansum(np.where(p > 0, p * np.log(p), 0), -1) / np.log(len(self.cats))
        ent[tot[..., 0] == 0] = np.nan
        out["fnb_entropy"] = pd.DataFrame(ent, index=S.index, columns=S.columns)
        return out

    # ---------------- 개수 자신 ----------------
    def own_count(self, col, h, pre):
        y = self.w(col)
        out = {f"{pre}_l0": y, f"{pre}_l1": y.shift(1), f"{pre}_l2": y.shift(2)}
        for k in (3, 6, 12):
            out[f"{pre}_ma{k}"] = _roll(y, k, "mean")
        out[f"{pre}_snaive"] = _roll(y, h).shift(12 - h)            # 1년 전 같은 h개월 합
        out[f"{pre}_win"] = _roll(y, h)                               # 최근 h개월 합
        return out

    # ---------------- 검색 ----------------
    def search(self, h):
        sh = D.search_wide("share", final=self.final).reindex(self.months)
        sh = sh.reindex(columns=self.dongs)
        sf = pd.read_parquet(C.INTERIM / "search_seasonal.parquet")
        sf = sf[(sf.fold == self.fold) & (sf.tier == "전체")].pivot(index="moy", columns="dong_code8", values="sf")
        sf = sf.reindex(columns=self.dongs)
        moy = [m[5:] for m in self.months]
        SF = pd.DataFrame(sf.reindex(moy).values, index=self.months, columns=self.dongs)
        sa = sh / SF
        ma3 = _roll(sh, 3, "mean")
        sa3 = _roll(sa, 3, "mean")
        out = {"srch_log_ma3": np.log(ma3),
               "srch_chg_1m": sh / sh.shift(1) - 1,
               "srch_chg_3m": ma3 / ma3.shift(3) - 1,
               "srch_chg_3m_sa": sa3 / sa3.shift(3) - 1,
               "srch_chg_12m": ma3 / ma3.shift(12) - 1,
               "srch_level_pct": ma3.rank(axis=1, pct=True)}
        # 앞으로 h개월 계절 계수 / 최근 3개월 계절 계수 (fold 학습 구간으로 정한 값이라 미래 정보 아님)
        fut = D.h_agg(SF, h, "mean")
        out["srch_sf_ratio"] = fut / _roll(SF, 3, "mean")
        for k, v in out.items():
            out[k] = v.replace([np.inf, -np.inf], np.nan)
        return out, sh, ma3

    def own_search(self, sh, ma3):
        ls = np.log(sh)
        lm = np.log(ma3)
        out = {f"own_s_l{k}": ls.shift(k) - lm for k in (0, 1, 2, 3, 6, 12)}
        out["own_s_ma6"] = np.log(_roll(sh, 6, "mean")) - lm
        out["own_s_ma12"] = np.log(_roll(sh, 12, "mean")) - lm
        for k, v in out.items():
            out[k] = v.replace([np.inf, -np.inf], np.nan)
        return out

    # ---------------- 표 만들기 ----------------
    def table(self, target, h, origins, groups):
        """target: search | open | close_v_drop | close_v_keep | net_rate_v_drop | net_rate_v_keep
        groups: 'base' 또는 'base+cross' (검색 <-> F&B 교차 피처)."""
        cross = groups == "base+cross"
        panel = "v_keep" if target.endswith("v_keep") else "v_drop"
        feats = {}
        srch, sh, ma3 = self.search(h)
        fnb = self.fnb(panel)
        if target == "search":
            feats.update(self.own_search(sh, ma3))
            feats.update(srch)
            if cross:
                feats.update(fnb)
                feats.update(self.own_count("O", h, "x_open"))
                feats.update(self.own_count(f"C_{panel}", h, "x_close"))
        else:
            feats.update(self.own_count("O", h, "own_open"))
            feats.update(self.own_count(f"C_{panel}", h, "own_close"))
            feats.update(fnb)
            if cross:
                feats.update(srch)
        idx = pd.Index(origins, name="month")
        X = pd.concat([stack(k, v[self.dongs], idx) for k, v in feats.items()], axis=1)
        X.index.names = ["origin", "series_id"]
        X = X.reset_index()
        X["cal_month"] = X.origin.str[5:].astype(int)
        X["cal_covid"] = covid(X.origin)
        cv = pd.Series(covid(self.months), index=self.months, dtype=float)
        cfut = cv[::-1].rolling(h, min_periods=1).mean()[::-1].shift(-1).fillna(0)
        X["cal_covid_fut"] = X.origin.map(cfut).values
        X["gu"] = X.series_id.map(self.gu).values

        # 대상 (t+1..t+h). 학습 구간 안 origin만 실제값을 쓴다
        S = self.w("S")
        if target == "search":
            fut = D.h_agg(sh, h, "mean")
            yv = np.log(fut) - np.log(ma3)
            base = ma3
        elif target.startswith("net_rate"):
            fut = D.h_agg(self.w("O") - self.w(f"C_{panel}"), h, "sum")
            yv = pd.DataFrame(safe_div(fut, S), index=S.index, columns=S.columns)
            base = S
        else:
            col = "O" if target == "open" else f"C_{panel}"
            yv = D.h_agg(self.w(col), h, "sum")
            base = S
        X["y"] = stack("y", yv[self.dongs].replace([np.inf, -np.inf], np.nan), idx).values
        X["base"] = stack("base", base[self.dongs], idx).values
        X["S_t"] = stack("S", S[self.dongs], idx).values
        # 동 수준: fold 학습 구간 평균
        tr = slice(C.TRAIN_START, self.train_end)
        if target == "search":
            lvl = np.log(sh.loc[tr].mean())
        elif target.startswith("net_rate"):
            lvl = (self.w("O") - self.w(f"C_{panel}")).loc[tr].sum() / S.loc[tr].mean() / len(sh.loc[tr])
        else:
            col = "O" if target == "open" else f"C_{panel}"
            lvl = np.log1p(self.w(col).loc[tr].mean())
        X["cal_dong_level"] = X.series_id.map(lvl).values
        return X
