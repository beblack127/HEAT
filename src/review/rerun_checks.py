"""Re-run an experiment script with (a) data cut at 2025-08 and/or (b) S4 correction done on days.
usage: python rerun.py <heat_dir> <script> <codex_root> <out_dir> <mode: cut|daily|cut+daily>"""
import os, sys, runpy
from datetime import date, timedelta
import numpy as np
HEAT, SCRIPT, ROOT, OUT, MODE = sys.argv[1:6]
sys.path.insert(0, os.path.join(HEAT, "src", "gt_refine"))
import panel as P
LAG_D = 364
DAY_WIN = [(date(2024, 8, 1), date(2024, 10, 31)), (date(2025, 1, 2), date(2025, 1, 24)), (date(2025, 3, 1), date(2025, 4, 30))]
orig_load = P.load_weekly_extended
CUT = date(2025, 8, 25)

def daily_corrected_weekly(root):
    dates, dongs, Y, _ = P.load_daily(root)
    idx = {d: i for i, d in enumerate(dates)}
    win = np.zeros(len(dates), bool)
    for a, b in DAY_WIN:
        win |= np.array([a <= d <= b for d in dates])
    C = Y.copy()
    for a, b in DAY_WIN:
        rows = np.array([i for i, d in enumerate(dates) if a <= d <= b])
        pre = [i for i in range(rows[0] - 1, -1, -1) if not win[i] and i - LAG_D >= 0][:56]
        lagi = lambda ii: [idx[dates[i] - timedelta(days=LAG_D)] for i in ii]
        r_d = C[pre].sum(0) / C[lagi(pre)].sum(0)
        r_s = C[pre].sum() / C[lagi(pre)].sum()
        f_d = C[rows].sum(0) / (C[lagi(rows)] * r_d).sum(0)
        f_s = C[rows].sum() / (C[lagi(rows)].sum() * r_s)
        C[rows] /= np.exp(0.5 * np.log(f_d) + 0.5 * np.log(f_s))
    weeks, Wc = P.weekly(dates, C)
    return weeks, dongs, Wc

state = {}
def load(root):
    weeks, dongs, W = orig_load(root)
    if "daily" in MODE:
        wk, dg, Wc = daily_corrected_weekly(root)
        j = [dg.index(c) for c in dongs]
        C = W.copy(); C[:len(wk)] = Wc[:, j]
        state["C"] = C
    if "cut" in MODE:
        k = sum(w <= CUT for w in weeks)
        weeks, W = weeks[:k], W[:k]
        if "C" in state: state["C"] = state["C"][:k]
    return weeks, dongs, W

orig_s4 = P.correct_s4
P.load_weekly_extended = load
if "daily" in MODE:
    P.correct_s4 = lambda weeks, W: state["C"].copy()
sys.argv = [SCRIPT, ROOT, OUT]
runpy.run_path(SCRIPT, run_name="__main__")
