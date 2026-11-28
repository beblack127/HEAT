"""Render a presentation-ready WAPE comparison from the saved JSON summaries."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
family = json.loads((HERE / "arima_family_summary.json").read_text(encoding="utf-8"))
baseline = json.loads((HERE / "pilot_summary.json").read_text(encoding="utf-8"))
font = "Malgun Gothic" if any(f.name == "Malgun Gothic" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams["font.family"] = font
plt.rcParams["axes.unicode_minus"] = False

names = ["2개월 전 값", "ARIMA", "SARIMA", "ARIMAX", "SARIMAX"]
colors = ["#8492a6", "#2176ae", "#4c9f70", "#d98932", "#b15d8c"]
periods = [
    ("2025년 전체 · 480개 동·월", "all_2025", "evaluation_2025"),
    ("2025년 1–8월 · 320개 동·월", "jan_aug", "evaluation_2025_jan_aug"),
]
fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=150, sharey=True)
fig.patch.set_facecolor("#f8fafc")
for ax, (title, key, base_key) in zip(axes, periods):
    vals = [baseline[base_key]["lag2_daily"]["wape"] * 100]
    vals += [family["models"][key][n]["wape"] * 100 for n in names[1:]]
    bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1], height=0.62)
    ax.set_xlim(0, 14)
    ax.set_title(title, fontsize=16, weight="bold", pad=15)
    ax.set_xlabel("WAPE (%) · 낮을수록 좋음", fontsize=12)
    ax.set_facecolor("#f8fafc")
    ax.grid(axis="x", color="#dbe3eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, vals[::-1]):
        ax.text(v + 0.15, bar.get_y() + bar.get_height()/2, f"{v:.2f}%",
                ha="left", va="center", fontsize=11, weight="bold", color="#17202a")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, labelsize=11)
fig.suptitle("행정동 생활이동 예측: 배운 모델 4종 비교", fontsize=22, weight="bold", y=0.98)
fig.text(0.5, 0.055,
         "목표: 목적 4+5+7 일평균 도착 이동량  |  매월 1일 기준, m-2까지 사용  |  외생변수: 2개월 전 주간 생활인구",
         ha="center", fontsize=10, color="#425466")
fig.text(0.5, 0.025, "네이버 검색량은 포함하지 않음 · 2025년 9월 이후 원천 품질은 별도 점검 필요",
         ha="center", fontsize=10, color="#8b4351")
fig.subplots_adjust(top=0.82, bottom=0.16, left=0.15, right=0.97, wspace=0.26)
fig.savefig(HERE / "arima_family_wape.png", facecolor=fig.get_facecolor())
