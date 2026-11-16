"""작업 흐름 그림 (영어, 슬라이드 디자인). 출력: reports/gt_midterm/slides/fig/S11_pipeline.png"""
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
src = open("11_slide_figs.py", encoding="utf-8").read(); exec(compile(src[:src.index("# ---- S1")], "h", "exec"))
f, ax = plt.subplots(figsize=(20, 8.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 43); ax.set_axis_off()
def box(x, y, w, h, title, lines, ec=NAVY, fc="white", tc=NAVY, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=1.8, ls=ls))
    ax.text(x + .9, y + h - 1.0, title, fontsize=19, fontweight="bold", color=tc, va="top")
    ax.text(x + .9, y + h - 4.1, "\n".join(lines), fontsize=14.5, color=INK, va="top", linespacing=1.35)
def arr(x0, y0, x1, y1, ls="-", c=NAVY):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=22, lw=1.8, color=c, ls=ls, shrinkA=2, shrinkB=2))
# 두 줄: 위 = 정답(생활이동), 아래 = 입력
ax.text(0.5, 41.5, "Target", fontsize=17, color=GREY, fontweight="bold")
ax.text(0.5, 19.5, "Features", fontsize=17, color=GREY, fontweight="bold")
box(0.5, 24, 16, 16, "Mobility data", ["Seoul OA-22298", "daily trips by", "arrival dong"])
box(20, 24, 17, 16, "Cleaning", ["daily → weekly mean", "(4+ valid days)", "Aug–Oct 2024 jump", "rescaled (÷1.52)"])
box(40.5, 24, 17, 16, "Ground truth", ["arrivals, age 20–39,", "shopping · leisure", "· other trips", "199 weeks × 416 dongs"], ec=RED, tc=RED)
box(0.5, 2, 16, 16, "External data", ["Naver search", "F&B licenses", "subway exits", "holidays"])
box(20, 2, 17, 16, "Data mapping", ["427 dongs (2025)", "name tags by area", "(neighborhood ·", "station · commercial)"])
box(40.5, 2, 17, 16, "Features", ["search: Seoul share", "F&B net openings", "subway exits", "only past values (lag)"])
box(61, 13, 17, 16, "Forecast 13 weeks", ["baselines: last week,", "same week last year", "ARIMA + Fourier", "combined 1:1", "(+ features)"])
box(81.5, 24, 18, 16, "Evaluate", ["4 forecast dates", "× 416 dongs", "WAPE · DM test", "top-10% hit rate"])
box(81.5, 2, 18, 16, "Hotspots → HEAT", ["hot: top 10% level", "emerging: top 10%", "growth vs last year", "interactive map"], ec=RED, tc=RED)
arr(16.9, 32, 19.8, 32); arr(37.4, 32, 40.3, 32); arr(16.9, 10, 19.8, 10); arr(37.4, 10, 40.3, 10)
arr(57.9, 30, 61, 24); arr(57.9, 12, 61, 18)
arr(78.4, 24, 81.3, 30); arr(78.4, 18, 81.3, 12)
ax.text(69.5, 33.5, "Next: pooled LightGBM + Ridge,\nevent peaks, 2026 test, 2027 H1", fontsize=14.5, color=GREY, ha="center", style="italic")
f.savefig(C.REPORT / "slides" / "fig" / "S11_pipeline.png", bbox_inches="tight", pad_inches=0.15, dpi=200)
