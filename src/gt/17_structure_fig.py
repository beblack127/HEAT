"""프로젝트 구조 그림 (단위 → 정답 → 입력 → 13주 예측 → 판정). 아이콘은 fig/icons. 출력: reports/gt_midterm/slides/fig/S12_structure.png"""
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import matplotlib.image as mpimg
src = open("11_slide_figs.py", encoding="utf-8").read(); exec(compile(src[:src.index("# ---- S1")], "h", "exec"))
IC = C.REPORT / "slides" / "fig" / "icons"
f, ax = plt.subplots(figsize=(17.2, 11)); ax.set_xlim(0, 86); ax.set_ylim(0, 55); ax.set_aspect("equal"); ax.set_axis_off()
def icon(x, y, r, k, c=NAVY):
    ax.add_patch(Circle((x, y), r, fc=c, ec="none", zorder=2)); s = r * .55
    ax.imshow(mpimg.imread(IC / f"{k}.png"), extent=(x - s, x + s, y - s, y + s), zorder=3)
def down(x, y0, y1): ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=26, lw=2.2, color=GREY))
def right(x0, x1, y): ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=22, lw=2, color=GREY))
def lab(x, y, t, size=17, c=INK, w="normal", ha="center"): ax.text(x, y, t, fontsize=size, color=c, fontweight=w, ha=ha, va="center")
def row(y, t): lab(1, y, t, 19, NAVY, "bold", "left")
# 1 단위
row(51, "Unit"); icon(33, 51, 2.6, "mapping"); icon(40, 51, 2.6, "calendar"); lab(44.5, 51, "dong × week", 19, INK, "bold", "left")
down(40, 47.6, 45)
# 2 정답
ax.add_patch(FancyBboxPatch((14, 35.5), 52, 9, boxstyle="round,pad=0.3,rounding_size=1", fc="white", ec=RED, lw=2.4))
row(40, "Ground truth"); icon(21, 40, 3, "truth", RED)
lab(27, 41.8, "Weekly visitors", 23, RED, "bold", "left"); lab(27, 38.0, "age 20–39 · shopping, leisure, other", 17, INK, "normal", "left")
down(40, 35, 32.4)
# 3 입력
row(28, "Inputs"); xs = [(20, "mobility", "Past mobility", "core", NAVY), (33, "search", "Search", "tested", NAVY), (46, "subway", "Subway", "tested", NAVY), (59, "store", "F&B openings", "tested", NAVY)]
for x, k, t, s_, c in xs:
    icon(x, 29.5, 2.4, k, c); lab(x, 25.6, t, 16, INK, "bold"); lab(x, 23.6, s_, 14, GREY)
    ax.add_patch(FancyArrowPatch((x, 22.4), (40, 19.8), arrowstyle="-", lw=1.4, color="#C9CED6", connectionstyle="arc3,rad=0"))
icon(40, 17.6, 2.2, "brain"); down(40, 15.2, 13.0)
# 4 예측
row(9.5, "13-week\nforecast"); fx = [(25, "history", "Baseline", "last year"), (36, "wave", "ARIMA · Fourier", "seasonality"), (47, "plus", "+ Features", "search etc.")]
for x, k, t, s_ in fx: icon(x, 10, 2.3, k); lab(x, 6.3, t, 16, INK, "bold"); lab(x, 4.3, s_, 14, GREY)
right(27.8, 33.2, 10); right(38.8, 44.2, 10)
# 5 판정
lab(70.5, 15.2, "Assessment", 19, NAVY, "bold"); icon(64, 10, 2.3, "evaluate"); icon(77, 10, 2.3, "heat", RED)
lab(64, 6.3, "Evaluate", 16, INK, "bold"); lab(64, 4.3, "4 dates × 416 dongs", 14, GREY)
lab(77, 6.3, "Hot & emerging", 16, INK, "bold"); lab(77, 4.3, "HEAT map", 14, GREY)
right(49.8, 61.2, 10); ax.plot([57.5, 57.5], [3, 16], color='#C9CED6', lw=1.2, ls='--')
f.savefig(C.REPORT / "slides" / "fig" / "S12_structure.png", bbox_inches="tight", pad_inches=0.15, dpi=200)
