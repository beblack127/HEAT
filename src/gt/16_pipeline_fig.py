"""작업 흐름 그림 (단순·아이콘). 아이콘: react-icons(Font Awesome)을 흰색 PNG로 그려 둔 것. 출력: reports/gt_midterm/slides/fig/S11_pipeline.png"""
import sys
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.image as mpimg
src = open("11_slide_figs.py", encoding="utf-8").read(); exec(compile(src[:src.index("# ---- S1")], "h", "exec"))
ICON = sys.argv[1] if len(sys.argv) > 1 else str(C.REPORT / "slides" / "fig" / "icons")
steps = [("data", "Data", "mobility, search,\nF&B, subway", NAVY), ("mapping", "Mapping", "427 dongs", NAVY), ("cleaning", "Cleaning", "weekly,\nanomaly fixed", NAVY),
         ("truth", "Ground truth", "visitors 20–39", RED), ("forecast", "Forecast", "13 weeks ahead", NAVY), ("evaluate", "Evaluate", "4 dates × 416 dongs", NAVY),
         ("heat", "HEAT map", "hot & emerging", RED)]
f, ax = plt.subplots(figsize=(20, 5.2)); ax.set_xlim(0, 70); ax.set_ylim(0, 18); ax.set_aspect("equal"); ax.set_axis_off()
for i, (k, t, sub, c) in enumerate(steps):
    x = 5 + i * 10
    ax.add_patch(Circle((x, 11.5), 3.6, fc=c, ec="none"))
    im = mpimg.imread(f"{ICON}/{k}.png"); ax.imshow(im, extent=(x - 1.9, x + 1.9, 9.6, 13.4), zorder=3)
    ax.text(x, 6.6, t, ha="center", va="center", fontsize=21, fontweight="bold", color=c)
    ax.text(x, 4.6, sub, ha="center", va="top", fontsize=16, color=INK, linespacing=1.25)
    if i < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((x + 4.0, 11.5), (x + 6.0, 11.5), arrowstyle="-|>", mutation_scale=24, lw=2, color=GREY))
f.savefig(C.REPORT / "slides" / "fig" / "S11_pipeline.png", bbox_inches="tight", pad_inches=0.15, dpi=200)
