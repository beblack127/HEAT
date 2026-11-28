"""Make a restrained, publication-style figure from the verified subway CSVs."""

from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from subway_eda import load


HERE = Path(__file__).resolve().parent
DATA_SOURCE = "서울열린데이터광장 OA-12914 | 2024–2025"
FONT_REGULAR = "C:/Windows/Fonts/malgun.ttf"
FONT_BOLD = "C:/Windows/Fonts/malgunbd.ttf"


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def main():
    raw = load()
    line2 = raw[raw["호선명"] == "2호선"].copy()
    daily = line2.groupby(["date", "역명"], as_index=False)["하차총승객수"].sum()
    daily["year"] = daily.date.dt.year
    annual = daily.groupby(["역명", "year"]).agg(
        days=("date", "nunique"), daily_exits=("하차총승객수", "mean")
    ).reset_index()
    values = annual.pivot(index="역명", columns="year", values="daily_exits")
    days = annual.pivot(index="역명", columns="year", values="days")
    values = values[(days[2024] >= 350) & (days[2025] >= 350)].copy()
    values["growth"] = (values[2025] / values[2024] - 1) * 100
    top = values.nlargest(12, 2025)
    scatter = values[values[2024] >= 3000]

    # White canvas, black axes, grey marks, direct labels only for reference stations.
    W, H = 2400, 1350
    im = Image.new("RGB", (W, H), "white")
    g = ImageDraw.Draw(im)
    black, grey, grid, light = "#222222", "#777777", "#d9d9d9", "#a8a8a8"
    f_panel = font(35, True)
    f_axis = font(26)
    f_tick = font(23)
    f_name = font(26)
    f_note = font(20)

    # Panel (a): 2025 station ranking.
    g.text((92, 70), "(a) 2025년 역별 일평균 하차 인원", font=f_panel, fill=black)
    x0, x1 = 550, 1050
    y0, y1 = 295, 1075
    scale_a = lambda v: x0 + v / 80000 * (x1 - x0)
    for tick in (0, 20000, 40000, 60000, 80000):
        x = round(scale_a(tick))
        g.line((x, y0, x, y1), fill=grid, width=2)
        label = f"{tick // 1000}"
        box = g.textbbox((0, 0), label, font=f_tick)
        g.text((x - (box[2] - box[0]) / 2, y1 + 18), label, font=f_tick, fill=grey)
    for rank, (name, row) in enumerate(top.iterrows(), 1):
        y = 325 + (rank - 1) * 66
        v = row[2025]
        g.text((92, y - 17), f"{rank:>2}.  {name}", font=f_name, fill=black)
        g.line((x0, y, round(scale_a(v)), y), fill=light, width=3)
        radius = 10 if name == "성수" else 7
        color = black if name == "성수" else grey
        x = round(scale_a(v))
        g.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
        g.text((1080, y - 16), f"{v/1000:.1f}", font=f_name, fill=black)
    g.text((655, 1140), "일평균 하차 인원 (천 명/일)", font=f_axis, fill=black)

    # Panel (b): station size versus annual change; all eligible stations shown.
    g.text((1260, 70), "(b) 역 규모와 전년 대비 변화", font=f_panel, fill=black)
    sx0, sx1 = 1360, 2240
    sy0, sy1 = 260, 1075
    x_map = lambda v: sx0 + v / 85000 * (sx1 - sx0)
    y_map = lambda v: sy1 - (v + 10) / 30 * (sy1 - sy0)
    for tick in (0, 20000, 40000, 60000, 80000):
        x = round(x_map(tick))
        g.line((x, sy0, x, sy1), fill=grid, width=2)
        label = str(tick // 1000)
        box = g.textbbox((0, 0), label, font=f_tick)
        g.text((x - (box[2] - box[0]) / 2, sy1 + 18), label, font=f_tick, fill=grey)
    for tick in (-10, 0, 10, 20):
        y = round(y_map(tick))
        g.line((sx0, y, sx1, y), fill=black if tick == 0 else grid, width=3 if tick == 0 else 2)
        g.text((1295, y - 14), str(tick), font=f_tick, fill=grey)
    for name, row in scatter.iterrows():
        x, y = round(x_map(row[2024])), round(y_map(row["growth"]))
        if name == "성수":
            g.ellipse((x-12, y-12, x+12, y+12), fill=black)
            g.text((x+20, y-38), f"성수  +{row['growth']:.1f}%", font=f_name, fill=black)
        else:
            g.ellipse((x-7, y-7, x+7, y+7), fill=grey)
    g.text((1570, 1140), "2024년 일평균 하차 인원 (천 명/일)", font=f_axis, fill=black)
    g.text((1240, 185), "전년 대비 증가율 (%)", font=f_axis, fill=black)

    g.line((90, 1220, 2310, 1220), fill=grid, width=2)
    g.text((92, 1245), "(b) 2024년 일평균 하차 3천 명 이상·연간 350일 이상 관측된 2호선 역만 표시", font=f_note, fill=grey)
    g.text((92, 1285), DATA_SOURCE, font=f_note, fill=grey)
    output = HERE / "subway_eda_paper.png"
    im.save(output, dpi=(300, 300))
    print(output)
    print(f"ranking_stations={len(top)}, scatter_stations={len(scatter)}")


if __name__ == "__main__":
    main()
