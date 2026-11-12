#!/usr/bin/env python3
"""Create four English monthly charts and a single four-panel image."""

import csv
import html
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "EDA_동별분석" / "월별_구동_개폐업_2023-2025.csv"
OUT = Path(__file__).resolve().parent
MONTHS = [f"{year:04d}-{month:02d}" for year in range(2023, 2026) for month in range(1, 13)]

REGIONS = [
    ("구", "중랑구", "", "01_district_jungnang_openings_closures_increase.svg", "(a) Jungnang-gu"),
    ("구", "중구", "", "02_district_jung_decrease.svg", "(b) Jung-gu"),
    ("동", "강남구", "삼성1동", "03_neighborhood_samseong_1_increase.svg", "(c) Samseong 1-dong"),
    ("동", "강남구", "역삼2동", "04_neighborhood_yeoksam_2_decrease.svg", "(d) Yeoksam 2-dong"),
]

PANEL_W, PANEL_H = 1100, 760
PLOT_X, PLOT_Y, PLOT_W, PLOT_H = 150, 88, 765, 515
PLOT_RIGHT = PLOT_X + PLOT_W
PLOT_BOTTOM = PLOT_Y + PLOT_H
LABEL_X = PLOT_RIGHT + 34
OPEN_COLOR, CLOSE_COLOR = "#2478B8", "#E87735"
INK, MUTED, GRID = "#202A35", "#536273", "#E1E6EA"
FONT_REGULAR_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

OUTER = 50
GAP_X, GAP_Y = 50, 50
COMPOSITE_W = OUTER * 2 + PANEL_W * 2 + GAP_X
COMPOSITE_H = OUTER * 2 + PANEL_H * 2 + GAP_Y
COMPOSITE_STEM = "00_four_panel_monthly_openings_closures"


def escape(value):
    return html.escape(str(value), quote=True)


def read_data():
    data = defaultdict(lambda: defaultdict(lambda: {"개업": 0, "폐업": 0}))
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["단위"], row["자치구"], row["행정동"])
            data[key][row["월"]] = {"개업": int(row["개업"]), "폐업": int(row["폐업"])}
    return data


def chart_values(key, data):
    values = data[key]
    openings = [values[month]["개업"] for month in MONTHS]
    closures = [values[month]["폐업"] for month in MONTHS]
    maximum = max(openings + closures)
    step = 20 if maximum <= 140 else 50 if maximum <= 360 else 100
    y_max = max(step * 4, math.ceil(maximum / step) * step)
    return openings, closures, y_max


def positions(y_max):
    x_at = lambda index: PLOT_X + index * PLOT_W / (len(MONTHS) - 1)
    y_at = lambda value: PLOT_BOTTOM - value / y_max * PLOT_H
    return x_at, y_at


def endpoint_labels(openings, closures, y_at):
    opening_y = y_at(openings[-1])
    closure_y = y_at(closures[-1])
    opening_label_y, closure_label_y = opening_y, closure_y
    if abs(opening_y - closure_y) < 36:
        if opening_y <= closure_y:
            opening_label_y = max(PLOT_Y + 18, opening_y - 22)
            closure_label_y = min(PLOT_BOTTOM - 8, closure_y + 22)
        else:
            closure_label_y = max(PLOT_Y + 18, closure_y - 22)
            opening_label_y = min(PLOT_BOTTOM - 8, opening_y + 22)
    return ((opening_y, opening_label_y), (closure_y, closure_label_y))


def svg_chart_body(key, label, data):
    openings, closures, y_max = chart_values(key, data)
    x_at, y_at = positions(y_max)
    pieces = [f'<rect width="{PANEL_W}" height="{PANEL_H}" fill="#FFFFFF"/>']
    pieces.append(f'<text x="28" y="45" font-family="Arial, sans-serif" font-size="30" font-weight="700" fill="{INK}">{escape(label)}</text>')

    for tick in range(5):
        value = y_max * tick / 4
        y = y_at(value)
        pieces.append(f'<line x1="{PLOT_X}" y1="{y:.2f}" x2="{PLOT_RIGHT}" y2="{y:.2f}" stroke="{GRID}" stroke-width="1.4"/>')
        pieces.append(f'<text x="{PLOT_X - 16}" y="{y + 7:.2f}" font-family="Arial, sans-serif" font-size="21" font-weight="700" fill="{MUTED}" text-anchor="end">{int(round(value)):,}</text>')

    for index in (12, 24):
        x = x_at(index)
        pieces.append(f'<line x1="{x:.2f}" y1="{PLOT_Y}" x2="{x:.2f}" y2="{PLOT_BOTTOM}" stroke="#AAB4BE" stroke-width="1.2" stroke-dasharray="4 5"/>')

    tick_indices = (0, 6, 12, 18, 24, 30)
    for index in tick_indices:
        x = x_at(index)
        month_label = MONTHS[index].replace("-", ".")
        anchor = "start" if index == 0 else "middle"
        pieces.append(f'<line x1="{x:.2f}" y1="{PLOT_BOTTOM}" x2="{x:.2f}" y2="{PLOT_BOTTOM + 7}" stroke="{MUTED}" stroke-width="1.4"/>')
        pieces.append(f'<text x="{x:.2f}" y="{PLOT_BOTTOM + 34}" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="{INK}" text-anchor="{anchor}">{escape(month_label)}</text>')

    pieces.extend([
        f'<line x1="{PLOT_X}" y1="{PLOT_Y}" x2="{PLOT_X}" y2="{PLOT_BOTTOM}" stroke="{INK}" stroke-width="1.8"/>',
        f'<line x1="{PLOT_X}" y1="{PLOT_BOTTOM}" x2="{PLOT_RIGHT}" y2="{PLOT_BOTTOM}" stroke="{INK}" stroke-width="1.8"/>',
        f'<text x="{PLOT_X + PLOT_W / 2}" y="{PANEL_H - 25}" font-family="Arial, sans-serif" font-size="25" font-weight="700" fill="{INK}" text-anchor="middle">Month</text>',
        f'<text x="37" y="{PLOT_Y + PLOT_H / 2}" font-family="Arial, sans-serif" font-size="25" font-weight="700" fill="{INK}" text-anchor="middle" transform="rotate(-90 37 {PLOT_Y + PLOT_H / 2})">Monthly event count</text>',
    ])

    def add_series(series, color):
        points = " ".join(f"{x_at(i):.2f},{y_at(value):.2f}" for i, value in enumerate(series))
        elements = [f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3.8" stroke-linejoin="round" stroke-linecap="round"/>']
        for i, value in enumerate(series):
            elements.append(f'<circle cx="{x_at(i):.2f}" cy="{y_at(value):.2f}" r="4.1" fill="{color}" stroke="#FFFFFF" stroke-width="1.2"/>')
        return elements

    pieces.extend(add_series(openings, OPEN_COLOR))
    pieces.extend(add_series(closures, CLOSE_COLOR))

    for name, color, (point_y, label_y) in (
        ("Openings", OPEN_COLOR, endpoint_labels(openings, closures, y_at)[0]),
        ("Closures", CLOSE_COLOR, endpoint_labels(openings, closures, y_at)[1]),
    ):
        pieces.append(f'<line x1="{PLOT_RIGHT + 4}" y1="{point_y:.2f}" x2="{LABEL_X - 9}" y2="{label_y:.2f}" stroke="{color}" stroke-width="1.8"/>')
        pieces.append(f'<text x="{LABEL_X}" y="{label_y + 8:.2f}" font-family="Arial, sans-serif" font-size="21" font-weight="700" fill="{color}">{name}</text>')
    return "\n".join(pieces)


def write_individual_svgs(data):
    for unit, gu, dong, filename, label in REGIONS:
        body = svg_chart_body((unit, gu, dong), label, data)
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{PANEL_W}" height="{PANEL_H}" viewBox="0 0 {PANEL_W} {PANEL_H}">\n{body}\n</svg>'
        (OUT / filename).write_text(svg, encoding="utf-8")


def svg_composite(data):
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{COMPOSITE_W}" height="{COMPOSITE_H}" viewBox="0 0 {COMPOSITE_W} {COMPOSITE_H}">', '<rect width="100%" height="100%" fill="#FFFFFF"/>']
    for index, (unit, gu, dong, _filename, label) in enumerate(REGIONS):
        col, row = index % 2, index // 2
        x = OUTER + col * (PANEL_W + GAP_X)
        y = OUTER + row * (PANEL_H + GAP_Y)
        body = svg_chart_body((unit, gu, dong), label, data)
        pieces.append(f'<svg x="{x}" y="{y}" width="{PANEL_W}" height="{PANEL_H}" viewBox="0 0 {PANEL_W} {PANEL_H}">{body}</svg>')
    pieces.append('</svg>')
    return "\n".join(pieces)


def load_font(size, bold=False):
    path = FONT_BOLD_PATH if bold else FONT_REGULAR_PATH
    return ImageFont.truetype(path, size=size)


def draw_centered(draw, text, center_x, top_y, font, fill):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text((center_x - (right - left) / 2, top_y - top), text, font=font, fill=fill)


def draw_right_aligned(draw, text, right_x, center_y, font, fill):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text((right_x - (right - left), center_y - (bottom - top) / 2 - top), text, font=font, fill=fill)


def raster_chart(key, label, data):
    openings, closures, y_max = chart_values(key, data)
    x_at, y_at = positions(y_max)
    canvas = Image.new("RGB", (PANEL_W, PANEL_H), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(30, bold=True)
    tick_font = load_font(21, bold=True)
    axis_font = load_font(25, bold=True)
    series_font = load_font(21, bold=True)

    draw.text((28, 20), label, font=title_font, fill=INK)
    for tick in range(5):
        value = y_max * tick / 4
        y = y_at(value)
        draw.line((PLOT_X, y, PLOT_RIGHT, y), fill=GRID, width=2)
        draw_right_aligned(draw, f"{int(round(value)):,}", PLOT_X - 16, y, tick_font, MUTED)

    for index in (12, 24):
        x = x_at(index)
        dash_y = PLOT_Y
        while dash_y < PLOT_BOTTOM:
            draw.line((x, dash_y, x, min(dash_y + 5, PLOT_BOTTOM)), fill="#AAB4BE", width=1)
            dash_y += 10

    tick_indices = (0, 6, 12, 18, 24, 30)
    for index in tick_indices:
        x = x_at(index)
        draw.line((x, PLOT_BOTTOM, x, PLOT_BOTTOM + 7), fill=MUTED, width=2)
        label_text = MONTHS[index].replace("-", ".")
        if index == 0:
            draw.text((x, PLOT_BOTTOM + 10), label_text, font=tick_font, fill=INK, anchor="lt")
        else:
            draw_centered(draw, label_text, x, PLOT_BOTTOM + 10, tick_font, INK)

    draw.line((PLOT_X, PLOT_Y, PLOT_X, PLOT_BOTTOM), fill=INK, width=2)
    draw.line((PLOT_X, PLOT_BOTTOM, PLOT_RIGHT, PLOT_BOTTOM), fill=INK, width=2)
    draw_centered(draw, "Month", PLOT_X + PLOT_W / 2, PANEL_H - 43, axis_font, INK)

    y_label = "Monthly event count"
    label_image = Image.new("RGBA", (360, 40), (255, 255, 255, 0))
    ImageDraw.Draw(label_image).text((0, 0), y_label, font=axis_font, fill=INK)
    rotated = label_image.rotate(90, expand=True)
    canvas.paste(rotated, (int(37 - rotated.width / 2), int(PLOT_Y + PLOT_H / 2 - rotated.height / 2)), rotated)

    def add_series(series, color):
        points = [(x_at(i), y_at(value)) for i, value in enumerate(series)]
        draw.line(points, fill=color, width=4, joint="curve")
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline="white", width=1)

    add_series(openings, OPEN_COLOR)
    add_series(closures, CLOSE_COLOR)

    for name, color, (point_y, label_y) in (
        ("Openings", OPEN_COLOR, endpoint_labels(openings, closures, y_at)[0]),
        ("Closures", CLOSE_COLOR, endpoint_labels(openings, closures, y_at)[1]),
    ):
        draw.line((PLOT_RIGHT + 4, point_y, LABEL_X - 9, label_y), fill=color, width=2)
        bbox = draw.textbbox((0, 0), name, font=series_font)
        draw.text((LABEL_X, label_y - (bbox[3] - bbox[1]) / 2 - bbox[1]), name, font=series_font, fill=color)
    return canvas


def write_images(data):
    panels = []
    for unit, gu, dong, _filename, label in REGIONS:
        panels.append(raster_chart((unit, gu, dong), label, data))
    composite = Image.new("RGB", (COMPOSITE_W, COMPOSITE_H), "white")
    for index, panel in enumerate(panels):
        col, row = index % 2, index // 2
        x = OUTER + col * (PANEL_W + GAP_X)
        y = OUTER + row * (PANEL_H + GAP_Y)
        composite.paste(panel, (x, y))
    composite.save(OUT / f"{COMPOSITE_STEM}.png", format="PNG", dpi=(180, 180), optimize=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = read_data()
    write_individual_svgs(data)
    (OUT / f"{COMPOSITE_STEM}.svg").write_text(svg_composite(data), encoding="utf-8")
    write_images(data)
    print(f"Generated four labeled charts and a 2x2 image in {OUT}")


if __name__ == "__main__":
    main()
