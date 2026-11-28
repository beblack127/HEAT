"""Create a WAPE chart for the final common-dong 427-nameplate comparison."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = json.loads(args.summary.read_text(encoding="utf-8"))
    count = result["evaluated_dongs_common"]
    font = "Malgun Gothic" if any(f.name == "Malgun Gothic" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False})
    names = ["2개월 전 값", "ARIMA", "SARIMA", "ARIMAX", "SARIMAX"]
    keys = ["lag2_naive", "ARIMA", "SARIMA", "ARIMAX", "SARIMAX"]
    colors = ["#8492a6", "#2176ae", "#4c9f70", "#d98932", "#b15d8c"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=150, sharey=True)
    fig.patch.set_facecolor("#f8fafc")
    for ax, (label, period) in zip(axes, (("2025년 전체", "all_2025"), ("2025년 1~8월", "jan_aug"))):
        values = [result["aggregate_same_dongs"][period][key]["wape"] * 100 for key in keys]
        bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], height=.62)
        ax.set_xlim(0, max(values) * 1.25)
        ax.set_title(f"{label} · {count}개 동", fontsize=16, weight="bold", pad=15)
        ax.set_xlabel("WAPE (%) · 낮을수록 좋음", fontsize=12)
        ax.set_facecolor("#f8fafc")
        ax.grid(axis="x", color="#dbe3eb", linewidth=.8)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values[::-1]):
            ax.text(value + max(values) * .015, bar.get_y() + bar.get_height()/2,
                    f"{value:.2f}%", ha="left", va="center", fontsize=11, weight="bold")
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.tick_params(axis="both", length=0, labelsize=11)
    fig.suptitle("행정동 생활이동 예측 · 4개 모델과 기준선", fontsize=22, weight="bold", y=.98)
    fig.text(.5, .055, "목표: 목적 4+5+7 일평균 도착 이동량 · 매월 1일 기준 m-2까지 사용 · X: 2개월 전 주간 생활인구",
             ha="center", fontsize=10, color="#425466")
    fig.text(.5, .025, "2024년 8~10월 원인 미확인 급증이 학습 자료에 포함됨 · 네이버 검색량은 미포함",
             ha="center", fontsize=10, color="#8b4351")
    fig.subplots_adjust(top=.82, bottom=.16, left=.15, right=.97, wspace=.26)
    fig.savefig(args.out, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
