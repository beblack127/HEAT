from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "subway_raw"

def load():
    files = sorted(RAW.glob("CARD_SUBWAY_MONTH_202[45][0-9][0-9].csv"))
    if len(files) != 24:
        raise RuntimeError(f"Expected 24 monthly files, found {len(files)}")
    frames = []
    for path in files:
        encoding = "utf-8-sig" if path.read_bytes()[:3] == b"\xef\xbb\xbf" else "cp949"
        df = pd.read_csv(path, encoding=encoding, index_col=False, dtype={"사용일자": str})
        df = df.rename(columns={"노선명": "호선명"})
        expected = ["사용일자", "호선명", "역명", "승차총승객수", "하차총승객수", "등록일자"]
        if df.columns.tolist() != expected:
            raise RuntimeError(f"Unexpected columns in {path.name}: {df.columns.tolist()}")
        df["date"] = pd.to_datetime(df["사용일자"], format="%Y%m%d", errors="raise")
        df["month"] = df["date"].dt.to_period("M").astype(str)
        if df["month"].nunique() != 1 or df["month"].iat[0] != path.stem[-6:-2] + "-" + path.stem[-2:]:
            raise RuntimeError(f"Date mismatch in {path.name}")
        frames.append(df)
    all_rows = pd.concat(frames, ignore_index=True)
    for col in ("승차총승객수", "하차총승객수"):
        all_rows[col] = pd.to_numeric(all_rows[col], errors="raise")
    return all_rows

if __name__ == "__main__":
    d = load()
    print("rows", len(d), "date range", d.date.min().date(), d.date.max().date())
    print("nulls", d.isna().sum().to_dict())
    print("duplicate date-line-station", d.duplicated(["date", "호선명", "역명"]).sum())
    print("unique stations", d["역명"].nunique(), "line-station pairs", d[["호선명", "역명"]].drop_duplicates().shape[0])
    print("monthly coverage")
    print(d.groupby("month").agg(rows=("date", "size"), days=("date", "nunique"), stations=("역명", "nunique")).to_string())
    print("top lines", d.groupby("호선명")["하차총승객수"].sum().sort_values(ascending=False).head(15).to_dict())
    # Sum lines that serve the same named station before ranking stations.
    daily = d.groupby(["date", "역명"], as_index=False)[["승차총승객수", "하차총승객수"]].sum()
    daily["year"] = daily.date.dt.year
    daily["weekend"] = daily.date.dt.weekday >= 5
    annual = daily.groupby(["역명", "year"]).agg(
        days=("date", "nunique"),
        exits=("하차총승객수", "sum"),
        avg_exits=("하차총승객수", "mean"),
    ).reset_index()
    wide = annual.pivot(index="역명", columns="year", values=["days", "exits", "avg_exits"])
    wide.columns = [f"{field}_{year}" for field, year in wide.columns]
    weekend = daily[daily.year == 2025].groupby(["역명", "weekend"])["하차총승객수"].mean().unstack()
    wide["weekend_ratio_2025"] = weekend[True] / weekend[False]
    weekend_y = daily.groupby(["역명", "year", "weekend"])["하차총승객수"].mean().unstack("year")
    weekend_y["yoy_pct"] = (weekend_y[2025] / weekend_y[2024] - 1) * 100
    wide["weekend_growth_pct"] = weekend_y.xs(True, level="weekend")["yoy_pct"]
    wide["weekday_growth_pct"] = weekend_y.xs(False, level="weekend")["yoy_pct"]
    wide["change_pct"] = (wide["avg_exits_2025"] / wide["avg_exits_2024"] - 1) * 100
    wide["delta_daily"] = wide["avg_exits_2025"] - wide["avg_exits_2024"]
    valid = wide[(wide["days_2024"] >= 350) & (wide["days_2025"] >= 350) & (wide["avg_exits_2024"] >= 3000)]
    print("eligible stations", len(valid))
    for label, sort in [
        ("top_2025_exits", ["avg_exits_2025"]),
        ("top_growth_pct", ["change_pct"]),
        ("top_daily_gain", ["delta_daily"]),
        ("top_growth_weekend", ["change_pct"]),
    ]:
        pool = valid if label != "top_growth_weekend" else valid[valid.weekend_ratio_2025 >= 1.1]
        cols = ["avg_exits_2024", "avg_exits_2025", "change_pct", "delta_daily", "weekend_ratio_2025", "weekend_growth_pct"]
        print(label)
        print(pool.sort_values(sort, ascending=False)[cols].head(20).round(1).to_string())
    print("top_weekend_growth")
    print(valid.sort_values("weekend_growth_pct", ascending=False)[cols].head(20).round(1).to_string())
    for station in ("서울역", "성수", "둔촌동", "마곡", "마곡나루(서울식물원)", "홍대입구"):
        print("station", station)
        print(d[d["역명"] == station].groupby([d.date.dt.year, "호선명"])["하차총승객수"].agg(["sum", "count"]).to_string())
    two = d[d["호선명"] == "2호선"].groupby(["date", "역명"], as_index=False)["하차총승객수"].sum()
    two["year"] = two.date.dt.year
    two_annual = two.groupby(["역명", "year"])["하차총승객수"].mean().unstack()
    two_annual["growth_pct"] = (two_annual[2025] / two_annual[2024] - 1) * 100
    two_annual["daily_gain"] = two_annual[2025] - two_annual[2024]
    print("2호선 top volume")
    print(two_annual.sort_values(2025, ascending=False).head(15).round(1).to_string())
    print("2호선 top growth baseline>=3000")
    print(two_annual[two_annual[2024] >= 3000].sort_values("growth_pct", ascending=False).head(15).round(1).to_string())

    # Slide-ready exploratory figure. Both rankings use the same 2호선 station scope.
    regular = "C:/Windows/Fonts/malgun.ttf"
    bold = "C:/Windows/Fonts/malgunbd.ttf"
    f_title = ImageFont.truetype(bold, 48)
    f_sub = ImageFont.truetype(regular, 26)
    f_head = ImageFont.truetype(bold, 30)
    f_label = ImageFont.truetype(regular, 23)
    f_value = ImageFont.truetype(bold, 22)
    f_callout = ImageFont.truetype(bold, 30)
    f_note = ImageFont.truetype(regular, 18)
    canvas = Image.new("RGB", (1920, 1080), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    navy, orange = "#55718f", "#e07a2d"
    dark, muted = "#182c40", "#526579"
    draw.text((80, 70), "지하철 역별 EDA | 규모와 증가세를 함께 보기", font=f_title, fill=dark)
    draw.text((80, 143), "2호선 · 2024~2025년 일별 승하차 데이터 · 역별 일평균 하차 비교", font=f_sub, fill=muted)
    draw.line((80, 200, 1840, 200), fill="#d9e2ec", width=2)
    draw.text((80, 234), "2025년 하차 규모 TOP 10", font=f_head, fill=dark)
    draw.text((990, 234), "전년 대비 증가율 TOP 10", font=f_head, fill=dark)
    draw.text((80, 280), "단위: 만 명/일", font=f_note, fill=muted)
    draw.text((990, 280), "2024년 일평균 3천 명 이상인 역", font=f_note, fill=muted)
    top_volume = two_annual.sort_values(2025, ascending=False).head(10)
    top_growth = two_annual[two_annual[2024] >= 3000].sort_values("growth_pct", ascending=False).head(10)
    for rank, (name, row) in enumerate(top_volume.iterrows(), 1):
        y = 340 + (rank - 1) * 54
        val = row[2025] / 10000
        draw.text((80, y), f"{rank:>2}. {name}", font=f_label, fill=dark)
        draw.rounded_rectangle((340, y + 4, 340 + round(val / 8 * 430), y + 30), radius=7,
                               fill=orange if name == "성수" else navy)
        draw.text((790, y), f"{val:.1f}", font=f_value, fill=dark)
    for rank, (name, row) in enumerate(top_growth.iterrows(), 1):
        y = 340 + (rank - 1) * 54
        val = row["growth_pct"]
        draw.text((990, y), f"{rank:>2}. {name}", font=f_label, fill=dark)
        draw.rounded_rectangle((1390, y + 4, 1390 + round(val / 16 * 330), y + 30), radius=7,
                               fill=orange if name == "성수" else navy)
        draw.text((1740, y), f"+{val:.1f}%", font=f_value, fill=dark)
    draw.line((80, 900, 1840, 900), fill="#d9e2ec", width=2)
    draw.text((80, 925), "성수역: 하차 규모 5위 · 증가율 1위 (+15.6%) · 하루 평균 약 7,142명 증가", font=f_callout, fill="#c55f1d")
    draw.text((80, 1000), "자료: 서울열린데이터광장 OA-12914. 하차 인원은 방문 목적을 구분하지 않으므로 상권 성장의 단독 근거로 해석하지 않음.", font=f_note, fill=muted)
    out = ROOT / "subway_eda_slide.png"
    canvas.save(out)
    print("saved figure", out)
