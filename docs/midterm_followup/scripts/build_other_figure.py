"""Figure: example trips that likely fall under purpose 7 ("Other"), in Korean and English.

Usage: python build_other_figure.py <out_dir>
"""
import os
import sys

OUT = sys.argv[1]
TEXT = {
    "ko": {
        "font": "font-family='\"Malgun Gothic\", \"Apple SD Gothic Neo\", sans-serif'",
        "title": "'기타(7)' 목적에 들어갈 만한 이동",
        "sub": "출근·등교·귀가·쇼핑·관광·병원에 속하지 않는 이동이 모두 '기타'로 분류된다",
        "head": ("핫플 방문과 관련 있음", "핫플 방문과 거리 멂"),
        "rows": [("외식 (식당·카페)", "학원"),
                 ("친구 만나기, 술자리, 밤 문화", "외근, 출장, 업무 미팅"),
                 ("영화·공연·전시", "친척·지인 집 방문"),
                 ("운동, 취미 활동 (헬스, 클라이밍, 공방)", "종교 활동"),
                 ("팝업 스토어, 행사", "은행, 관공서, 미용실 같은 개인 용무"),
                 ("산책, 공원, 한강", "배달·운송처럼 일하면서 다니는 이동")],
        "foot": "예시이며 공식 분류 기준이 아님. 2025년 기준 4+5+7 이동 중 '기타'가 98.7%.",
    },
    "en": {
        "font": "font-family='Georgia, \"Times New Roman\", serif'",
        "title": "What falls under the \"Other\" purpose (7)",
        "sub": "Any trip that is not commute, school, return home, shopping, tourism or hospital",
        "head": ("Related to hot-place visits", "Not related to hot-place visits"),
        "rows": [("Eating out (restaurants, cafés)", "Private academies (hagwon)"),
                 ("Meeting friends, drinks, nightlife", "Field work, business trips, meetings"),
                 ("Movies, performances, exhibitions", "Visiting family or friends at home"),
                 ("Sports and hobbies (gym, climbing, workshops)", "Religious activities"),
                 ("Pop-up stores, events", "Errands (bank, public office, hair salon)"),
                 ("Walks, parks, Han River", "Trips made while working (delivery, transport)")],
        "foot": "Examples only, not the official category definition. In 2025, \"Other\" made up 98.7% of purpose 4+5+7 trips.",
    },
}

W, H = 1600, 900
L, MID, RGT = 80, 800, 1520
T, ROW = 220, 86
os.makedirs(OUT, exist_ok=True)
for lang, t in TEXT.items():
    F = t["font"]
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>",
         f"<rect width='{W}' height='{H}' fill='#F7F7F7'/>",
         f"<text x='{L}' y='70' {F} font-size='38' font-weight='bold' fill='#222'>{t['title']}</text>",
         f"<text x='{L}' y='112' {F} font-size='24' fill='#555'>{t['sub']}</text>",
         # header bands
         f"<rect x='{L}' y='{T - 64}' width='{MID - L - 20}' height='56' fill='#1F4E9A'/>",
         f"<rect x='{MID + 20}' y='{T - 64}' width='{RGT - MID - 20}' height='56' fill='#9AA3AF'/>",
         f"<text x='{L + 24}' y='{T - 26}' {F} font-size='27' font-weight='bold' fill='#FFF'>{t['head'][0]}</text>",
         f"<text x='{MID + 44}' y='{T - 26}' {F} font-size='27' font-weight='bold' fill='#FFF'>{t['head'][1]}</text>"]
    for i, (a, b) in enumerate(t["rows"]):
        y = T + i * ROW
        if i % 2 == 0:
            s.append(f"<rect x='{L}' y='{y}' width='{MID - L - 20}' height='{ROW}' fill='#E6EDF7'/>")
            s.append(f"<rect x='{MID + 20}' y='{y}' width='{RGT - MID - 20}' height='{ROW}' fill='#ECEEF1'/>")
        s.append(f"<text x='{L + 24}' y='{y + ROW / 2 + 10}' {F} font-size='27' fill='#1A2E55'>{a}</text>")
        s.append(f"<text x='{MID + 44}' y='{y + ROW / 2 + 10}' {F} font-size='27' fill='#555'>{b}</text>")
    s.append(f"<text x='{L}' y='{H - 48}' {F} font-size='19' fill='#666'>{t['foot']}</text>")
    s.append("</svg>")
    with open(os.path.join(OUT, f"other_purpose_{lang}.svg"), "w", encoding="utf-8") as f:
        f.write("\n".join(s))
