"""Research sampling frame for 100 named Seoul areas (not administrative dongs)."""

from collect_naver_areas import AREAS as INITIAL_AREAS

# Deliberately includes residential and less-publicized areas as comparisons.
# Labels indicate sampling intent, not an empirically verified popularity ranking.
COMPARISON_TERMS = [
    "가양동", "개봉동", "고덕동", "공덕동", "공릉동", "구로동", "구의동",
    "군자동", "금호동", "길동", "길음동", "녹번동", "대림동", "대방동",
    "도곡동", "도봉동", "독산동", "동작동", "둔촌동", "등촌동", "마장동",
    "마천동", "면목동", "목동", "미아동", "반포동", "방배동", "방학동",
    "봉천동", "불광동", "사당동", "상계동", "상도동", "상봉동", "석관동",
    "성내동", "성북동", "세곡동", "송파동", "수서동", "수유동", "시흥동",
    "신내동", "신림동", "신정동", "쌍문동", "암사동", "양재동", "염창동",
    "오류동", "오금동", "우이동", "월계동", "응암동", "이문동", "자양동",
    "잠원동", "장안동", "장위동", "정릉동", "중계동", "중곡동", "창동",
    "천호동", "청량리", "하계동", "홍은동", "화곡동", "휘경동",
]

assert len(INITIAL_AREAS) == 31
assert len(COMPARISON_TERMS) == 69
assert len({name for name, _ in INITIAL_AREAS} | set(COMPARISON_TERMS)) == 100

AREAS_100 = [(name, term, "initial_31") for name, term in INITIAL_AREAS]
AREAS_100 += [(name, name, "comparison_69") for name in COMPARISON_TERMS]
