"""공통 도구: 분석 단위 대응, 실행 기록, 한글 폰트."""
import csv, subprocess, datetime
import pandas as pd
import config as C

# 427동과 원자료 코드가 다른 경우
#  - 용신동(11230536)이 신설동(11230515)·용두동(11230533)으로 나뉨: 생활이동이 나뉘기 전 코드를 쓰므로 둘을 합쳐 하나로 본다
#  - 일원2동(11680740) -> 개포3동(11680675): 이름·코드만 바뀜
#  - 생활인구(옛 코드): 강북구 6개 동 번호 변경, 상일동(11740520) -> 상일1·2동, 오류2동 -> 항동 분리 (생활인구에만 해당)
MERGE = {"11230515": "11230536", "11230533": "11230536"}
RENAME = {"11680740": "11680675"}
LP_RENAME = {"11305590": "11305595", "11305600": "11305603", "11305606": "11305608",
             "11305610": "11305615", "11305620": "11305625", "11305630": "11305635"}
LP_SPLIT_PARENT = {"11740520": ["11740525", "11740526"]}


def dong_list():
    return pd.read_csv(C.DONG_LIST, encoding="utf-8-sig", dtype={"dong_code8": str})


def unit_map():
    """원자료 코드 -> 분석 단위. src_code(원자료 코드), unit, reason"""
    L = dong_list()
    rows = []
    for c in L.dong_code8:
        u = MERGE.get(c, c)
        rows.append((c, u, "용신동 분리 전 코드로 합침" if c in MERGE else ""))
    rows.append(("11230536", "11230536", "용신동(신설동+용두동) 분리 전 코드"))
    for old, new in RENAME.items():
        rows.append((old, new, "코드 변경 (일원2동 -> 개포3동)"))
    return pd.DataFrame(rows, columns=["src_code", "unit", "reason"]).drop_duplicates("src_code")


def units_of_dongs():
    """427동 -> 분석 단위 (지도 칠하기용)"""
    L = dong_list()
    return dict(zip(L.dong_code8, [MERGE.get(c, c) for c in L.dong_code8]))


def unit_names():
    L = dong_list().set_index("dong_code8")
    out = {c: (L.gu[c], L.dong_name[c]) for c in L.index if c not in MERGE}
    out["11230536"] = ("동대문구", "용신동(신설·용두)")
    return out


def log_run(step, note=""):
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=C.ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        h = ""
    new = not C.EXP_LOG.exists()
    with open(C.EXP_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new: w.writerow(["time", "step", "git", "spec_version", "plan", "seed", "note", "extra"])
        w.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"gt/{step}", h, "", "gt_midterm", C.SEED, note, ""])


def set_korean_font():
    import matplotlib
    from matplotlib import font_manager
    for n in ("AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "UnDotum"):
        if any(n == f.name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = n; break
    matplotlib.rcParams["axes.unicode_minus"] = False
