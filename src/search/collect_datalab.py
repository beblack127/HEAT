"""NAVER API HUB '검색어 트렌드' API로 이름표 검색어 묶음의 월별 검색 지수를 모은다.

API HUB 제약: 요청당 키워드 그룹 5개, 그룹당 키워드 5개, 월 50,000회.
- 묶음(최대 18개 검색어)을 5개씩 '조각'으로 나눠 받고, 같은 척도로 맞춘 뒤 묶음별로 더한다.
  (데이터랩 그룹 값은 그룹 안 검색어 검색량의 합에 비례하므로, 같은 척도 위에서 조각을 더하면 묶음 전체 값이 된다)
- 요청 한 번에 조각 4개 + 기준 조각(ANCHOR) 1개. 값은 '그 요청 안 최댓값 = 100'인 상대값이라
  모든 요청에 같은 기준 조각을 넣고 그 값으로 척도를 맞춘다. 결과 척도: 기준 조각의 2022년 월평균 = 100
- 요청 결과는 raw/batch_XXX.json 으로 저장하고, 이미 받은 요청은 건너뛴다(이어받기).

사용:
  python3 collect_datalab.py --groups 02_검색어_묶음.csv --env .env --out 검색량_수집 --pilot "강남,연남,망원,합정,문래"
  python3 collect_datalab.py --groups 02_검색어_묶음.csv --env .env --out 검색량_수집 --anchor-root 연남
  ... --mock  : API 없이 가짜 응답으로 전체 흐름 시험
"""
import argparse, json, os, sys, time, random
import urllib.request, urllib.error
import numpy as np, pandas as pd

API = "https://naverapihub.apigw.ntruss.com/search-trend/v1/search"
MAX_KW = 5     # 그룹당 키워드 한도 (API HUB)
PER_REQ = 4    # 기준 조각 포함 그룹 5개가 요청당 한도


def read_env(path):
    env = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    for k in ("NAVER_CLIENT_ID", "NAVER_CLIENT_SECRET"):
        if not env.get(k):
            sys.exit(f".env에 {k} 가 없습니다")
    return env


def call_api(body, env, tries=6):
    data = json.dumps(body).encode("utf-8")
    for i in range(tries):
        req = urllib.request.Request(API, data=data, method="POST", headers={
            "X-NCP-APIGW-API-KEY-ID": env["NAVER_CLIENT_ID"], "X-NCP-APIGW-API-KEY": env["NAVER_CLIENT_SECRET"],
            "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "ignore")[:300]
            if e.code == 401:
                sys.exit(f"인증 실패 401: Client ID/Secret 또는 Application에 '검색어 트렌드' 선택 여부를 확인하세요. {msg}")
            if e.code in (400, 403):
                sys.exit(f"요청 오류 {e.code}: {msg}")
            wait = 2 ** i + random.random()
            print(f"  HTTP {e.code}, {wait:.1f}초 후 재시도: {msg}", flush=True)
            time.sleep(wait)
        except urllib.error.URLError as e:
            wait = 2 ** i + random.random()
            print(f"  연결 오류 {e.reason}, {wait:.1f}초 후 재시도", flush=True)
            time.sleep(wait)
    sys.exit("재시도 한도를 넘었습니다. 다시 실행하면 이어서 받습니다.")


# ---- 가짜 응답(시험용): 조각마다 숨은 '실제 검색량'을 만들고 API처럼 요청 안 최댓값 = 100으로 바꿔 돌려준다
_LATENT = {}
def latent(name, n):
    key = name
    if key not in _LATENT:
        r = np.random.default_rng(abs(hash(key)) % 2**32)
        base = float(np.exp(r.normal(1.5, 1.6)))
        trend = r.normal(0, 0.015)
        season = 1 + 0.1 * np.sin(np.arange(n) / 12 * 2 * np.pi)
        _LATENT[key] = base * np.exp(trend * np.arange(n)) * season
    return _LATENT[key]

def mock_api(body, months):
    ser = {g["groupName"]: latent("|".join(g["keywords"]), len(months)) for g in body["keywordGroups"]}
    mx = max(s.max() for s in ser.values())
    return {"results": [{"title": g["groupName"], "keywords": g["keywords"],
                         "data": [{"period": m, "ratio": round(float(v), 5)} for m, v in zip(months, ser[g["groupName"]] / mx * 100)
                                  if round(float(v), 5) > 0]} for g in body["keywordGroups"]]}


def series_of(res_item, months):
    vals = {d["period"]: d["ratio"] for d in res_item["data"]}
    return np.array([float(vals.get(m, 0.0)) for m in months])  # 0인 달은 응답에서 빠지므로 0으로 채운다


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True)
    ap.add_argument("--env", default=".env")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2026-10-31")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--anchor-root", default="", help="기준으로 쓸 묶음의 뿌리. 그 묶음의 첫 조각(검색어 5개)이 기준이 된다")
    ap.add_argument("--pilot", default="", help="기준 후보 뿌리(쉼표, 최대 5개): 첫 조각끼리 한 번 비교하고 끝냄")
    a = ap.parse_args()

    g = pd.read_csv(a.groups, encoding="utf-8-sig")
    g["kw"] = g.keywords.str.split(r"\s*\|\s*")
    months = pd.date_range(a.start, a.end, freq="MS").strftime("%Y-%m-%d").tolist()
    os.makedirs(os.path.join(a.out, "raw"), exist_ok=True)
    env = {}
    def ask(body):  # 키는 실제로 요청할 때만 읽는다 (이미 받은 요청만 있으면 .env 없이 다시 계산 가능)
        if a.mock:
            return mock_api(body, months)
        if not env:
            env.update(read_env(a.env))
        return call_api(body, env)
    first_chunk = lambda root: g.set_index("root").kw[root][:MAX_KW]

    if a.pilot:
        roots = [r.strip() for r in a.pilot.split(",") if r.strip()][:5]
        body = {"startDate": a.start, "endDate": a.end, "timeUnit": "month",
                "keywordGroups": [{"groupName": r, "keywords": first_chunk(r)} for r in roots]}
        res = ask(body)
        json.dump({"request": body, "response": res}, open(os.path.join(a.out, "pilot.json"), "w", encoding="utf-8"), ensure_ascii=False)
        print("후보 | 평균 | 최소 | 0인 달 | 변동계수 | 검색어")
        for r in res["results"]:
            v = series_of(r, months)
            print(f"{r['title']} | {v.mean():.2f} | {v.min():.2f} | {(v <= 0).sum()} | {v.std() / max(v.mean(), 1e-9):.3f} | {', '.join(r['keywords'])}")
        return

    if not a.anchor_root:
        sys.exit("--anchor-root 로 기준 묶음을 정해 주세요 (먼저 --pilot 으로 비교)")
    anchor = {"groupName": "ANCHOR", "keywords": first_chunk(a.anchor_root)}

    # 조각 만들기
    units = []
    for r in g.itertuples():
        for j in range(0, len(r.kw), MAX_KW):
            units.append((f"{r.keyword_group_id}#{j // MAX_KW}", r.keyword_group_id, r.kw[j:j + MAX_KW]))
    batches = [units[i:i + PER_REQ] for i in range(0, len(units), PER_REQ)]
    print(f"묶음 {len(g)}개 -> 조각 {len(units)}개 -> 요청 {len(batches)}회 (기간 {a.start} ~ {a.end}, 기준 '{a.anchor_root}' 첫 조각)", flush=True)
    for bi, b in enumerate(batches):
        fp = os.path.join(a.out, "raw", f"batch_{bi:04d}.json")
        if os.path.exists(fp):
            continue
        body = {"startDate": a.start, "endDate": a.end, "timeUnit": "month",
                "keywordGroups": [anchor] + [{"groupName": uid, "keywords": kws} for uid, _, kws in b]}
        res = ask(body)
        json.dump({"request": body, "response": res}, open(fp, "w", encoding="utf-8"), ensure_ascii=False)
        if (bi + 1) % 25 == 0 or bi == len(batches) - 1:
            print(f"  {bi + 1}/{len(batches)} 완료", flush=True)
        if not a.mock:
            time.sleep(a.sleep)

    # 척도 맞추기: 요청마다 기준 조각의 합이 첫 요청과 같아지도록 배율 k_b
    unit_group = {u: kg for u, kg, _ in units}
    anc, val = {}, []
    for bi in range(len(batches)):
        res = json.load(open(os.path.join(a.out, "raw", f"batch_{bi:04d}.json"), encoding="utf-8"))["response"]
        for r in res["results"]:
            v = series_of(r, months)
            if r["title"] == "ANCHOR": anc[bi] = v
            else: val.append((bi, r["title"], v))
    ref = anc[0]
    if (ref <= 0).any():
        print("주의: 기준 조각에 0인 달이 있습니다. 다른 기준을 고르는 것이 좋습니다.", flush=True)
    k = {bi: ref.sum() / anc[bi].sum() for bi in anc}
    base = ref[[i for i, m in enumerate(months) if m.startswith("2022")]].mean()
    shape_corr = min(np.corrcoef(ref, anc[bi])[0, 1] for bi in anc)
    unit_idx = pd.DataFrame([(unit_group[u], u, m[:7], x * k[bi] / base * 100, x, bi)
                             for bi, u, v in val for m, x in zip(months, v)],
                            columns=["keyword_group_id", "unit", "month", "index", "ratio", "batch"])
    unit_idx.round(5).to_csv(os.path.join(a.out, "검색량_조각별_월별.csv"), index=False, encoding="utf-8-sig")
    grp = unit_idx.groupby(["keyword_group_id", "month"])["index"].sum().reset_index()
    grp = grp.merge(g[["keyword_group_id", "root"]], on="keyword_group_id", how="left")
    grp[["keyword_group_id", "root", "month", "index"]].round(5).to_csv(
        os.path.join(a.out, "검색량_묶음별_월별.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame({"month": [m[:7] for m in months], "anchor_index": ref / base * 100,
                  "anchor": " | ".join(anchor["keywords"])}).to_csv(os.path.join(a.out, "기준묶음_월별.csv"), index=False, encoding="utf-8-sig")
    print(f"저장: 묶음 {grp.keyword_group_id.nunique()}개 x {len(months)}개월. 요청 간 배율 {min(k.values()):.3f}~{max(k.values()):.3f}, "
          f"기준 조각 모양 상관 최소 {shape_corr:.4f}", flush=True)


if __name__ == "__main__":
    main()
