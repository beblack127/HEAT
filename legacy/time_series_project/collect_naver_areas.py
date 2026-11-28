"""Collect monthly Naver search interest for named Seoul commercial areas.

Run with the bundled Python. Credentials are entered in a masked window and
are held only in process memory. Search areas are provisional, not dong codes.
"""
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from naver_search_test import API_URL, read_credentials

ROOT = Path(__file__).resolve().parent
AREAS = [
    ("홍대", "홍대"), ("성수", "성수"), ("연남", "연남동"),
    ("합정", "합정"), ("망원", "망원동"), ("상수", "상수"),
    ("을지로", "을지로"), ("익선동", "익선동"), ("서촌", "서촌"),
    ("북촌", "북촌"), ("삼청동", "삼청동"), ("신사", "신사동"),
    ("압구정", "압구정"), ("청담", "청담동"), ("한남", "한남동"),
    ("이태원", "이태원"), ("해방촌", "해방촌"), ("문래", "문래동"),
    ("여의도", "여의도"), ("강남", "강남역"), ("잠실", "잠실"),
    ("건대", "건대입구"), ("혜화", "혜화"), ("샤로수길", "샤로수길"),
    ("노원", "노원역"), ("신촌", "신촌"), ("왕십리", "왕십리"),
    ("가로수길", "가로수길"), ("경리단길", "경리단길"),
    ("방이", "방이동"), ("불광", "불광역"),
]

def group(name, term):
    return {"groupName": name, "keywords": [f"{term} 맛집", f"{term} 카페"]}

def request(payload, client_id, client_secret):
    req = Request(API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"X-NCP-APIGW-API-KEY-ID": client_id,
                 "X-NCP-APIGW-API-KEY": client_secret,
                 "Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))

def main():
    client_id, client_secret = read_credentials()
    if not client_id or not client_secret:
        raise SystemExit("인증 정보를 입력하세요.")
    if any(ord(c) < 33 or ord(c) > 126 for c in client_id + client_secret):
        raise SystemExit("키에 공백이나 제어문자가 포함되어 있습니다.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "data" / "naver" / f"areas_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    anchor = AREAS[0]
    targets = AREAS[1:]
    batches = [targets[i:i+4] for i in range(0, len(targets), 4)]
    rows = []
    manifest = {"collected_at_utc": stamp, "period": ["2023-01-01", "2025-12-31"],
                "note": "Named commercial areas; administrative-dong mapping not yet verified.",
                "anchor": anchor[0], "batches": []}
    try:
        for index, batch in enumerate(batches, 1):
            payload = {"startDate": "2023-01-01", "endDate": "2025-12-31",
                       "timeUnit": "month", "keywordGroups": [group(*x) for x in [anchor, *batch]]}
            response = request(payload, client_id, client_secret)
            if not isinstance(response.get("results"), list):
                raise ValueError(f"batch {index}: results 없음")
            (out / f"raw_batch_{index:02}.json").write_text(
                json.dumps({"request": payload, "response": response}, ensure_ascii=False, indent=2),
                encoding="utf-8")
            for result in response["results"]:
                if len(result.get("data", [])) != 36:
                    raise ValueError(f"batch {index}, {result.get('title')}: 36개월 미만")
                for item in result["data"]:
                    rows.append((index, item["period"][:7], result["title"],
                                 item["ratio"], " / ".join(result["keywords"])))
            manifest["batches"].append({"index": index, "areas": [x[0] for x in batch], "status": "ok"})
            (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{index}/{len(batches)} 완료: {', '.join(x[0] for x in batch)}", flush=True)
            time.sleep(0.3)
    except (HTTPError, URLError, ValueError) as exc:
        print(f"수집 중단: {type(exc).__name__}. 완료된 원본은 {out}에 보존했습니다.", flush=True)
        raise SystemExit(1)
    with (out / "monthly_search_interest_raw.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["batch", "month", "search_area", "ratio", "keywords"])
        writer.writerows(rows)
    print(f"완료: {len(AREAS)}개 지역, {len(rows)}행. 저장 위치: {out}", flush=True)
    print("각 배치의 ratio는 별도 상대척도입니다. 공통 기준어 보정 전에 합산·순위화하지 마세요.", flush=True)

if __name__ == "__main__":
    main()
