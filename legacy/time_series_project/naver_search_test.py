"""NAVER API HUB 검색어 트렌드 수집. 실행: python naver_search_test.py"""

import csv
import json
import sys
import warnings
from datetime import datetime, timezone
from getpass import GetPassWarning, getpass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "https://naverapihub.apigw.ntruss.com/search-trend/v1/search"

# 수집 기간과 검색어를 수정하려면 아래 설정을 변경하세요.
# 한 요청에서 최대 5개 그룹, 그룹당 최대 20개 검색어를 지정할 수 있습니다.
PAYLOAD = {
    "startDate": "2023-01-01",
    "endDate": "2025-12-31",
    "timeUnit": "month",
    "keywordGroups": [
        {"groupName": "성수", "keywords": ["성수 맛집", "성수 카페"]},
        {"groupName": "연남", "keywords": ["연남동 맛집", "연남동 카페"]},
        {"groupName": "홍대", "keywords": ["홍대 맛집", "홍대 카페"]},
    ],
}


def read_credentials():
    """Use the same masked GUI input as the successful diagnostic run."""
    if "--console" in sys.argv:
        with warnings.catch_warnings():
            warnings.simplefilter("error", GetPassWarning)
            return getpass("Client ID: ").strip(), getpass("Client Secret: ").strip()
    import tkinter as tk
    window = tk.Tk()
    window.title("NAVER 검색 데이터 수집")
    window.geometry("550x260")
    tk.Label(window, text="API HUB 키를 입력하세요. 키는 파일에 저장하지 않습니다.").pack(pady=12)
    tk.Label(window, text="Client ID").pack()
    id_entry = tk.Entry(window, width=60, show="*")
    id_entry.pack()
    tk.Label(window, text="Client Secret").pack()
    secret_entry = tk.Entry(window, width=60, show="*")
    secret_entry.pack()
    credentials = []
    def submit():
        credentials.extend([id_entry.get().strip(), secret_entry.get().strip()])
        id_entry.delete(0, "end")
        secret_entry.delete(0, "end")
        window.destroy()
    tk.Button(window, text="수집 시작", command=submit).pack(pady=16)
    window.bind("<Return>", lambda event: submit())
    id_entry.focus_set()
    window.mainloop()
    if not credentials:
        raise SystemExit("입력을 취소했습니다.")
    return credentials


def main():
    print("NAVER API HUB 검색어 트렌드 수집")
    print("기간: 2023-01 ~ 2025-12 / 월별 / 성수·연남·홍대")
    print("키는 입력 중 표시되지 않으며 파일에 저장하지 않습니다.")
    print("이 프로그램은 API를 1회 호출합니다.\n")

    # 숨김 입력을 지원하지 않는 환경에서는 평문 입력으로 전환하지 않습니다.
    try:
        client_id, client_secret = read_credentials()
    except GetPassWarning:
        raise SystemExit("숨김 입력을 지원하지 않습니다. PowerShell 또는 VS Code 터미널에서 실행하세요.")

    if not client_id or not client_secret:
        raise SystemExit("Client ID와 Client Secret을 모두 입력하세요.")
    for label, credential in (("Client ID", client_id), ("Client Secret", client_secret)):
        if any(ord(char) < 33 or ord(char) > 126 for char in credential):
            raise SystemExit(
                f"{label}에 공백 또는 제어문자가 포함되어 있습니다. "
                "키 값만 다시 복사하고, PowerShell에서 마우스 오른쪽 버튼으로 붙여넣어 보세요."
            )
        if credential.startswith(('\"', "'")) or credential.endswith(('\"', "'")):
            raise SystemExit(f"{label}의 양끝 따옴표를 제외하고 키 값만 입력하세요.")
    try:
        client_id.encode("ascii")
        client_secret.encode("ascii")
    except UnicodeEncodeError:
        raise SystemExit("인증 정보에 한글 등 잘못된 문자가 포함되어 있습니다. 키를 다시 복사하세요.")

    request = Request(
        API_URL,
        data=json.dumps(PAYLOAD, ensure_ascii=False).encode("utf-8"),
        headers={
            "X-NCP-APIGW-API-KEY-ID": client_id,
            "X-NCP-APIGW-API-KEY": client_secret,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    print("\n검색 데이터를 요청하고 있습니다...")
    try:
        with urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read(65536).decode("utf-8", errors="replace")
        # 인증 정보가 오류 메시지에 포함되더라도 표시하지 않습니다.
        for credential in (client_id, client_secret):
            error_body = error_body.replace(credential, "[REDACTED]")
        try:
            error_data = json.loads(error_body)
        except ValueError:
            error_data = None
        print(f"\nHTTP {exc.code} — 서버 상세 오류:")
        for header in ("Content-Type", "Content-Length", "Server", "Via", "X-NCP-APIGW-UUID"):
            value = exc.headers.get(header)
            if value:
                for credential in (client_id, client_secret):
                    value = value.replace(credential, "[REDACTED]")
                print(f"  {header}: {value}")
        if isinstance(error_data, dict):
            nested = error_data.get("error")
            details = nested if isinstance(nested, dict) else error_data
            shown = False
            for field in ("errorCode", "errorMessage", "message", "details", "errMsg", "errId"):
                if field in details:
                    print(f"  {field}: {str(details[field])[:2000]}")
                    shown = True
            if not shown:
                print("  표준 오류 설명이 없습니다.")
        else:
            print(error_body[:2000] or "  서버가 상세 내용을 반환하지 않았습니다.")
        hints = {
            400: "위 서버 상세 오류를 확인하세요. 400 코드만으로 원인을 확정할 수 없습니다.",
            401: "API HUB의 Client ID/Secret과 검색어 트렌드 사용 권한을 확인하세요.",
            403: "애플리케이션의 접근 제한 및 API 이용 설정을 확인하세요.",
            429: "호출 한도를 확인한 후 나중에 다시 실행하세요.",
        }
        raise SystemExit(f"HTTP {exc.code}: {hints.get(exc.code, '콘솔의 API 상태를 확인하세요.')}")
    except (URLError, TimeoutError):
        raise SystemExit("네트워크 연결에 실패했거나 응답 시간이 초과되었습니다. 연결 상태를 확인하세요.")
    except (ValueError, UnicodeDecodeError):
        raise SystemExit("API 응답을 해석할 수 없습니다. 콘솔의 서비스 상태를 확인하세요.")

    if not isinstance(result, dict) or not isinstance(result.get("results"), list):
        raise SystemExit("예상한 results 목록이 없는 응답입니다. API 설정을 확인하세요.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = Path(__file__).resolve().parent / "data" / "naver" / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    # 요청 헤더는 저장하지 않습니다.
    raw = {"collected_at_utc": timestamp, "request": PAYLOAD, "response": result}
    (output_dir / "raw.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    count = 0
    csv_path = output_dir / "monthly_search_interest.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["month", "search_area", "search_interest", "keywords"])
        for group in result["results"]:
            observations = group.get("data", [])
            print(f"  {group['title']}: {len(observations)}개 관측값")
            for observation in observations:
                writer.writerow([
                    observation["period"][:7], group["title"],
                    observation["ratio"], " / ".join(group["keywords"]),
                ])
                count += 1

    print(f"\n완료: {count}행 저장")
    print(f"CSV: {csv_path}")
    print(f"원본: {output_dir / 'raw.json'}")
    if count == 0:
        print("관측값이 없습니다. 검색어와 원본 응답을 확인하세요.")
    print("\n주의: search_interest는 실제 검색 건수가 아닌 상대지수입니다.")
    print("서로 다른 요청의 지수를 보정 없이 지역 간 순위로 비교하지 마세요.")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n입력을 취소했습니다.")
        sys.exit(1)
