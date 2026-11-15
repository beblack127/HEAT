"""서울 열린데이터광장 파일 받기. 사용: python3 seoul_dl.py INF_ID OUTDIR REGEX  (파일 이름이 REGEX와 맞는 것만)"""
import re, sys, os, subprocess, time
inf, out, pat = sys.argv[1], sys.argv[2], re.compile(sys.argv[3])
os.makedirs(out, exist_ok=True)
page = f"https://data.seoul.go.kr/dataList/{inf}/{'S' if inf=='OA-12914' else 'F'}/1/datasetView.do"
cj = os.path.join(out, ".cj")
for k in range(6):   # 목록 페이지가 가끔 끊긴다
    html = subprocess.run(["curl", "-sS", "-c", cj, "-b", cj, "--max-time", "60", page], capture_output=True, text=True).stdout
    m = re.search(r'name="frmFile".*?name="infSeq" value="(\d+)"', html, re.S)
    if m: break
    time.sleep(5 * (k + 1))
else:
    sys.exit(f"목록 페이지를 못 받음: {page}")
infseq = m.group(1)
rows = re.findall(r'title="([^"]+\.(?:zip|csv))" onclick="javascript:downloadFile\(\'([^\']+)\'\);"', html)
todo = [(n, s) for n, s in rows if pat.search(n)]
print(inf, "infSeq", infseq, "files", len(todo), flush=True)
for name, seq in sorted(set(todo)):
    fp = os.path.join(out, name)
    if os.path.exists(fp) and os.path.getsize(fp) > 1000: continue
    for k in range(4):
        r = subprocess.run(["curl", "-sS", "-c", cj, "-b", cj, "--max-time", "900", "-X", "POST", "-e", page,
                            "https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false",
                            "-d", f"infId={inf}&seq={seq}&infSeq={infseq}&seqNo=", "-o", fp + ".part"], capture_output=True, text=True)
        ok = os.path.exists(fp + ".part") and os.path.getsize(fp + ".part") > 1000
        if ok: os.replace(fp + ".part", fp); print("ok", name, os.path.getsize(fp), flush=True); break
        print("retry", name, r.stderr[:200], flush=True); time.sleep(5 * (k + 1))
