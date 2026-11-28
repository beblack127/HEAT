"""Download Seoul commercial trade-area quarterly footfall CSVs, 2023-2025."""
from __future__ import annotations
import csv,io,json,time,urllib.parse,urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/reference/commercial_footfall'
OUT.mkdir(parents=True,exist_ok=True)
URL='https://datafile.seoul.go.kr/bigfile/iot/sheet/csv/download.do'
HEAD={'Referer':'https://data.seoul.go.kr/dataList/OA-15568/S/1/datasetView.do','User-Agent':'Mozilla/5.0'}

def fetch(q):
    path=OUT/f'footfall_{q}.csv'
    if not path.exists():
        form={'srvType':'S','infId':'OA-15568','serviceKind':'0','pageNo':'1',
              'gridTotalCnt':'1650','ssUserId':'SAMPLE_VIEW','strWhere':'',
              'strOrderby':'STDR_YYQU_CD DESC','filterCol':'STDR_YYQU_CD','txtFilter':q}
        req=urllib.request.Request(URL,urllib.parse.urlencode(form).encode(),headers=HEAD)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req,timeout=60) as response: data=response.read()
                break
            except Exception:
                if attempt==2:raise
                time.sleep(2**attempt)
        rows=list(csv.reader(io.StringIO(data.decode('cp949'))))
        if len(rows)<1000 or rows[0][0]!='기준_년분기_코드' or {r[0] for r in rows[1:]}!={q}:
            raise ValueError(f'Unexpected data for {q}: rows={len(rows)}')
        path.write_bytes(data)
    with path.open(encoding='cp949',newline='') as f:
        rows=list(csv.reader(f))
    return {'quarter':q,'rows':len(rows)-1,'bytes':path.stat().st_size,'file':str(path.relative_to(ROOT))}

if __name__=='__main__':
    quarters=[f'{y}{q}' for y in (2023,2024,2025) for q in (1,2,3,4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        result=sorted((f.result() for f in as_completed([pool.submit(fetch,q) for q in quarters])),key=lambda x:x['quarter'])
    (OUT/'manifest.json').write_text(json.dumps({'source':'https://data.seoul.go.kr/dataList/OA-15568/S/1/datasetView.do','files':result},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True))
