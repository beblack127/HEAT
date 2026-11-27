"""Fill 2023 monthly station ridership files from Seoul OA-12914."""
import csv,io,re,urllib.request,urllib.parse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
html=urllib.request.urlopen('https://data.seoul.go.kr/dataList/OA-12914/F/1/datasetView.do').read().decode('utf-8')
matches=re.findall(r'title="(CARD_SUBWAY_MONTH_(2023\d{2})\.csv)" onclick="javascript:downloadFile\(\'(\d+)\'\)',html)
assert len({month for _,month,_ in matches})==12
out=ROOT/'data/subway_raw';out.mkdir(exist_ok=True)
for name,month,seq in matches:
    path=out/name
    if path.exists():continue
    form=urllib.parse.urlencode({'infId':'OA-12914','infSeq':'3','seq':seq,'seqNo':seq}).encode()
    req=urllib.request.Request('https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false',form,headers={'User-Agent':'Mozilla/5.0'})
    data=urllib.request.urlopen(req,timeout=60).read()
    if len(data)<50000:raise ValueError((name,len(data)))
    path.write_bytes(data)
    print(name,len(data),flush=True)
