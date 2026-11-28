"""Download and summarize 2023-25 monthly Seoul dong population and arrivals.

Raw monthly archives are streamed to OS temp and removed after aggregation.
The resulting dong statistics are context, never place visitor counts.
"""
from __future__ import annotations
import csv,hashlib,io,json,os,sys,tempfile,time,urllib.parse,urllib.request,zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/context_monthly'
OUT.mkdir(parents=True,exist_ok=True)
BRIDGE=ROOT/'data/place_dong_crosswalk/place_to_admin_dong_overlap.csv'
with BRIDGE.open(encoding='utf-8-sig',newline='') as f:
    dongs={r['admin_dong_code_202506'] for r in csv.DictReader(f) if r['selected_by_5pct_rule']=='1'}
assert len(dongs)==40
URL='https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false'
SOURCES={'living':('OA-14991','3'),'mobility':('OA-22298','1')}

def download(kind,month):
    inf,seq=SOURCES[kind]
    code=month[2:] if kind=='living' else month
    form=urllib.parse.urlencode({'infId':inf,'infSeq':seq,'seq':code,'seqNo':code}).encode()
    req=urllib.request.Request(URL,form,headers={'User-Agent':'Mozilla/5.0','Referer':f'https://data.seoul.go.kr/dataList/{inf}/F/1/datasetView.do'})
    tmp=Path(tempfile.gettempdir())/f'codex_seoul_{kind}_{month}.zip'
    if not tmp.exists():
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req,timeout=180) as response,tmp.open('wb') as f:
                    while block:=response.read(1024*1024):f.write(block)
                break
            except Exception:
                tmp.unlink(missing_ok=True)
                if attempt==2:raise
                time.sleep(2**attempt)
    if not zipfile.is_zipfile(tmp):raise ValueError(f'Invalid archive: {kind} {month}')
    return tmp

def living_month(month,path):
    sums=defaultdict(lambda:[0.0,0,0.0,0,0.0,0,0.0,0])
    with zipfile.ZipFile(path) as z:
        assert len(z.namelist())==1
        with z.open(z.namelist()[0]) as f:
            encoding='utf-8-sig' if f.peek(3)[:3]==b'\xef\xbb\xbf' else 'cp949'
            r=csv.reader(io.TextIOWrapper(f,encoding=encoding,newline=''))
            h=next(r)
            assert h[1:4]==['시간대구분','행정동코드','총생활인구수']
            for row in r:
                if len(row)<4 or row[2] not in dongs:continue
                try:v=float(row[3]);hour=int(row[1])
                except ValueError:continue
                a=sums[row[2]];a[0]+=v;a[1]+=1
                if 11<=hour<=21:a[2]+=v;a[3]+=1
                if hour<=6:a[4]+=v;a[5]+=1
                if 11<=hour<=21 and __import__('datetime').date(int(row[0][:4]),int(row[0][4:6]),int(row[0][6:8])).weekday()>=5:
                    a[6]+=v;a[7]+=1
    records=[]
    for code,a in sums.items():
        records.append({'month':month[:4]+'-'+month[4:],'admin_dong_code':code,
                        'living_all_hour_mean':a[0]/a[1], 'living_day_mean':a[2]/a[3] if a[3] else '',
                        'living_night_mean':a[4]/a[5] if a[5] else '',
                        'living_weekend_day_mean':a[6]/a[7] if a[7] else '',
                        'living_hour_records':a[1]})
    return records

def mobility_month(month,path):
    sums=defaultdict(lambda:[0.0,0.0,0.0,0])
    dates=set()
    with zipfile.ZipFile(path) as z:
        for member in z.namelist():
            if not member.lower().endswith('.csv'):continue
            with z.open(member) as f:
                r=csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig',newline=''))
                assert {'d_admdong_cd','move_purpose','total_cnt','etl_ymd'}<=set(r.fieldnames or [])
                for row in r:
                    if row['d_admdong_cd'] not in dongs:continue
                    dates.add(row['etl_ymd'])
                    try:v=float(row['total_cnt'])
                    except (ValueError,TypeError):continue
                    a=sums[row['d_admdong_cd']]
                    a[0]+=v;a[3]+=1
                    if row['move_purpose'] in ('4','5'):a[1]+=v
                    if row['move_purpose'] in ('4','5','7'):a[2]+=v
    n_days=len(dates)
    if n_days<27:raise ValueError(f'Incomplete movement month {month}: {n_days} days')
    return [{'month':month[:4]+'-'+month[4:],'admin_dong_code':code,
             'mobility_all_arrivals_daily':a[0]/n_days,
             'mobility_shop_tour_daily':a[1]/n_days,
             'mobility_shop_tour_other_daily':a[2]/n_days,
             'mobility_records':a[3],'days':n_days} for code,a in sums.items()]

def write(kind,month,records,source_size):
    path=OUT/f'{kind}_{month}.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(sorted(records,key=lambda x:x['admin_dong_code']))
    if len(records)<35:raise ValueError(f'Too few dongs {kind} {month}: {len(records)}')
    return {'kind':kind,'month':month,'dongs':len(records),'source_bytes':source_size,'summary_file':str(path.relative_to(ROOT))}

def work(kind,month):
    target=OUT/f'{kind}_{month}.csv'
    if target.exists():
        with target.open(encoding='utf-8-sig',newline='') as f:n=sum(1 for _ in csv.DictReader(f))
        if n>=35:return {'kind':kind,'month':month,'dongs':n,'summary_file':str(target.relative_to(ROOT)),'cached':True}
    sample=ROOT/f'tmp/source_samples/{kind}_{month[2:] if kind=="living" else month}.zip'
    path=sample if sample.exists() else download(kind,month)
    try:
        records=living_month(month,path) if kind=='living' else mobility_month(month,path)
        return write(kind,month,records,path.stat().st_size)
    finally:
        if path!=sample:path.unlink(missing_ok=True)

if __name__=='__main__':
    months=[f'{y}{m:02d}' for y in (2023,2024,2025) for m in range(1,13)]
    jobs=[(kind,month) for kind in SOURCES for month in months]
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(work,*job):job for job in jobs}
        for f in as_completed(futures):
            item=f.result();results.append(item)
            print(item['kind'],item['month'],item['dongs'],flush=True)
            (OUT/'manifest.json').write_text(json.dumps({'sources':{'living':'https://data.seoul.go.kr/dataList/OA-14991/F/1/datasetView.do','mobility':'https://data.seoul.go.kr/dataList/OA-22298/F/1/datasetView.do'},'purpose_codes':{'4':'shopping','5':'tourism','7':'other'},'files':sorted(results,key=lambda x:(x['kind'],x['month']))},ensure_ascii=False,indent=2),encoding='utf-8')
