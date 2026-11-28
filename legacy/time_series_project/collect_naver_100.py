"""Collect 69 comparison areas and combine them with the verified initial 31."""
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError

from collect_naver_areas import group, request
from naver_search_test import read_credentials
from naver_100_areas import AREAS_100

ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'data/naver/areas_20261106T102026Z/monthly_search_interest_31areas.csv'
NEW = [(name, term) for name, term, kind in AREAS_100 if kind == 'comparison_69']
ANCHOR = ('홍대', '홍대')

def main():
    prior=sorted((ROOT/'data/naver').glob('areas_100_*/raw_batch_05.json'))
    if prior and not (prior[-1].parent/'monthly_search_interest_100areas.csv').exists():
        out=prior[-1].parent
    else:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        out=ROOT/'data/naver'/f'areas_100_{stamp}'
        out.mkdir(parents=True,exist_ok=False)
    client_id=client_secret=None
    with OLD.open(encoding='utf-8-sig',newline='') as f: old=list(csv.DictReader(f))
    old_anchor={r['month']:float(r['search_interest']) for r in old if r['search_area']=='홍대'}
    batches=[NEW[i:i+4] for i in range(0,len(NEW),4)]
    collected=[]; manifest={'started_at_utc':out.name.removeprefix('areas_100_'),'source_31':str(OLD),'batches':[]}
    months=[f'{year}-{month:02}' for year in (2023,2024,2025) for month in range(1,13)]
    for index,batch in enumerate(batches,1):
        payload={'startDate':'2023-01-01','endDate':'2025-12-31','timeUnit':'month',
                 'keywordGroups':[group(*x) for x in [ANCHOR,*batch]]}
        raw_path=out/f'raw_batch_{index:02}.json'
        if raw_path.exists():
            cached=json.loads(raw_path.read_text(encoding='utf-8'))
            if cached['request']!=payload:raise SystemExit(f'배치 {index}: 기존 요청 조건 불일치')
            response=cached['response']
        else:
            if client_id is None:
                client_id,client_secret=read_credentials()
                if not client_id or not client_secret or any(ord(c)<33 or ord(c)>126 for c in client_id+client_secret):
                    raise SystemExit('Client ID와 Client Secret을 다시 확인하세요.')
            try: response=request(payload,client_id,client_secret)
            except (HTTPError,URLError,ValueError) as exc:
                manifest['error']={'batch':index,'type':type(exc).__name__,'status':getattr(exc,'code',None)}
                (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
                raise SystemExit(f'배치 {index} 중단. 완료된 원본은 {out}에 보존됨.')
        if len(response.get('results',[]))!=len(batch)+1:
            raise SystemExit(f'배치 {index}: 결과 그룹 수 불일치')
        if not raw_path.exists():raw_path.write_text(json.dumps({'request':payload,'response':response},ensure_ascii=False,indent=2),encoding='utf-8')
        anchor=next((x for x in response['results'] if x['title']=='홍대'),None)
        if anchor is None or len(anchor['data'])!=36:
            raise SystemExit(f'배치 {index}: 공통 기준어 누락')
        new_anchor={x['period'][:7]:float(x['ratio']) for x in anchor['data']}
        # The prior batch anchor must agree; never silently merge incompatible scales.
        max_gap=max(abs(old_anchor[m]-new_anchor[m]) for m in old_anchor)
        if max_gap>0.001:
            raise SystemExit(f'배치 {index}: 이전 수집과 척도 차이 발생 (홍대 최대 차이 {max_gap:.3f}). 원본만 저장함.')
        for result in response['results']:
            if result['title']=='홍대':continue
            observed={x['period'][:7]:x['ratio'] for x in result.get('data',[])}
            for month in months:
                collected.append({'month':month,'search_area':result['title'],
                                  'search_interest':observed.get(month,''),'keywords':' / '.join(result['keywords']),
                                  'source_batch':index})
        manifest['batches'].append({'index':index,'areas':[x[0] for x in batch],
                                    'anchor_max_gap':max_gap,'missing_search_months':sum(not r['search_interest'] for r in collected),'status':'ok'})
        (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        time.sleep(0.3)
    combined=[{'month':r['month'],'search_area':r['search_area'],
               'search_interest':r['search_interest'],'keywords':r['keywords'],
               'source_batch':'prior_31'} for r in old]+collected
    assert len(combined)==100*36
    assert len({(r['search_area'],r['month']) for r in combined})==len(combined)
    with (out/'monthly_search_interest_100areas.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['month','search_area','search_interest','keywords','source_batch'])
        writer.writeheader();writer.writerows(sorted(combined,key=lambda x:(x['search_area'],x['month'])))
    with (out/'area_list_100.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['search_area','search_term','sample_group','admin_dong_code','mapping_status'])
        writer.writerows((name,term,kind,'','unverified') for name,term,kind in AREAS_100)
    manifest['completed_at_utc']=datetime.now(timezone.utc).isoformat()
    manifest['rows']=len(combined)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(out,flush=True)

if __name__=='__main__':main()
